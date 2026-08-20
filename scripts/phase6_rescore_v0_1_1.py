"""One-time Phase 6 rescore for the terminal-record-separator parser defect.

The original held-out generations are immutable.  This module never invokes model
generation.  It removes only the frozen terminal serialization delimiter, reparses the
stored text, and recomputes confidence for newly valid rows from the exact frozen token
sequence.  The corrected report remains bound to both the original report and the
single held-out access record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Literal

import torch
from pydantic import Field, model_validator

from reactorbench.evaluation.config import Phase5Config, Phase6Config, load_phase6_config
from reactorbench.evaluation.data import ExperimentExample, materialize_phase6_data
from reactorbench.evaluation.decoding import DecodedPrediction, _prediction
from reactorbench.model import TransformerLM, load_checkpoint, resolve_project_path
from reactorbench.schemas.base import ContractModel, canonical_json_bytes, canonical_sha256
from reactorbench.tokenizer import BOS_ID, PAD_ID, ProjectTokenizer
from reactorbench.training.main import (
    MAX_PREDICTIONS_BYTES,
    TEST_SPLITS,
    ExperimentDecodedPrediction,
    ModelSplitEvaluation,
    Phase6EvaluationReport,
    Phase6SelectionReport,
    _acceptance_checks,
    _classification_results,
    _comparison_prediction,
    _event_order_records,
    _failure_gallery,
    _golden_examples,
    _heldout_access_path,
    _load_inputs,
    _main_metrics,
    _read_comparison_predictions,
    _read_predictions,
    _sha256,
    _source_commit,
    verify_phase6_evaluation,
    verify_phase6_selection,
)
from reactorbench.training.pilot import Phase5RunReport

CORRECTION_SUFFIX = "-rescore-v0.1.1"
CORRECTION_REASON: Literal["strip_exact_frozen_record_separator_before_strict_json_parse"] = (
    "strip_exact_frozen_record_separator_before_strict_json_parse"
)


class Phase6EvaluationCorrectionRecord(ContractModel):
    correction_version: Literal["0.1.1"] = "0.1.1"
    run_status: Literal["phase6_evaluation_rescored"] = "phase6_evaluation_rescored"
    correction_source_commit: str = Field(pattern=r"^[0-9a-f]{7,64}$")
    correction_reason: Literal["strip_exact_frozen_record_separator_before_strict_json_parse"] = (
        CORRECTION_REASON
    )
    original_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    original_report_raw_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    original_predictions_raw_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    original_comparison_predictions_raw_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    original_golden_predictions_raw_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    heldout_access_record_raw_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    corrected_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    corrected_report_raw_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    new_model_generation: Literal[False] = False
    generated_text_unchanged: Literal[True] = True
    valid_prediction_confidence_recomputed: Literal[True] = True
    checksum_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def checksum_matches(self) -> Phase6EvaluationCorrectionRecord:
        expected = canonical_sha256(
            self.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        if self.checksum_sha256 != expected:
            raise ValueError("Phase 6 correction record checksum mismatch")
        return self


def _correction_directory(config: Phase6Config, project_root: Path) -> Path:
    root = resolve_project_path(project_root, config.phase6.run_root, must_exist=False)
    return root / f"{config.phase6.run_name}{CORRECTION_SUFFIX}"


def _original_directory(config: Phase6Config, project_root: Path) -> Path:
    root = resolve_project_path(project_root, config.phase6.run_root, must_exist=True)
    directory = root / config.phase6.run_name
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("original Phase 6 evaluation directory is missing or unsafe")
    return directory


def _prediction_jsonl(values: tuple[ContractModel, ...]) -> bytes:
    payload = b"".join(
        canonical_json_bytes(value.model_dump(mode="json", round_trip=True)) + b"\n"
        for value in values
    )
    if not 0 < len(payload) <= MAX_PREDICTIONS_BYTES:
        raise ValueError("corrected prediction artifact is empty or oversized")
    return payload


def _confidence_for_valid_predictions(
    model: TransformerLM,
    tokenizer: ProjectTokenizer,
    examples: tuple[ExperimentExample, ...],
    predictions: tuple[DecodedPrediction, ...],
    *,
    prompt_prefix: str,
    target_prefix: str,
    maximum_generated_tokens: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    """Teacher-force only stored greedy tokens; never create or replace a token."""

    if not predictions:
        return {}
    by_id = {item.example_id: item for item in examples}
    if len(by_id) != len(examples):
        raise ValueError("confidence examples contain duplicate IDs")
    prepared: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    for prediction in predictions:
        example = by_id.get(prediction.example_id)
        if example is None:
            raise ValueError("confidence prediction references an unknown example")
        prefix_text = (
            f"{prompt_prefix}\nTASK={example.task_name.value}\n"
            f"{example.prompt_text}\n{target_prefix}\n"
        )
        prefix = tokenizer.encode(prefix_text, add_bos=True, add_eos=False)
        maximum_prefix = model.config.context_length - maximum_generated_tokens
        if maximum_prefix < 2:
            raise ValueError("confidence rescore has no valid prompt boundary")
        truncated = len(prefix) > maximum_prefix
        if truncated:
            prefix = (BOS_ID, *prefix[-(maximum_prefix - 1) :])
        if truncated != prediction.prompt_truncated:
            raise ValueError("stored prompt truncation differs from deterministic reconstruction")
        generated = tokenizer.encode(prediction.generated_text, add_bos=False, add_eos=False)
        if len(generated) != prediction.generated_token_count or not generated:
            raise ValueError("stored generated text does not round-trip to its exact token count")
        sequence = (*prefix, *generated)
        if len(sequence) > model.config.context_length:
            raise ValueError("stored generation exceeds the frozen model context")
        prepared.append((prediction.example_id, prefix, generated))

    result: dict[str, float] = {}
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(prepared), batch_size):
                chunk = prepared[start : start + batch_size]
                sequences = tuple((*prefix, *generated) for _identifier, prefix, generated in chunk)
                width = max(len(sequence) - 1 for sequence in sequences)
                input_ids = torch.full((len(chunk), width), PAD_ID, dtype=torch.long, device=device)
                mask = torch.zeros((len(chunk), width), dtype=torch.bool, device=device)
                for row, sequence in enumerate(sequences):
                    visible = sequence[:-1]
                    input_ids[row, : len(visible)] = torch.tensor(
                        visible, dtype=torch.long, device=device
                    )
                    mask[row, : len(visible)] = True
                logits = model(input_ids, mask)
                log_probabilities = torch.log_softmax(logits, dim=-1)
                for row, (identifier, prefix, generated) in enumerate(chunk):
                    first = len(prefix) - 1
                    last = first + len(generated)
                    targets = torch.tensor(generated, dtype=torch.long, device=device)
                    selected = log_probabilities[row, first:last, :]
                    if selected.shape[0] != len(generated):
                        raise RuntimeError("confidence token window is misaligned")
                    if not torch.equal(selected.argmax(dim=-1), targets):
                        raise ValueError("stored token is not the frozen model's greedy choice")
                    mean_log_probability = selected.gather(1, targets[:, None]).mean()
                    confidence = math.exp(float(mean_log_probability.item()))
                    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                        raise ValueError("recomputed confidence is not finite in [0,1]")
                    result[identifier] = confidence
    finally:
        model.train(was_training)
    if len(result) != len(predictions):
        raise RuntimeError("confidence rescore did not cover every valid prediction")
    return result


def _reparse_predictions(
    model: TransformerLM,
    tokenizer: ProjectTokenizer,
    examples: tuple[ExperimentExample, ...],
    predictions: tuple[DecodedPrediction, ...],
    config: Phase6Config,
    *,
    prompt_prefix: str,
    target_prefix: str,
    record_separator: str,
    batch_size: int,
    device: torch.device,
) -> tuple[DecodedPrediction, ...]:
    by_id = {item.example_id: item for item in examples}
    if set(by_id) != {item.example_id for item in predictions}:
        raise ValueError("stored predictions do not exactly match their example inventory")
    provisional = tuple(
        _prediction(
            by_id[item.example_id],
            generated_text=item.generated_text,
            generated_token_count=item.generated_token_count,
            prompt_truncated=item.prompt_truncated,
            generation_truncated=item.generation_truncated,
            confidence=0.0,
            record_separator=record_separator,
        )
        for item in predictions
    )
    newly_valid = tuple(item for item in provisional if item.schema_valid)
    confidence = _confidence_for_valid_predictions(
        model,
        tokenizer,
        examples,
        newly_valid,
        prompt_prefix=prompt_prefix,
        target_prefix=target_prefix,
        maximum_generated_tokens=config.decoder.maximum_generated_target_tokens,
        batch_size=batch_size,
        device=device,
    )
    corrected = tuple(
        _prediction(
            by_id[item.example_id],
            generated_text=item.generated_text,
            generated_token_count=item.generated_token_count,
            prompt_truncated=item.prompt_truncated,
            generation_truncated=item.generation_truncated,
            confidence=confidence.get(item.example_id, 0.0),
            record_separator=record_separator,
        )
        for item in predictions
    )
    return tuple(sorted(corrected, key=lambda item: item.example_id))


def _model_result_with_predictions(
    original: ModelSplitEvaluation,
    records: tuple[ExperimentExample, ...],
    predictions: tuple[DecodedPrediction, ...],
) -> ModelSplitEvaluation:
    return ModelSplitEvaluation(
        experiment_id=original.experiment_id,
        split_name=original.split_name,
        sample_count=original.sample_count,
        scored_sample_count=original.scored_sample_count,
        insufficient_context_by_design=original.insufficient_context_by_design,
        language_model=original.language_model,
        classification=_classification_results(records, predictions),
    )


def _load_evaluation_models(
    config: Phase6Config,
    project_root: Path,
    selection: Phase6SelectionReport,
    tokenizer: ProjectTokenizer,
    phase5: Phase5Config,
    phase5_report: Phase5RunReport,
) -> dict[str, tuple[TransformerLM, int, torch.device]]:
    selection_dir = (
        resolve_project_path(project_root, config.phase6.run_root, must_exist=True)
        / f"{config.phase6.run_name}-selection"
    )
    models: dict[str, tuple[TransformerLM, int, torch.device]] = {}
    for experiment_id, directory_name, phase6_result in (
        ("E3_main_transformer", "main-checkpoint", selection.results[0]),
        ("E5_renderer_diversity_ablation", "renderer-ablation-checkpoint", selection.results[1]),
        ("E6_abstention_ablation", "abstention-ablation-checkpoint", selection.results[2]),
    ):
        device = torch.device(phase6_result.device)
        model, _manifest = load_checkpoint(
            selection_dir / directory_name,
            expected_manifest_sha256=phase6_result.checkpoint_manifest_sha256,
            expected_tokenizer_sha256=tokenizer.manifest.checksum_sha256,
            device=device,
        )
        models[experiment_id] = (model, phase6_result.batch_size, device)

    phase5_run = (
        resolve_project_path(
            project_root,
            phase5.phase5.run_root,
            must_exist=True,
        )
        / phase5.phase5.run_name
    )
    for experiment_id, directory_name, pilot_result in (
        ("E2_smaller_transformer", "smaller-checkpoint", phase5_report.transformer_results[0]),
        ("E2_pilot_transformer", "pilot-checkpoint", phase5_report.transformer_results[1]),
    ):
        device = torch.device(pilot_result.device)
        model, _manifest = load_checkpoint(
            phase5_run / directory_name,
            expected_manifest_sha256=pilot_result.checkpoint_manifest_sha256,
            expected_tokenizer_sha256=tokenizer.manifest.checksum_sha256,
            device=device,
        )
        models[experiment_id] = (model, pilot_result.batch_size, device)
    return models


def _corrected_graph(
    config: Phase6Config,
    *,
    project_root: Path,
    original: Phase6EvaluationReport,
) -> tuple[
    tuple[DecodedPrediction, ...],
    tuple[ExperimentDecodedPrediction, ...],
    tuple[DecodedPrediction, ...],
    Phase6EvaluationReport,
]:
    selection = verify_phase6_selection(config, project_root=project_root)
    phase5, _phase4, phase5_report, verified, tokenizer, _candidate_sha = _load_inputs(
        config, project_root
    )
    data = materialize_phase6_data(
        verified,
        freeze=config.test_freeze,
        maximum_prompt_utf8_bytes=phase5.serialization.maximum_prompt_utf8_bytes,
    )
    all_records = tuple(record for split in TEST_SPLITS for record in data.by_split[split])
    golden_examples = _golden_examples(config, project_root)
    original_dir = _original_directory(config, project_root)
    original_main = _read_predictions(
        original_dir / "predictions.jsonl", expected_count=original.test_example_count
    )
    original_comparisons = _read_comparison_predictions(
        original_dir / "comparison-predictions.jsonl",
        expected_count=original.test_example_count * 5,
    )
    original_golden = _read_predictions(
        original_dir / "golden-predictions.jsonl", expected_count=original.golden_example_count
    )
    models = _load_evaluation_models(
        config, project_root, selection, tokenizer, phase5, phase5_report
    )
    prompt_prefix = phase5.serialization.prompt_prefix
    target_prefix = phase5.serialization.target_prefix
    record_separator = phase5.serialization.record_separator

    main_model, main_batch, main_device = models["E3_main_transformer"]
    corrected_main = _reparse_predictions(
        main_model,
        tokenizer,
        all_records,
        original_main,
        config,
        prompt_prefix=prompt_prefix,
        target_prefix=target_prefix,
        record_separator=record_separator,
        batch_size=main_batch,
        device=main_device,
    )
    corrected_golden = _reparse_predictions(
        main_model,
        tokenizer,
        golden_examples,
        original_golden,
        config,
        prompt_prefix=prompt_prefix,
        target_prefix=target_prefix,
        record_separator=record_separator,
        batch_size=main_batch,
        device=main_device,
    )

    originals_by_experiment = {
        experiment_id: tuple(
            item.prediction for item in original_comparisons if item.experiment_id == experiment_id
        )
        for experiment_id in {
            "E2_smaller_transformer",
            "E2_pilot_transformer",
            "E4_event_order_ablation",
            "E5_renderer_diversity_ablation",
            "E6_abstention_ablation",
        }
    }
    corrected_by_experiment: dict[str, tuple[DecodedPrediction, ...]] = {}
    event_records = tuple(
        record for split in TEST_SPLITS for record in _event_order_records(data.by_split[split])
    )
    corrected_by_experiment["E4_event_order_ablation"] = _reparse_predictions(
        main_model,
        tokenizer,
        event_records,
        originals_by_experiment["E4_event_order_ablation"],
        config,
        prompt_prefix=prompt_prefix,
        target_prefix=target_prefix,
        record_separator=record_separator,
        batch_size=main_batch,
        device=main_device,
    )
    for experiment_id in (
        "E2_smaller_transformer",
        "E2_pilot_transformer",
        "E5_renderer_diversity_ablation",
        "E6_abstention_ablation",
    ):
        model, batch_size, device = models[experiment_id]
        corrected_by_experiment[experiment_id] = _reparse_predictions(
            model,
            tokenizer,
            all_records,
            originals_by_experiment[experiment_id],
            config,
            prompt_prefix=prompt_prefix,
            target_prefix=target_prefix,
            record_separator=record_separator,
            batch_size=batch_size,
            device=device,
        )

    corrected_comparison = tuple(
        _comparison_prediction(experiment_id, prediction)
        for experiment_id in sorted(corrected_by_experiment)
        for prediction in corrected_by_experiment[experiment_id]
    )
    records_by_split = {split: data.by_split[split] for split in TEST_SPLITS}
    main_by_id = {item.example_id: item for item in corrected_main}
    comparison_by_key = {
        (item.experiment_id, item.prediction.example_id): item.prediction
        for item in corrected_comparison
    }
    corrected_model_results: list[ModelSplitEvaluation] = []
    for item in original.model_split_results:
        records = records_by_split[item.split_name]
        if item.experiment_id == "E3_main_transformer":
            predictions = tuple(main_by_id[record.example_id] for record in records)
        else:
            predictions = tuple(
                comparison_by_key[(item.experiment_id, record.example_id)] for record in records
            )
        corrected_model_results.append(_model_result_with_predictions(item, records, predictions))
    prediction_metrics = tuple(
        _main_metrics(
            split,
            records_by_split[split],
            tuple(main_by_id[record.example_id] for record in records_by_split[split]),
            config,
        )
        for split in TEST_SPLITS
    )
    checks = _acceptance_checks(
        config,
        baselines=original.baseline_results,
        model_results=tuple(corrected_model_results),
        prediction_metrics=prediction_metrics,
        selection_thresholds_passed=selection.selection_thresholds_passed,
    )
    main_payload = _prediction_jsonl(corrected_main)
    comparison_payload = _prediction_jsonl(corrected_comparison)
    golden_payload = _prediction_jsonl(corrected_golden)
    golden_by_id = {item.example_id: item for item in corrected_golden}
    golden_exact = sum(
        golden_by_id[record.example_id].predicted_target_json == record.target_text
        for record in golden_examples
    ) / len(golden_examples)
    draft = Phase6EvaluationReport.model_construct(
        source_commit=original.source_commit,
        phase6_config_sha256=original.phase6_config_sha256,
        selection_report_sha256=original.selection_report_sha256,
        golden_packet_sha256=original.golden_packet_sha256,
        golden_review_record_sha256=original.golden_review_record_sha256,
        heldout_access_count=1,
        heldout_access_after_selection=True,
        test_example_count=894,
        experiment_matrix=original.experiment_matrix,
        baseline_results=original.baseline_results,
        model_split_results=tuple(corrected_model_results),
        main_prediction_metrics=prediction_metrics,
        golden_case_count=15,
        golden_example_count=len(golden_examples),
        golden_exact_match_rate=golden_exact,
        acceptance_checks=checks,
        negative_results=tuple(name for name, passed in checks if not passed),
        failure_gallery=_failure_gallery(all_records, corrected_main),
        predictions_sha256=hashlib.sha256(main_payload).hexdigest(),
        comparison_predictions_sha256=hashlib.sha256(comparison_payload).hexdigest(),
        golden_predictions_sha256=hashlib.sha256(golden_payload).hexdigest(),
        checksum_sha256="0" * 64,
    )
    checksum = canonical_sha256(
        draft.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
    )
    report = Phase6EvaluationReport(
        **draft.model_dump(mode="python", exclude={"checksum_sha256"}),
        checksum_sha256=checksum,
    )
    return corrected_main, corrected_comparison, corrected_golden, report


def _correction_record(
    config: Phase6Config,
    *,
    project_root: Path,
    correction_source_commit: str,
    original: Phase6EvaluationReport,
    corrected: Phase6EvaluationReport,
    corrected_report_raw_sha256: str,
) -> Phase6EvaluationCorrectionRecord:
    original_dir = _original_directory(config, project_root)
    access_path = _heldout_access_path(config, project_root)
    draft = Phase6EvaluationCorrectionRecord.model_construct(
        correction_source_commit=correction_source_commit,
        original_report_sha256=original.checksum_sha256,
        original_report_raw_sha256=_sha256(original_dir / "report.json"),
        original_predictions_raw_sha256=_sha256(original_dir / "predictions.jsonl"),
        original_comparison_predictions_raw_sha256=_sha256(
            original_dir / "comparison-predictions.jsonl"
        ),
        original_golden_predictions_raw_sha256=_sha256(original_dir / "golden-predictions.jsonl"),
        heldout_access_record_raw_sha256=_sha256(access_path),
        corrected_report_sha256=corrected.checksum_sha256,
        corrected_report_raw_sha256=corrected_report_raw_sha256,
        new_model_generation=False,
        generated_text_unchanged=True,
        valid_prediction_confidence_recomputed=True,
        checksum_sha256="0" * 64,
    )
    checksum = canonical_sha256(
        draft.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
    )
    return Phase6EvaluationCorrectionRecord(
        **draft.model_dump(mode="python", exclude={"checksum_sha256"}),
        checksum_sha256=checksum,
    )


def run_phase6_evaluation_correction(
    config: Phase6Config,
    *,
    project_root: Path,
    correction_source_commit: str,
) -> Phase6EvaluationCorrectionRecord:
    if type(config) is not Phase6Config:
        raise TypeError("config must be an exact Phase6Config")
    correction_source_commit = _source_commit(correction_source_commit)
    original = verify_phase6_evaluation(config, project_root=project_root)
    output = _correction_directory(config, project_root)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Phase 6 correction output already exists")
    corrected_main, corrected_comparison, corrected_golden, report = _corrected_graph(
        config, project_root=project_root, original=original
    )
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        (temporary / "predictions.jsonl").write_bytes(_prediction_jsonl(corrected_main))
        (temporary / "comparison-predictions.jsonl").write_bytes(
            _prediction_jsonl(corrected_comparison)
        )
        (temporary / "golden-predictions.jsonl").write_bytes(_prediction_jsonl(corrected_golden))
        report_payload = (
            canonical_json_bytes(report.model_dump(mode="json", round_trip=True)) + b"\n"
        )
        (temporary / "report.json").write_bytes(report_payload)
        record = _correction_record(
            config,
            project_root=project_root,
            correction_source_commit=correction_source_commit,
            original=original,
            corrected=report,
            corrected_report_raw_sha256=hashlib.sha256(report_payload).hexdigest(),
        )
        (temporary / "correction-record.json").write_bytes(
            canonical_json_bytes(record.model_dump(mode="json", round_trip=True)) + b"\n"
        )
        os.rename(temporary, output)
        return record
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def verify_phase6_evaluation_correction(
    config: Phase6Config, *, project_root: Path
) -> Phase6EvaluationCorrectionRecord:
    if type(config) is not Phase6Config:
        raise TypeError("config must be an exact Phase6Config")
    original = verify_phase6_evaluation(config, project_root=project_root)
    directory = _correction_directory(config, project_root)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Phase 6 correction directory is missing or unsafe")
    expected_files = {
        "correction-record.json",
        "report.json",
        "predictions.jsonl",
        "comparison-predictions.jsonl",
        "golden-predictions.jsonl",
    }
    if {item.name for item in directory.iterdir()} != expected_files:
        raise ValueError("Phase 6 correction inventory is unexpected")
    record_payload = (directory / "correction-record.json").read_bytes()
    record = Phase6EvaluationCorrectionRecord.model_validate_json(record_payload)
    corrected_main, corrected_comparison, corrected_golden, report = _corrected_graph(
        config, project_root=project_root, original=original
    )
    stored_report = Phase6EvaluationReport.model_validate_json(
        (directory / "report.json").read_bytes()
    )
    if stored_report != report:
        raise ValueError("corrected Phase 6 report does not reconstruct exactly")
    if (
        _read_predictions(directory / "predictions.jsonl", expected_count=report.test_example_count)
        != corrected_main
    ):
        raise ValueError("corrected main predictions do not reconstruct exactly")
    if (
        _read_comparison_predictions(
            directory / "comparison-predictions.jsonl",
            expected_count=report.test_example_count * 5,
        )
        != corrected_comparison
    ):
        raise ValueError("corrected comparison predictions do not reconstruct exactly")
    if (
        _read_predictions(
            directory / "golden-predictions.jsonl", expected_count=report.golden_example_count
        )
        != corrected_golden
    ):
        raise ValueError("corrected golden predictions do not reconstruct exactly")
    expected_record = _correction_record(
        config,
        project_root=project_root,
        correction_source_commit=record.correction_source_commit,
        original=original,
        corrected=report,
        corrected_report_raw_sha256=_sha256(directory / "report.json"),
    )
    if record != expected_record:
        raise ValueError("Phase 6 correction record does not reconstruct exactly")
    return record


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="phase6-rescore-v0.1.1")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--correction-source-commit", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--config", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        project_root = Path.cwd().resolve(strict=True)
        config_path = args.config.resolve(strict=True)
        if not config_path.is_relative_to(project_root):
            raise ValueError("config must be inside the current project checkout")
        config = load_phase6_config(config_path)
        if args.command == "run":
            record = run_phase6_evaluation_correction(
                config,
                project_root=project_root,
                correction_source_commit=args.correction_source_commit,
            )
        else:
            record = verify_phase6_evaluation_correction(config, project_root=project_root)
        print(
            json.dumps(
                {
                    "corrected_report_sha256": record.corrected_report_sha256,
                    "new_model_generation": record.new_model_generation,
                    "run_status": record.run_status,
                },
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        return 0
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
