from __future__ import annotations

from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, Literal, cast

import pytest
import torch

from reactorbench.dataset.catalog import AliasFamily
from reactorbench.dataset.contracts import PromptContinuationTarget, PromptEvidenceTarget
from reactorbench.evaluation.baselines import _result
from reactorbench.evaluation.config import (
    TransformerTrainingConfig,
    load_phase5_config,
    load_phase6_config,
)
from reactorbench.evaluation.data import ExperimentData, ExperimentExample, Phase6ExperimentData
from reactorbench.evaluation.decoding import DecodedPrediction, _prediction
from reactorbench.evaluation.metrics import classification_metrics, language_model_metrics
from reactorbench.evaluation.serialization import TokenizedExample
from reactorbench.model import TransformerConfig, load_phase4_config
from reactorbench.schemas.base import canonical_json_bytes, canonical_sha256
from reactorbench.schemas.enums import (
    ActionLabel,
    DiagnosisStatus,
    EventType,
    EvidenceSlot,
    SplitName,
    TaskName,
)
from reactorbench.schemas.target import FaultDiagnosisTarget, NextActionTarget
from reactorbench.tokenizer import ProjectTokenizer
from reactorbench.training import cli as phase6_cli
from reactorbench.training import main as phase6_main
from reactorbench.training.cli import _parser
from reactorbench.training.main import (
    ExperimentDecodedPrediction,
    HeldoutAccessRecord,
    MainPredictionMetrics,
    ModelSplitEvaluation,
    Phase6ModelResult,
    Phase6SelectionReport,
    _event_order_records,
    _golden_examples,
    run_phase6_evaluation,
    run_phase6_selection,
    verify_phase6_evaluation,
    verify_phase6_selection,
)
from reactorbench.training.pilot import ValidationPoint


def _model_result(
    experiment_id: Literal[
        "E3_main_transformer",
        "E5_renderer_diversity_ablation",
        "E6_abstention_ablation",
    ],
) -> Phase6ModelResult:
    return Phase6ModelResult(
        experiment_id=experiment_id,
        device="cpu",
        parameter_count=32,
        train_example_count=8,
        validation_example_count=4,
        context_length=32,
        batch_size=2,
        training_steps=10,
        selected_step=10,
        initial_validation_nll=2.0,
        selected_validation_nll=0.2,
        validation_nll_reduction_fraction=0.9,
        final_training_nll=0.1,
        validation_curve=(
            ValidationPoint(step=0, target_nll=2.0),
            ValidationPoint(step=10, target_nll=0.2),
        ),
        elapsed_seconds=1.0,
        scored_target_tokens=100,
        target_tokens_per_second=100.0,
        process_peak_rss_bytes=1,
        mps_peak_current_allocated_bytes=0,
        mps_peak_driver_allocated_bytes=0,
        checkpoint_manifest_sha256="1" * 64,
        checkpoint_weights_sha256="2" * 64,
        checkpoint_size_bytes=10,
    )


def test_selection_report_is_checksum_bound_and_canonical() -> None:
    values = (
        _model_result("E3_main_transformer"),
        _model_result("E5_renderer_diversity_ablation"),
        _model_result("E6_abstention_ablation"),
    )
    draft = Phase6SelectionReport.model_construct(
        source_commit="abcdef0",
        phase6_config_sha256="3" * 64,
        phase5_report_sha256="4" * 64,
        dataset_candidate_sha256="5" * 64,
        tokenizer_manifest_sha256="6" * 64,
        training_inventory_sha256="7" * 64,
        test_records_materialized=False,
        test_predictions_generated=False,
        all_ablation_selection_complete=True,
        e7_status="not_applicable_no_compound_iid_train_rows",
        results=values,
        selection_thresholds_passed=True,
        checksum_sha256="0" * 64,
    )
    checksum = canonical_sha256(
        draft.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
    )
    report = Phase6SelectionReport(
        source_commit="abcdef0",
        phase6_config_sha256="3" * 64,
        phase5_report_sha256="4" * 64,
        dataset_candidate_sha256="5" * 64,
        tokenizer_manifest_sha256="6" * 64,
        training_inventory_sha256="7" * 64,
        test_records_materialized=False,
        test_predictions_generated=False,
        all_ablation_selection_complete=True,
        e7_status="not_applicable_no_compound_iid_train_rows",
        results=values,
        selection_thresholds_passed=True,
        checksum_sha256=checksum,
    )
    assert report.checksum_sha256 == checksum
    with pytest.raises(ValueError, match="checksum"):
        report.model_copy(update={"checksum_sha256": "0" * 64}, deep=True).model_validate(
            report.model_copy(update={"checksum_sha256": "0" * 64}).model_dump(mode="python")
        )


def test_event_order_ablation_reverses_only_same_tick_lines() -> None:
    record = ExperimentExample(
        example_id="example:order",
        split_name=SplitName.IID_TEST,
        task_name=TaskName.NEXT_ACTION,
        prompt_text="header\n[T+002] first\n[T+002] second\n[T+003] third\nfooter",
        target_text='{"immediate_action":"CONTINUE_MONITORING","task_name":"next_action"}',
        classification_label="CONTINUE_MONITORING",
        source_checksum_sha256="a" * 64,
    )
    changed = _event_order_records((record,))[0]
    assert changed.prompt_text.splitlines() == [
        "header",
        "[T+002] second",
        "[T+002] first",
        "[T+003] third",
        "footer",
    ]
    assert changed.target_text == record.target_text


def test_phase6_commands_are_explicit_two_gate_workflow() -> None:
    parser = _parser()
    for command in (
        "run-phase6-selection",
        "verify-phase6-selection",
        "run-phase6-evaluation",
        "verify-phase6-evaluation",
    ):
        arguments = [command, "--config", "phase6.toml"]
        if command.startswith("run-"):
            arguments.extend(("--source-commit", "abcdef0"))
        assert parser.parse_args(arguments).command == command


def test_approved_golden_packet_projects_sixty_examples() -> None:
    project_root = Path.cwd()
    config = load_phase6_config(project_root / "configs/experiments/phase6-main-v0.1.0.toml")
    record = project_root / config.phase6.golden_review_record_path
    if not record.exists():
        pytest.skip("local owner review record is not present in this checkout")
    examples = _golden_examples(config, project_root)
    assert len(examples) == 60
    assert len({example.example_id for example in examples}) == 60
    assert {example.task_name for example in examples} == {
        TaskName.FAULT_FAMILY,
        TaskName.EXTRACT_EVIDENCE,
        TaskName.NEXT_ACTION,
        TaskName.INCIDENT_SUMMARY,
    }


def _example(
    identifier: str,
    split: SplitName,
    task: TaskName = TaskName.NEXT_ACTION,
) -> ExperimentExample:
    target = NextActionTarget(immediate_action=ActionLabel.CONTINUE_MONITORING)
    return ExperimentExample(
        example_id=identifier,
        split_name=split,
        task_name=task,
        prompt_text="[T+000] fictional bounded observation",
        target_text=canonical_json_bytes(target.model_dump(mode="json", round_trip=True)).decode(
            "utf-8"
        ),
        classification_label="CONTINUE_MONITORING" if task is TaskName.NEXT_ACTION else None,
        source_checksum_sha256="a" * 64,
    )


def test_real_tiny_phase6_training_selects_and_reloads_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tokenized = (
        TokenizedExample(
            example_id="tiny",
            token_ids=(1, 4, 5, 2),
            target_mask=(False, False, True, True),
            truncated_prompt=False,
        ),
    )
    monkeypatch.setattr(phase6_main, "_safe_tokenize", lambda *_args, **_kwargs: (tokenized, ()))
    saved: SimpleNamespace | None = None

    def save(model: object, **_kwargs: object) -> SimpleNamespace:
        nonlocal saved
        model_holder["model"] = model
        saved = SimpleNamespace(
            checksum_sha256="1" * 64,
            weights_sha256="2" * 64,
            weights_size_bytes=10,
        )
        return saved

    model_holder: dict[str, object] = {}

    monkeypatch.setattr(phase6_main, "save_checkpoint", save)
    monkeypatch.setattr(
        phase6_main,
        "load_checkpoint",
        lambda *_args, **_kwargs: (
            model_holder["model"],
            SimpleNamespace(weights_sha256="2" * 64),
        ),
    )
    model_config = TransformerConfig(
        model_version="0.1.0",
        layers=1,
        width=32,
        heads=4,
        context_length=16,
        feed_forward_multiplier=2,
        dropout=0.0,
        tie_embeddings=True,
        bias=True,
    )
    training = TransformerTrainingConfig(
        seed=11,
        device="cpu",
        allow_cpu_fallback=True,
        steps=1,
        batch_size=1,
        learning_rate=0.001,
        weight_decay=0.0,
        gradient_clip_norm=1.0,
        evaluation_interval=1,
    )
    data = ExperimentData(
        train=(_example("train", SplitName.IID_TRAIN),),
        validation=(_example("validation", SplitName.IID_VALIDATION),),
        inventory_sha256="3" * 64,
    )
    phase5 = load_phase5_config(Path("configs/experiments/phase5-pilot-v0.1.0.toml"))
    result = phase6_main._train_model(
        experiment_id="E5_renderer_diversity_ablation",
        model_config=model_config,
        training=training,
        tokenizer=cast(Any, SimpleNamespace(vocab_size=64)),
        tokenizer_manifest=cast(Any, SimpleNamespace(checksum_sha256="4" * 64)),
        data=data,
        serialization_config=phase5,
        output=tmp_path / "checkpoint",
        source_commit="abcdef0",
    )
    assert saved is not None
    assert result.selected_step in {0, 1}
    assert result.parameter_count > 0
    assert result.scored_target_tokens == 2


def test_selection_gate_writes_and_verifies_without_test_materialization(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = load_phase6_config(Path("configs/experiments/phase6-main-v0.1.0.toml"))
    phase5 = load_phase5_config(Path(config.phase6.phase5_config_path))
    phase4 = load_phase4_config(Path(config.phase6.phase4_config_path))
    data = ExperimentData(
        train=(_example("train", SplitName.IID_TRAIN),),
        validation=(_example("validation", SplitName.IID_VALIDATION),),
        inventory_sha256="8" * 64,
    )
    phase5_report = SimpleNamespace(
        checksum_sha256=config.phase6.phase5_report_sha256,
        transformer_results=(SimpleNamespace(selected_validation_nll=1.0),),
    )
    verified = SimpleNamespace(candidate=SimpleNamespace(checksum_sha256="5" * 64))
    tokenizer = SimpleNamespace(manifest=SimpleNamespace(checksum_sha256="6" * 64))
    monkeypatch.setattr(
        phase6_main,
        "_load_inputs",
        lambda *_args: (phase5, phase4, phase5_report, verified, tokenizer, "5" * 64),
    )
    monkeypatch.setattr(phase6_main, "materialize_experiment_data", lambda *_a, **_k: data)
    monkeypatch.setattr(
        phase6_main, "_ablation_training_sets", lambda _verified: ({"train"}, {"train"})
    )
    monkeypatch.setattr(
        phase6_main,
        "_train_model",
        lambda *, experiment_id, **_kwargs: _model_result(experiment_id),
    )
    report = run_phase6_selection(config, project_root=tmp_path, source_commit="abcdef0")
    assert not report.test_records_materialized
    assert (tmp_path / "runs" / f"{config.phase6.run_name}-selection").is_dir()
    selection = tmp_path / "runs" / f"{config.phase6.run_name}-selection"
    for name in (
        "main-checkpoint",
        "renderer-ablation-checkpoint",
        "abstention-ablation-checkpoint",
    ):
        (selection / name).mkdir()
    monkeypatch.setattr(
        phase6_main,
        "load_checkpoint",
        lambda *_a, **_k: (object(), SimpleNamespace(weights_sha256="2" * 64)),
    )
    assert verify_phase6_selection(config, project_root=tmp_path) == report


def _target_example(identifier: str, split: SplitName, task: TaskName) -> ExperimentExample:
    if task is TaskName.FAULT_FAMILY:
        target: (
            FaultDiagnosisTarget
            | NextActionTarget
            | PromptContinuationTarget
            | PromptEvidenceTarget
        ) = FaultDiagnosisTarget(diagnosis_status=DiagnosisStatus.NO_FAULT)
        label = "NO_FAULT"
    elif task is TaskName.NEXT_ACTION:
        target = NextActionTarget(immediate_action=ActionLabel.INSUFFICIENT_EVIDENCE)
        label = "INSUFFICIENT_EVIDENCE"
    elif task is TaskName.CONTINUE_LOG:
        target = PromptContinuationTarget(next_event_type=EventType.BENIGN_NOTE)
        label = "BENIGN_NOTE"
    else:
        target = PromptEvidenceTarget(
            evidence_slots=(EvidenceSlot.STABLE_OPERATION,), fact_refs=("e-0000",)
        )
        label = None
    target_text = canonical_json_bytes(target.model_dump(mode="json", round_trip=True)).decode(
        "utf-8"
    )
    return ExperimentExample(
        example_id=identifier,
        split_name=split,
        task_name=task,
        prompt_text="[e-0000] [T+000] stable fictional note",
        target_text=target_text,
        classification_label=label,
        source_checksum_sha256="b" * 64,
    )


def _exact_predictions(
    records: tuple[ExperimentExample, ...],
) -> tuple[DecodedPrediction, ...]:
    return tuple(
        _prediction(
            record,
            generated_text=record.target_text,
            generated_token_count=4,
            prompt_truncated=False,
            generation_truncated=False,
            confidence=0.9,
        )
        for record in records
    )


def test_evaluation_gate_reports_every_split_and_verifies_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = load_phase6_config(Path("configs/experiments/phase6-main-v0.1.0.toml"))
    (tmp_path / "golden").mkdir()
    (tmp_path / "golden" / "golden-suite-v0.1.0.json").write_text("{}")
    (tmp_path / "artifacts" / "review").mkdir(parents=True)
    (tmp_path / "artifacts" / "review" / "golden-review-record-v0.1.0.json").write_text("{}")
    (tmp_path / "runs" / "phase5-pilot-v0.1.0").mkdir(parents=True)
    phase5 = load_phase5_config(Path(config.phase6.phase5_config_path))
    phase4 = load_phase4_config(Path(config.phase6.phase4_config_path))
    selection = Phase6SelectionReport.model_construct(
        source_commit="abcdef0",
        phase6_config_sha256=canonical_sha256(config.model_dump(mode="json", round_trip=True)),
        phase5_report_sha256=config.phase6.phase5_report_sha256,
        dataset_candidate_sha256="5" * 64,
        tokenizer_manifest_sha256="6" * 64,
        training_inventory_sha256="7" * 64,
        test_records_materialized=False,
        test_predictions_generated=False,
        all_ablation_selection_complete=True,
        e7_status="not_applicable_no_compound_iid_train_rows",
        results=(
            _model_result("E3_main_transformer"),
            _model_result("E5_renderer_diversity_ablation"),
            _model_result("E6_abstention_ablation"),
        ),
        selection_thresholds_passed=True,
        checksum_sha256="9" * 64,
    )
    monkeypatch.setattr(phase6_main, "verify_phase6_selection", lambda *_a, **_k: selection)
    phase5_result = SimpleNamespace(
        checkpoint_manifest_sha256="1" * 64,
        device="cpu",
        batch_size=1,
    )
    phase5_report = SimpleNamespace(transformer_results=(phase5_result, phase5_result))
    tokenizer = SimpleNamespace(manifest=SimpleNamespace(checksum_sha256="6" * 64))
    monkeypatch.setattr(
        phase6_main,
        "_load_inputs",
        lambda *_args: (
            phase5,
            phase4,
            phase5_report,
            SimpleNamespace(),
            tokenizer,
            "5" * 64,
        ),
    )
    monkeypatch.setattr(
        phase6_main,
        "load_golden_review_packet",
        lambda _p: SimpleNamespace(packet_sha256=config.phase6.golden_packet_sha256),
    )
    monkeypatch.setattr(
        phase6_main, "load_golden_review_record", lambda _p: SimpleNamespace(record_sha256="a" * 64)
    )
    monkeypatch.setattr(phase6_main, "verify_golden_review", lambda *_a, **_k: None)
    monkeypatch.setattr(phase6_main, "_raw_dataset_freeze", lambda *_a: None)
    monkeypatch.setattr(phase6_main, "load_checkpoint", lambda *_a, **_k: (object(), object()))
    by_split: dict[SplitName, tuple[ExperimentExample, ...]] = {
        SplitName.IID_TRAIN: (_target_example("train", SplitName.IID_TRAIN, TaskName.NEXT_ACTION),),
        SplitName.IID_VALIDATION: (),
    }
    for split in phase6_main.TEST_SPLITS:
        by_split[split] = tuple(
            _target_example(f"{split.value}:{task.value}", split, task)
            for task in (
                TaskName.FAULT_FAMILY,
                TaskName.NEXT_ACTION,
                TaskName.CONTINUE_LOG,
                TaskName.EXTRACT_EVIDENCE,
            )
        )
    data = Phase6ExperimentData(
        by_split=MappingProxyType(by_split),
        inventory_sha256_by_split=MappingProxyType(dict.fromkeys(by_split, "c" * 64)),
        all_records=tuple(record for records in by_split.values() for record in records),
    )
    monkeypatch.setattr(phase6_main, "materialize_phase6_data", lambda *_a, **_k: data)
    monkeypatch.setattr(
        phase6_main,
        "greedy_decode_predictions",
        lambda _model, _tokenizer, records, *_a, **_k: _exact_predictions(records),
    )
    classification = tuple(
        classification_metrics(task, (label,), (label,))
        for task, label in (
            (TaskName.FAULT_FAMILY, "NO_FAULT"),
            (TaskName.NEXT_ACTION, "INSUFFICIENT_EVIDENCE"),
            (TaskName.CONTINUE_LOG, "BENIGN_NOTE"),
        )
    )
    monkeypatch.setattr(
        phase6_main,
        "_evaluate_model_split",
        lambda experiment_id, _model, records, **_kwargs: ModelSplitEvaluation(
            experiment_id=experiment_id,
            split_name=records[0].split_name,
            sample_count=len(records),
            scored_sample_count=len(records),
            insufficient_context_by_design=0,
            language_model=language_model_metrics(
                sample_count=len(records), target_token_count=10, negative_log_likelihood=0.1
            ),
            classification=classification,
        ),
    )
    simple_results = (
        _result(
            name="majority_frequency",
            task_name=TaskName.FAULT_FAMILY.value,
            parameter_count=0,
            elapsed_seconds=0.0,
            classification=classification[0],
        ),
        _result(
            name="majority_frequency",
            task_name=TaskName.NEXT_ACTION.value,
            parameter_count=0,
            elapsed_seconds=0.0,
            classification=classification[1],
        ),
        _result(
            name="majority_frequency",
            task_name=TaskName.CONTINUE_LOG.value,
            parameter_count=0,
            elapsed_seconds=0.0,
            classification=classification[2],
        ),
        _result(
            name="token_trigram_additive",
            task_name="target_language_modeling",
            parameter_count=1,
            elapsed_seconds=0.0,
            language_model=language_model_metrics(
                sample_count=4, target_token_count=10, negative_log_likelihood=1.0
            ),
        ),
    )
    monkeypatch.setattr(phase6_main, "_baseline_split", lambda *_a, **_k: simple_results)
    golden = tuple(
        _target_example(f"golden:{index}", SplitName.IID_TEST, TaskName.NEXT_ACTION)
        for index in range(15)
    )
    monkeypatch.setattr(phase6_main, "_golden_examples", lambda *_a: golden)
    report = run_phase6_evaluation(config, project_root=tmp_path, source_commit="abcdef0")
    assert report.test_example_count == 894
    assert len(report.main_prediction_metrics) == 7
    assert report.golden_exact_match_rate == 1.0
    monkeypatch.setattr(phase6_main, "_read_predictions", lambda *_a, **_k: ())
    monkeypatch.setattr(phase6_main, "_read_comparison_predictions", lambda *_a, **_k: ())
    assert verify_phase6_evaluation(config, project_root=tmp_path) == report


def test_phase6_cli_executes_all_four_explicit_gates(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = load_phase6_config(Path("configs/experiments/phase6-main-v0.1.0.toml"))
    selection = SimpleNamespace(
        results=(SimpleNamespace(selected_step=100, selected_validation_nll=0.2),),
        checksum_sha256="1" * 64,
        run_status="phase6_selection_passed",
        selection_thresholds_passed=True,
    )
    evaluation = SimpleNamespace(
        negative_results=("schema_validity",),
        golden_exact_match_rate=0.5,
        checksum_sha256="2" * 64,
        run_status="phase6_evaluation_complete",
        test_example_count=894,
    )
    monkeypatch.setattr(phase6_cli, "load_phase6_config", lambda _path: config)
    monkeypatch.setattr(phase6_cli, "run_phase6_selection", lambda *_a, **_k: selection)
    monkeypatch.setattr(phase6_cli, "verify_phase6_selection", lambda *_a, **_k: selection)
    monkeypatch.setattr(phase6_cli, "run_phase6_evaluation", lambda *_a, **_k: evaluation)
    monkeypatch.setattr(phase6_cli, "verify_phase6_evaluation", lambda *_a, **_k: evaluation)
    config_path = "configs/experiments/phase6-main-v0.1.0.toml"
    commands = (
        ("run-phase6-selection", "phase6_selection_passed"),
        ("verify-phase6-selection", "phase6_selection_passed"),
        ("run-phase6-evaluation", "phase6_evaluation_complete"),
        ("verify-phase6-evaluation", "phase6_evaluation_complete"),
    )
    for command, status in commands:
        arguments = [command, "--config", config_path]
        if command.startswith("run-"):
            arguments.extend(("--source-commit", "abcdef0"))
        assert phase6_cli.main(arguments) == 0
        assert status in capsys.readouterr().out


def test_safe_tokenization_tracks_only_declared_context_overflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = (
        _example("fits", SplitName.IID_TEST),
        _example("overflow", SplitName.IID_TEST),
    )
    tokenized = TokenizedExample(
        example_id="fits",
        token_ids=(1, 2),
        target_mask=(False, True),
        truncated_prompt=False,
    )

    def tokenize(record: ExperimentExample, *_args: object, **_kwargs: object) -> TokenizedExample:
        if record.example_id == "overflow":
            raise ValueError("complete target does not fit the configured model context")
        return tokenized

    monkeypatch.setattr(phase6_main, "tokenize_example", tokenize)
    phase5 = load_phase5_config(Path("configs/experiments/phase5-pilot-v0.1.0.toml"))
    observed, insufficient = phase6_main._safe_tokenize(
        records, cast(Any, object()), phase5, context_length=512
    )
    assert observed == (tokenized,)
    assert insufficient == ("overflow",)
    monkeypatch.setattr(
        phase6_main,
        "tokenize_example",
        lambda *_a, **_k: (_ for _ in ()).throw(ValueError("unexpected")),
    )
    with pytest.raises(ValueError, match="unexpected"):
        phase6_main._safe_tokenize(records, cast(Any, object()), phase5, context_length=512)


def test_ablation_filters_use_alias_and_structured_abstention_truth() -> None:
    canonical = SimpleNamespace(render_id="r1", alias_family_id=AliasFamily.CANONICAL)
    heldout = SimpleNamespace(render_id="r2", alias_family_id=AliasFamily.HELDOUT)
    continue_target = NextActionTarget(immediate_action=ActionLabel.CONTINUE_MONITORING)
    abstain_target = NextActionTarget(immediate_action=ActionLabel.INSUFFICIENT_EVIDENCE)
    examples = (
        SimpleNamespace(
            example_id="keep",
            split_name=SplitName.IID_TRAIN,
            prompt_render_ids=("r1",),
            task_target=SimpleNamespace(target=continue_target),
        ),
        SimpleNamespace(
            example_id="heldout",
            split_name=SplitName.IID_TRAIN,
            prompt_render_ids=("r2",),
            task_target=SimpleNamespace(target=continue_target),
        ),
        SimpleNamespace(
            example_id="abstain",
            split_name=SplitName.IID_TRAIN,
            prompt_render_ids=("r1",),
            task_target=SimpleNamespace(target=abstain_target),
        ),
        SimpleNamespace(
            example_id="test",
            split_name=SplitName.IID_TEST,
            prompt_render_ids=("r1",),
            task_target=SimpleNamespace(target=continue_target),
        ),
    )
    verified = SimpleNamespace(
        candidate=SimpleNamespace(rendered_candidates=(canonical, heldout), task_examples=examples)
    )
    renderer, abstention = phase6_main._ablation_training_sets(verified)
    assert renderer == {"keep", "abstain"}
    assert abstention == {"keep", "heldout"}


def test_raw_freeze_access_ledger_and_prediction_artifacts_fail_closed(
    tmp_path: Path,
) -> None:
    config = load_phase6_config(Path("configs/experiments/phase6-main-v0.1.0.toml"))
    phase4 = load_phase4_config(Path(config.phase6.phase4_config_path))
    artifact = tmp_path / phase4.phase4.dataset_root / phase4.phase4.dataset_artifact_name
    artifact.mkdir(parents=True)
    manifest = artifact / "split-manifest.jsonl"
    examples = artifact / "task-examples.jsonl"
    manifest.write_text("manifest\n")
    examples.write_text("examples\n")
    freeze = config.test_freeze.model_copy(
        update={
            "split_manifest_raw_sha256": phase6_main._sha256(manifest),
            "task_examples_raw_sha256": phase6_main._sha256(examples),
        }
    )
    local = config.model_copy(update={"test_freeze": freeze})
    phase6_main._raw_dataset_freeze(local, phase4, tmp_path)
    examples.write_text("changed\n")
    with pytest.raises(ValueError, match="task-examples"):
        phase6_main._raw_dataset_freeze(local, phase4, tmp_path)

    selection = Phase6SelectionReport.model_construct(
        source_commit="abcdef0",
        phase6_config_sha256="1" * 64,
        phase5_report_sha256="2" * 64,
        dataset_candidate_sha256="3" * 64,
        tokenizer_manifest_sha256="4" * 64,
        training_inventory_sha256="5" * 64,
        test_records_materialized=False,
        test_predictions_generated=False,
        all_ablation_selection_complete=True,
        e7_status="not_applicable_no_compound_iid_train_rows",
        results=(
            _model_result("E3_main_transformer"),
            _model_result("E5_renderer_diversity_ablation"),
            _model_result("E6_abstention_ablation"),
        ),
        selection_thresholds_passed=True,
        checksum_sha256="6" * 64,
    )
    first = phase6_main._authorize_heldout_access(
        config,
        project_root=tmp_path,
        source_commit="abcdef0",
        selection=selection,
        golden_record_sha256="7" * 64,
    )
    second = phase6_main._authorize_heldout_access(
        config,
        project_root=tmp_path,
        source_commit="abcdef0",
        selection=selection,
        golden_record_sha256="7" * 64,
    )
    assert first == second
    with pytest.raises(ValueError, match="another evaluation"):
        phase6_main._authorize_heldout_access(
            config,
            project_root=tmp_path,
            source_commit="1234567",
            selection=selection,
            golden_record_sha256="7" * 64,
        )

    record = _target_example("prediction", SplitName.IID_TEST, TaskName.NEXT_ACTION)
    prediction = _exact_predictions((record,))[0]
    prediction_path = tmp_path / "predictions.jsonl"
    phase6_main._write_jsonl(prediction_path, (prediction,))
    assert phase6_main._read_predictions(prediction_path, expected_count=1) == (prediction,)
    with pytest.raises(ValueError, match="count or example IDs"):
        phase6_main._read_predictions(prediction_path, expected_count=2)
    comparison = phase6_main._comparison_prediction("E2_smaller_transformer", prediction)
    comparison_path = tmp_path / "comparison-predictions.jsonl"
    phase6_main._write_jsonl(comparison_path, (comparison,))
    assert phase6_main._read_comparison_predictions(comparison_path, expected_count=1) == (
        comparison,
    )
    with pytest.raises(ValueError, match="count or composite IDs"):
        phase6_main._read_comparison_predictions(comparison_path, expected_count=2)


def test_split_helpers_score_classification_language_and_failure_taxonomy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = tuple(
        _target_example(f"helper:{task.value}", SplitName.IID_TEST, task)
        for task in (
            TaskName.FAULT_FAMILY,
            TaskName.NEXT_ACTION,
            TaskName.CONTINUE_LOG,
            TaskName.EXTRACT_EVIDENCE,
        )
    )
    predictions = cast(tuple[Any, ...], _exact_predictions(records))
    classified = phase6_main._classification_results(records, predictions)
    assert tuple(item.task_name for item in classified) == (
        TaskName.FAULT_FAMILY,
        TaskName.NEXT_ACTION,
        TaskName.CONTINUE_LOG,
    )
    tokenized = (
        TokenizedExample(
            example_id="helper",
            token_ids=(1, 4, 2),
            target_mask=(False, True, True),
            truncated_prompt=False,
        ),
    )
    monkeypatch.setattr(phase6_main, "_safe_tokenize", lambda *_a, **_k: (tokenized, ("overflow",)))
    monkeypatch.setattr(phase6_main, "_nll", lambda *_a, **_k: (0.25, 2))
    phase5 = load_phase5_config(Path("configs/experiments/phase5-pilot-v0.1.0.toml"))
    result = phase6_main._evaluate_model_split(
        "E3_main_transformer",
        cast(Any, SimpleNamespace(config=SimpleNamespace(context_length=512))),
        records[:2],
        tokenizer=cast(Any, object()),
        phase5=phase5,
        batch_size=1,
        device=cast(Any, object()),
        predictions=predictions[:2],
    )
    assert result.scored_sample_count == 1
    assert result.insufficient_context_by_design == 1
    assert result.language_model is not None
    invalid = predictions[0].model_copy(
        update={
            "generated_text": "not-json",
            "json_parse_success": False,
            "schema_valid": False,
            "predicted_target_json": None,
            "classification_label": None,
            "confidence": 0.0,
        }
    )
    assert phase6_main._failure_category(records[0], invalid) == "invalid_json"
    gallery = phase6_main._failure_gallery(records, (invalid, *predictions[1:]))
    assert gallery[0].category == "invalid_json"
    assert phase6_main._target_object('{"a":1}') == {"a": 1}
    assert phase6_main._evidence_slots({"evidence_slots": "wrong"}) == ()


def test_baseline_split_reports_language_only_when_classification_is_not_applicable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenized = (
        TokenizedExample(
            example_id="baseline",
            token_ids=(1, 4, 2),
            target_mask=(False, True, True),
            truncated_prompt=False,
        ),
    )
    monkeypatch.setattr(phase6_main, "_safe_tokenize", lambda *_a, **_k: (tokenized, ()))
    expected = _result(
        name="token_trigram_additive",
        task_name="target_language_modeling",
        parameter_count=1,
        elapsed_seconds=0.0,
        language_model=language_model_metrics(
            sample_count=1, target_token_count=2, negative_log_likelihood=0.5
        ),
    )
    monkeypatch.setattr(phase6_main, "_token_ngram_language_model", lambda *_a, **_k: expected)
    phase5 = load_phase5_config(Path("configs/experiments/phase5-pilot-v0.1.0.toml"))
    train = (_target_example("train-only", SplitName.IID_TRAIN, TaskName.NEXT_ACTION),)
    heldout = (
        _target_example("counterfactual-only", SplitName.COUNTERFACTUAL_TEST, TaskName.NEXT_ACTION),
    )
    observed = phase6_main._baseline_split(
        train,
        heldout,
        tokenizer=cast(Any, SimpleNamespace(vocab_size=64)),
        phase5=phase5,
    )
    assert observed == (expected,)


def test_phase6_input_loader_binds_phase4_phase5_dataset_and_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_phase6_config(Path("configs/experiments/phase6-main-v0.1.0.toml"))
    phase5 = load_phase5_config(Path(config.phase6.phase5_config_path))
    phase4 = load_phase4_config(Path(config.phase6.phase4_config_path))
    phase5_report = SimpleNamespace(checksum_sha256=config.phase6.phase5_report_sha256)
    phase4_report = SimpleNamespace(tokenizer_manifest_sha256="1" * 64)
    verified = SimpleNamespace(candidate=SimpleNamespace(checksum_sha256="2" * 64))
    tokenizer: Any = SimpleNamespace(manifest=SimpleNamespace(checksum_sha256="1" * 64))
    monkeypatch.setattr(
        phase6_main, "resolve_project_path", lambda _root, relative, **_k: Path(relative)
    )
    monkeypatch.setattr(phase6_main, "load_phase5_config", lambda _path: phase5)
    monkeypatch.setattr(phase6_main, "verify_phase5_run", lambda *_a, **_k: phase5_report)
    monkeypatch.setattr(phase6_main, "load_phase4_config", lambda _path: phase4)
    monkeypatch.setattr(phase6_main, "verify_phase4_run", lambda *_a, **_k: phase4_report)
    monkeypatch.setattr(
        phase6_main,
        "ArtifactWriter",
        lambda root: SimpleNamespace(root=root),
    )
    monkeypatch.setattr(
        phase6_main,
        "verify_development_candidate_artifact",
        lambda *_a, **_k: verified,
    )
    monkeypatch.setattr(
        ProjectTokenizer,
        "load",
        lambda *_a, **_k: tokenizer,
    )
    observed = phase6_main._load_inputs(config, Path.cwd())
    assert observed[0] is phase5
    assert observed[1] is phase4
    assert observed[2] is phase5_report
    assert observed[3] is verified
    assert observed[4] is tokenizer
    assert observed[5] == "2" * 64
    wrong = config.phase6.model_copy(update={"phase5_config_sha256": "0" * 64})
    with pytest.raises(ValueError, match="Phase 5 config"):
        phase6_main._load_inputs(config.model_copy(update={"phase6": wrong}), Path.cwd())


def test_phase6_contracts_reject_bad_checksums_counts_rates_and_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="source_commit"):
        phase6_main._source_commit("NOT-HEX")
    training = TransformerTrainingConfig(
        seed=1,
        device="mps",
        allow_cpu_fallback=False,
        steps=1,
        batch_size=1,
        learning_rate=0.001,
        weight_decay=0.0,
        gradient_clip_norm=1.0,
        evaluation_interval=1,
    )
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="fallback is disabled"):
        phase6_main._device(training)
    with pytest.raises(ValueError, match="duplicate key"):
        phase6_main._strict_json(b'{"a":1,"a":2}')
    with pytest.raises(ValueError, match="non-finite"):
        phase6_main._strict_json(b'{"a":NaN}')

    result = _model_result("E3_main_transformer")
    payload = result.model_dump(mode="python")
    payload["selected_step"] = 0
    with pytest.raises(ValueError, match="validation-optimal"):
        Phase6ModelResult.model_validate(payload)
    with pytest.raises(ValueError, match="counts do not match"):
        ModelSplitEvaluation(
            experiment_id="E3",
            split_name=SplitName.IID_TEST,
            sample_count=2,
            scored_sample_count=1,
            insufficient_context_by_design=0,
            language_model=language_model_metrics(
                sample_count=1, target_token_count=1, negative_log_likelihood=0.1
            ),
            classification=(),
        )
    prediction_record = _target_example("contract", SplitName.IID_TEST, TaskName.NEXT_ACTION)
    prediction = _exact_predictions((prediction_record,))[0]
    with pytest.raises(ValueError, match="comparison prediction checksum"):
        ExperimentDecodedPrediction(
            experiment_id="E3",
            prediction=prediction,
            checksum_sha256="0" * 64,
        )
    with pytest.raises(ValueError, match="held-out access record checksum"):
        HeldoutAccessRecord(
            source_commit="abcdef0",
            phase6_config_sha256="1" * 64,
            selection_report_sha256="2" * 64,
            golden_review_record_sha256="3" * 64,
            access_count=1,
            checksum_sha256="0" * 64,
        )
    metric = phase6_main._main_metrics(
        SplitName.IID_TEST,
        (prediction_record,),
        (prediction,),
        load_phase6_config(Path("configs/experiments/phase6-main-v0.1.0.toml")),
    )
    bad_metric = metric.model_dump(mode="python")
    bad_metric["parse_success_rate"] = 2.0
    with pytest.raises(ValueError, match="finite probabilities"):
        MainPredictionMetrics.model_validate(bad_metric)
