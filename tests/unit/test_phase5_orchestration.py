from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from reactorbench.evaluation import (
    ExperimentData,
    ExperimentExample,
    classification_metrics,
    language_model_metrics,
    load_phase5_config,
)
from reactorbench.evaluation.baselines import BaselineResult, _result
from reactorbench.evaluation.serialization import TokenizedExample
from reactorbench.model import load_phase4_config
from reactorbench.schemas.base import canonical_sha256
from reactorbench.schemas.enums import SplitName, TaskName
from reactorbench.tokenizer import ProjectTokenizer, TokenizerArtifactManifest
from reactorbench.training.pilot import (
    TransformerPilotResult,
    ValidationPoint,
    run_phase5_pilot,
    verify_phase5_run,
)


def _data() -> ExperimentData:
    train = tuple(
        ExperimentExample(
            example_id=f"train-{index}",
            split_name=SplitName.IID_TRAIN,
            task_name=TaskName.NEXT_ACTION,
            prompt_text="fictional prompt",
            target_text="{}",
            classification_label="CONTINUE_MONITORING",
            source_checksum_sha256=f"{index % 10}" * 64,
        )
        for index in range(630)
    )
    validation = tuple(
        ExperimentExample(
            example_id=f"validation-{index}",
            split_name=SplitName.IID_VALIDATION,
            task_name=TaskName.NEXT_ACTION,
            prompt_text="fictional validation",
            target_text="{}",
            classification_label="CONTINUE_MONITORING",
            source_checksum_sha256=f"{index % 10}" * 64,
        )
        for index in range(252)
    )
    return ExperimentData(
        train=train,
        validation=validation,
        inventory_sha256=canonical_sha256(tuple(item.example_id for item in train + validation)),
    )


def _classification(task: TaskName) -> Any:
    return classification_metrics(task, ("A", "B"), ("A", "B"))


def _baselines() -> tuple[BaselineResult, ...]:
    return (
        _result(
            name="majority_frequency",
            task_name="fault_family",
            parameter_count=0,
            elapsed_seconds=0.1,
            classification=_classification(TaskName.FAULT_FAMILY),
        ),
        _result(
            name="majority_frequency",
            task_name="next_action",
            parameter_count=0,
            elapsed_seconds=0.1,
            classification=_classification(TaskName.NEXT_ACTION),
        ),
        _result(
            name="majority_frequency",
            task_name="continue_log",
            parameter_count=0,
            elapsed_seconds=0.1,
            classification=_classification(TaskName.CONTINUE_LOG),
        ),
        _result(
            name="deterministic_keyword_rules",
            task_name="fault_family",
            parameter_count=0,
            elapsed_seconds=0.1,
            classification=_classification(TaskName.FAULT_FAMILY),
        ),
        _result(
            name="deterministic_keyword_rules",
            task_name="next_action",
            parameter_count=0,
            elapsed_seconds=0.1,
            classification=_classification(TaskName.NEXT_ACTION),
        ),
        _result(
            name="word_ngram_suffix",
            task_name="continue_log",
            parameter_count=2,
            elapsed_seconds=0.1,
            classification=_classification(TaskName.CONTINUE_LOG),
        ),
        _result(
            name="token_trigram_additive",
            task_name="target_language_modeling",
            parameter_count=2,
            elapsed_seconds=0.1,
            language_model=language_model_metrics(
                sample_count=2, target_token_count=4, negative_log_likelihood=1.0
            ),
        ),
        _result(
            name="bag_of_words_logistic_regression",
            task_name="fault_family",
            parameter_count=2,
            elapsed_seconds=0.1,
            classification=_classification(TaskName.FAULT_FAMILY),
        ),
        _result(
            name="gru_sequence_classifier",
            task_name="fault_family",
            parameter_count=2,
            elapsed_seconds=0.1,
            classification=_classification(TaskName.FAULT_FAMILY),
        ),
        _result(
            name="gru_sequence_classifier",
            task_name="continue_log",
            parameter_count=2,
            elapsed_seconds=0.1,
            classification=_classification(TaskName.CONTINUE_LOG),
        ),
    )


def _transformer(tier: str, steps: int, context: int, marker: str) -> TransformerPilotResult:
    return TransformerPilotResult(
        tier=cast(Any, tier),
        device="cpu",
        parameter_count=100,
        context_length=context,
        batch_size=4,
        training_steps=steps,
        selected_step=steps,
        initial_validation_nll=10.0,
        selected_validation_nll=5.0,
        validation_nll_reduction_fraction=0.5,
        final_training_nll=4.0,
        validation_curve=(
            ValidationPoint(step=0, target_nll=10.0),
            ValidationPoint(step=steps, target_nll=5.0),
        ),
        elapsed_seconds=1.0,
        scored_target_tokens=100,
        target_tokens_per_second=100.0,
        process_peak_rss_bytes=1000,
        mps_peak_current_allocated_bytes=0,
        mps_peak_driver_allocated_bytes=0,
        checkpoint_manifest_sha256=marker * 64,
        checkpoint_weights_sha256=marker.upper().casefold() * 64,
        checkpoint_size_bytes=100,
    )


def test_phase5_run_and_verify_are_nonoverwriting_and_relationship_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_phase5_config(Path("configs/experiments/phase5-pilot-v0.1.0.toml"))
    phase4 = load_phase4_config(Path("configs/model/phase4-smoke-v0.1.0.toml"))
    data = _data()
    tokenizer = object.__new__(ProjectTokenizer)
    tokenizer_manifest = SimpleNamespace(checksum_sha256="a" * 64)
    tokenizer.manifest = cast(TokenizerArtifactManifest, tokenizer_manifest)
    smoke_report = SimpleNamespace(checksum_sha256="b" * 64)
    candidate_sha = "c" * 64
    (tmp_path / "uv.lock").write_text("locked", encoding="utf-8")

    def resolve(_root: Path, relative: str, *, must_exist: bool) -> Path:
        path = tmp_path.joinpath(*relative.split("/"))
        if must_exist and not path.exists():
            raise FileNotFoundError(path)
        return path

    monkeypatch.setattr(
        "reactorbench.training.pilot._load_phase5_inputs",
        lambda *_args, **_kwargs: (
            phase4,
            smoke_report,
            tokenizer_manifest,
            tokenizer,
            data,
            candidate_sha,
        ),
    )
    monkeypatch.setattr("reactorbench.training.pilot.resolve_project_path", resolve)
    monkeypatch.setattr(
        "reactorbench.training.pilot._tokenize_inventory",
        lambda records, *_args, **_kwargs: tuple(
            TokenizedExample(item.example_id, (1, 2), (False, True), False) for item in records
        ),
    )
    monkeypatch.setattr(
        "reactorbench.training.pilot.tokenize_example",
        lambda item, *_args, **_kwargs: TokenizedExample(
            item.example_id, (1, 2), (False, True), False
        ),
    )
    monkeypatch.setattr(
        "reactorbench.training.pilot.run_preregistered_baselines",
        lambda *_args, **_kwargs: _baselines(),
    )
    smaller = _transformer("smaller_transformer", 300, 128, "d")
    pilot = _transformer("pilot_transformer", 500, 256, "e")

    def train_stub(*, tier: str, output_directory: Path, **_kwargs: object) -> Any:
        output_directory.mkdir()
        return smaller if tier == "smaller_transformer" else pilot

    monkeypatch.setattr(
        "reactorbench.training.pilot._train_transformer",
        train_stub,
    )

    report = run_phase5_pilot(config, project_root=tmp_path, source_commit="abcdef0")

    assert report.run_status == "phase5_pilot_passed"
    assert len(report.baseline_results) == 10
    assert (tmp_path / "runs" / config.phase5.run_name / "report.json").is_file()
    with pytest.raises(FileExistsError, match="already exists"):
        run_phase5_pilot(config, project_root=tmp_path, source_commit="abcdef0")

    def load_checkpoint_stub(
        _directory: Path,
        *,
        expected_manifest_sha256: str,
        expected_tokenizer_sha256: str,
        device: Any,
    ) -> tuple[object, SimpleNamespace]:
        result = (
            smaller if expected_manifest_sha256 == smaller.checkpoint_manifest_sha256 else pilot
        )
        assert expected_tokenizer_sha256 == "a" * 64
        return object(), SimpleNamespace(
            weights_sha256=result.checkpoint_weights_sha256,
            weights_size_bytes=result.checkpoint_size_bytes,
        )

    monkeypatch.setattr("reactorbench.training.pilot.load_checkpoint", load_checkpoint_stub)
    assert verify_phase5_run(config, project_root=tmp_path) == report
