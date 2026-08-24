from __future__ import annotations

import ast
import json
import random
import shutil
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Literal, cast

import pytest
import torch
from pydantic import ValidationError

from reactorbench.model import TransformerConfig, TransformerLM
from reactorbench.remediation.config import RemediationTraining
from reactorbench.remediation.serialization import CompactTokenizedExample
from reactorbench.remediation.training import (
    CompactTrainingResult,
    CompactTrainingStopped,
    DeviceResolution,
    EvaluationCallback,
    MonotonicClock,
    SamplingStrategy,
    StopRequested,
    TrainingError,
    TrainingProgress,
    TrainingStateManifest,
    resolve_training_device,
    train_compact_model,
    uniform_control_batch_indices,
)
from reactorbench.schemas.base import canonical_json_bytes, canonical_sha256
from reactorbench.schemas.enums import SplitName, TaskName
from reactorbench.tokenizer import TokenizerArtifactManifest, TrainingCorpusManifest


def _model_config() -> TransformerConfig:
    return TransformerConfig(
        model_version="0.3.0",
        layers=1,
        width=16,
        heads=4,
        context_length=8,
        feed_forward_multiplier=2,
        dropout=0.2,
        tie_embeddings=True,
        bias=True,
    )


def _training(
    *,
    device: Literal["cpu", "mps"] = "cpu",
    allow_cpu_fallback: bool = False,
) -> RemediationTraining:
    return RemediationTraining(
        seed=29,
        device=device,
        allow_cpu_fallback=allow_cpu_fallback,
        steps=4,
        batch_size=6,
        learning_rate=0.003,
        weight_decay=0.01,
        gradient_clip_norm=1.0,
        evaluation_interval=1,
        durable_checkpoint_interval=2,
    )


def _tokenizer_manifest() -> TokenizerArtifactManifest:
    corpus = TrainingCorpusManifest(
        source_split=SplitName.IID_TRAIN,
        candidate_bundle_sha256="1" * 64,
        candidate_artifact_manifest_sha256="2" * 64,
        postrender_packet_sha256="3" * 64,
        postrender_approval_record_sha256="4" * 64,
        document_count=12,
        utf8_bytes=120,
        document_inventory_sha256="5" * 64,
        corpus_sha256="6" * 64,
    )
    draft = TokenizerArtifactManifest.model_construct(
        tokenizer_version="0.1.0",
        algorithm="sentencepiece_bpe",
        sentencepiece_version="0.2.2",
        requested_vocab_size=512,
        actual_vocab_size=512,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        special_symbols=("<|prompt|>", "<|target|>", "<|sep|>"),
        model_sha256="7" * 64,
        vocab_sha256="8" * 64,
        model_size_bytes=100,
        vocab_size_bytes=100,
        corpus=corpus,
        checksum_sha256="0" * 64,
    )
    checksum = canonical_sha256(
        draft.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
    )
    return TokenizerArtifactManifest(
        **draft.model_dump(mode="python", round_trip=True, exclude={"checksum_sha256"}),
        checksum_sha256=checksum,
    )


def _examples(count: int = 12, *, prefix: str = "example") -> tuple[CompactTokenizedExample, ...]:
    tasks = tuple(TaskName)
    return tuple(
        CompactTokenizedExample(
            example_id=f"{prefix}-{index:03d}",
            task_name=tasks[index % len(tasks)],
            group_id=f"{prefix}-group-{index:03d}",
            token_ids=(1, 4 + index % 7, 20 + index % 11, 2),
            target_mask=(False, False, True, True),
            prompt_token_count=2,
            target_token_count=2,
            prompt_tokens_retained=2,
            prompt_truncated=False,
        )
        for index in range(count)
    )


def _run(
    base: Path,
    *,
    resume: Path | None = None,
    stop_requested: StopRequested | None = None,
    progress: list[TrainingProgress] | None = None,
    evaluation_callback: EvaluationCallback | None = None,
    sampling: SamplingStrategy = "uniform_control",
    train_inventory_sha256: str = "a" * 64,
    candidate_id: str = "candidate-control",
    source_commit: str = "abcdef0",
    vocab_size: int = 512,
    output_name: str = "final-checkpoint",
    train_examples: tuple[CompactTokenizedExample, ...] | None = None,
    validation_examples: tuple[CompactTokenizedExample, ...] | None = None,
    monotonic_clock: MonotonicClock = time.perf_counter,
) -> CompactTrainingResult | CompactTrainingStopped:
    states = base / "states"
    states.mkdir(parents=True, exist_ok=True)
    progress_callback = None if progress is None else progress.append
    return train_compact_model(
        candidate_id=candidate_id,
        sampling_strategy=sampling,
        model_config=_model_config(),
        training=_training(),
        vocab_size=vocab_size,
        tokenizer_manifest=_tokenizer_manifest(),
        train_examples=_examples() if train_examples is None else train_examples,
        validation_examples=(
            _examples(6, prefix="validation")
            if validation_examples is None
            else validation_examples
        ),
        train_inventory_sha256=train_inventory_sha256,
        validation_inventory_sha256="b" * 64,
        output_directory=base / output_name,
        durable_state_root=states,
        source_commit=source_commit,
        resume_state_directory=resume,
        evaluation_callback=evaluation_callback,
        progress_callback=progress_callback,
        stop_requested=stop_requested,
        monotonic_clock=monotonic_clock,
    )


def _state_manifest(path: Path) -> TrainingStateManifest:
    return TrainingStateManifest.model_validate_json(
        (path / "manifest.json").read_bytes(), strict=True
    )


def test_uniform_control_is_deterministic_uniform_and_does_not_touch_rng() -> None:
    examples = _examples()
    python_state = random.getstate()
    torch_state = torch.get_rng_state().clone()

    first = uniform_control_batch_indices(examples, batch_size=len(examples), seed=17, cursor=0)
    second = uniform_control_batch_indices(examples, batch_size=len(examples), seed=17, cursor=0)
    wrapped = uniform_control_batch_indices(examples, batch_size=6, seed=17, cursor=10)

    assert first == second
    assert sorted(first) == list(range(len(examples)))
    assert len(wrapped) == 6
    assert random.getstate() == python_state
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_task_balanced_candidate_batch_has_equal_task_quotas() -> None:
    examples = _examples()
    indices = uniform_control_batch_indices(examples, batch_size=6, seed=29, cursor=0)
    assert len(indices) == 6

    from reactorbench.remediation.sampling import task_balanced_batch_indices

    balanced = task_balanced_batch_indices(examples, batch_size=6, seed=29, step=0)
    counts = Counter(examples[index].task_name for index in balanced)
    assert set(counts.values()) == {1}
    assert set(counts) == set(TaskName)


def test_uniform_sampler_rejects_invalid_inventory_and_bounds() -> None:
    examples = _examples()
    with pytest.raises(ValueError, match="non-empty exact tuple"):
        uniform_control_batch_indices((), batch_size=1, seed=1, cursor=0)
    with pytest.raises(ValueError, match="batch_size"):
        uniform_control_batch_indices(examples, batch_size=0, seed=1, cursor=0)
    with pytest.raises(ValueError, match="unique"):
        uniform_control_batch_indices(
            (examples[0], replace(examples[1], example_id=examples[0].example_id)),
            batch_size=1,
            seed=1,
            cursor=0,
        )
    with pytest.raises(ValueError, match="supervised token boundary"):
        uniform_control_batch_indices(
            (replace(examples[0], target_mask=(False, False, False, False)),),
            batch_size=1,
            seed=1,
            cursor=0,
        )


def test_uninterrupted_and_interrupted_resume_are_bit_exact(tmp_path: Path) -> None:
    def constant_score(model: TransformerLM, _step: int, _nll: float) -> float:
        assert not model.training
        return 1.0

    full_progress: list[TrainingProgress] = []
    full_rng = torch.get_rng_state().clone()
    full = _run(
        tmp_path / "full",
        progress=full_progress,
        evaluation_callback=constant_score,
    )
    assert isinstance(full, CompactTrainingResult)
    assert torch.equal(torch.get_rng_state(), full_rng)
    expected = min(
        full.validation_curve,
        key=lambda point: (point.selection_score, point.validation_nll, point.step),
    )
    assert (full.selected_step, full.selected_validation_nll) == (
        expected.step,
        expected.validation_nll,
    )

    stopped_progress: list[TrainingProgress] = []
    interrupted_rng = torch.get_rng_state().clone()
    stopped = _run(
        tmp_path / "resumed",
        stop_requested=lambda step: step == 2,
        progress=stopped_progress,
        evaluation_callback=constant_score,
    )
    assert isinstance(stopped, CompactTrainingStopped)
    assert torch.equal(torch.get_rng_state(), interrupted_rng)
    state_path = tmp_path / "resumed" / "states" / stopped.durable_state_name
    resumed = _run(
        tmp_path / "resumed",
        resume=state_path,
        evaluation_callback=constant_score,
    )
    assert isinstance(resumed, CompactTrainingResult)
    assert torch.equal(torch.get_rng_state(), interrupted_rng)

    assert resumed.validation_curve == full.validation_curve
    assert resumed.final_training_nll == full.final_training_nll
    assert resumed.scored_target_tokens == full.scored_target_tokens
    assert resumed.selected_step == full.selected_step
    assert resumed.selected_validation_nll == full.selected_validation_nll
    assert resumed.checkpoint_weights_sha256 == full.checkpoint_weights_sha256
    assert resumed.checkpoint_manifest_sha256 == full.checkpoint_manifest_sha256
    assert (tmp_path / "resumed/final-checkpoint/model.safetensors").read_bytes() == (
        tmp_path / "full/final-checkpoint/model.safetensors"
    ).read_bytes()
    full_final_state = _state_manifest(tmp_path / "full/states/state-step-00000004")
    resumed_final_state = _state_manifest(tmp_path / "resumed/states/state-step-00000004")
    assert full_final_state.files == resumed_final_state.files
    assert full_final_state.validation_curve == resumed_final_state.validation_curve

    assert [event.step for event in full_progress if event.event == "evaluation"] == [0, 1, 2, 3, 4]
    assert [event.step for event in full_progress if event.event == "durable_checkpoint"] == [2, 4]
    assert full_progress[-1].event == "final_checkpoint"
    assert [event.event for event in stopped_progress[-2:]] == [
        "durable_checkpoint",
        "stopped",
    ]


def test_checksum_bound_result_rejects_tampering(tmp_path: Path) -> None:
    result = _run(tmp_path / "result")
    assert isinstance(result, CompactTrainingResult)
    payload = result.model_dump(mode="python", round_trip=True)
    payload["selected_step"] = 1
    with pytest.raises(ValidationError):
        CompactTrainingResult.model_validate(payload, strict=True)


def test_result_tie_break_prefers_score_then_nll_then_earliest_step(tmp_path: Path) -> None:
    result = _run(tmp_path / "tie-break")
    assert isinstance(result, CompactTrainingResult)
    payload = result.model_dump(mode="python", round_trip=True)
    curve = cast(list[dict[str, object]], payload["validation_curve"])
    for point in curve:
        point["selection_score"] = 0.5
        point["validation_nll"] = 0.5
    payload["selected_step"] = 0
    payload["selected_score"] = 0.5
    payload["selected_validation_nll"] = 0.5
    payload["checksum_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "checksum_sha256"}
    )

    validated = CompactTrainingResult.model_validate(payload, strict=True)
    assert validated.selected_step == 0

    payload["selected_step"] = 1
    payload["checksum_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "checksum_sha256"}
    )
    with pytest.raises(ValidationError, match="lower score, lower validation NLL, then earlier"):
        CompactTrainingResult.model_validate(payload, strict=True)


def test_result_progress_and_state_contract_relationships_are_strict(tmp_path: Path) -> None:
    progress_cases = (
        {
            "event": "evaluation",
            "step": 1,
            "total_steps": 2,
        },
        {
            "event": "durable_checkpoint",
            "step": 1,
            "total_steps": 2,
        },
        {
            "event": "stopped",
            "step": 3,
            "total_steps": 2,
            "checkpoint_name": "state-step-00000003",
        },
    )
    for values in progress_cases:
        with pytest.raises(ValidationError):
            TrainingProgress.model_validate(values, strict=True)
    with pytest.raises(ValidationError, match="fallback flag"):
        DeviceResolution(requested="cpu", resolved="cpu", fallback_used=True)

    stopped = _run(tmp_path / "state", stop_requested=lambda step: step == 2)
    assert isinstance(stopped, CompactTrainingStopped)
    state_path = tmp_path / "state/states" / stopped.durable_state_name
    manifest = _state_manifest(state_path)
    base = manifest.model_dump(mode="python", round_trip=True)
    mutations = (
        {"step": 5},
        {"sampler_cursor": 13},
        {
            "optimizer_parameter_names": (
                *manifest.optimizer_parameter_names,
                manifest.optimizer_parameter_names[0],
            )
        },
        {"files": tuple(reversed(manifest.files))},
        {"best_step": 1},
        {"checksum_sha256": "f" * 64},
    )
    for mutation in mutations:
        payload = {**base, **mutation}
        with pytest.raises(ValidationError):
            TrainingStateManifest.model_validate(payload, strict=True)

    stopped_payload = stopped.model_dump(mode="python", round_trip=True)
    stopped_payload["completed_steps"] = stopped.total_steps
    with pytest.raises(ValidationError, match="cannot be marked"):
        CompactTrainingStopped.model_validate(stopped_payload, strict=True)
    cursor_payload = stopped.model_dump(mode="python", round_trip=True)
    cursor_payload["sampler_cursor"] = 1
    with pytest.raises(ValidationError, match="sampler cursor"):
        CompactTrainingStopped.model_validate(cursor_payload, strict=True)


def test_resume_rejects_hash_mismatch_tensor_tampering_and_extra_file(tmp_path: Path) -> None:
    source = tmp_path / "source"
    stopped = _run(source, stop_requested=lambda step: step == 2)
    assert isinstance(stopped, CompactTrainingStopped)
    original = source / "states" / stopped.durable_state_name

    wrong_hash = tmp_path / "wrong-hash"
    wrong_hash_state = wrong_hash / "states" / original.name
    wrong_hash_state.parent.mkdir(parents=True)
    shutil.copytree(original, wrong_hash_state)
    with pytest.raises(ValueError, match="frozen training inputs"):
        _run(
            wrong_hash,
            resume=wrong_hash_state,
            train_inventory_sha256="c" * 64,
        )

    tampered = tmp_path / "tampered"
    tampered_state = tampered / "states" / original.name
    tampered_state.parent.mkdir(parents=True)
    shutil.copytree(original, tampered_state)
    with (tampered_state / "optimizer.safetensors").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="checksum or size"):
        _run(tampered, resume=tampered_state)

    extra = tmp_path / "extra"
    extra_state = extra / "states" / original.name
    extra_state.parent.mkdir(parents=True)
    shutil.copytree(original, extra_state)
    (extra_state / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected file inventory"):
        _run(extra, resume=extra_state)


def test_resume_rejects_version_tamper_and_symlink(tmp_path: Path) -> None:
    source = tmp_path / "source"
    stopped = _run(source, stop_requested=lambda step: step == 2)
    assert isinstance(stopped, CompactTrainingStopped)
    original = source / "states" / stopped.durable_state_name

    version = tmp_path / "version"
    version_state = version / "states" / original.name
    version_state.parent.mkdir(parents=True)
    shutil.copytree(original, version_state)
    manifest_path = version_state / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["artifact_version"] = "9.9.9"
    manifest_path.write_bytes(canonical_json_bytes(payload) + b"\n")
    with pytest.raises(ValidationError):
        _run(version, resume=version_state)

    linked = tmp_path / "linked"
    linked_state = linked / "states" / original.name
    linked_state.parent.mkdir(parents=True)
    shutil.copytree(original, linked_state)
    optimizer = linked_state / "optimizer.safetensors"
    optimizer.unlink()
    optimizer.symlink_to(original / "optimizer.safetensors")
    with pytest.raises(ValueError, match="symlink"):
        _run(linked, resume=linked_state)


def test_existing_final_and_durable_outputs_are_never_overwritten(tmp_path: Path) -> None:
    complete_base = tmp_path / "complete"
    assert isinstance(_run(complete_base), CompactTrainingResult)
    with pytest.raises(FileExistsError, match="final checkpoint"):
        _run(complete_base)

    stopped_base = tmp_path / "stopped"
    assert isinstance(
        _run(stopped_base, stop_requested=lambda step: step == 2),
        CompactTrainingStopped,
    )
    with pytest.raises(FileExistsError, match="already exists"):
        _run(stopped_base, stop_requested=lambda step: step == 2)


def test_task_balanced_training_candidate_executes_the_frozen_sampler(tmp_path: Path) -> None:
    result = _run(tmp_path / "balanced", sampling="task_balanced")
    assert isinstance(result, CompactTrainingResult)
    assert result.sampling_strategy == "task_balanced"
    assert result.training_steps == 4


def test_device_resolution_has_explicit_fallback_and_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    fallback = resolve_training_device(_training(device="mps", allow_cpu_fallback=True))
    assert fallback == DeviceResolution(requested="mps", resolved="cpu", fallback_used=True)
    with pytest.raises(TrainingError, match="fallback is disabled"):
        resolve_training_device(_training(device="mps", allow_cpu_fallback=False))
    assert resolve_training_device(_training()) == DeviceResolution(
        requested="cpu", resolved="cpu", fallback_used=False
    )
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_training_device(_training(device="mps")) == DeviceResolution(
        requested="mps", resolved="mps", fallback_used=False
    )
    with pytest.raises(TypeError, match="exact RemediationTraining"):
        resolve_training_device(cast(RemediationTraining, object()))


def test_invalid_evaluation_and_stop_callbacks_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(TrainingError, match="finite non-negative"):
        _run(
            tmp_path / "invalid-evaluation",
            evaluation_callback=lambda _model, _step, _nll: float("nan"),
        )
    with pytest.raises(TrainingError, match="exact boolean"):
        _run(
            tmp_path / "invalid-stop",
            stop_requested=lambda _step: cast(bool, 1),
        )


def test_training_entry_boundary_rejections_are_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sampling_strategy"):
        _run(
            tmp_path / "sampling",
            sampling=cast(SamplingStrategy, "unsupported"),
        )
    with pytest.raises(ValueError, match="candidate_id"):
        _run(tmp_path / "candidate", candidate_id="bad/candidate")
    with pytest.raises(ValueError, match="source_commit"):
        _run(tmp_path / "source", source_commit="NOT-A-COMMIT")
    with pytest.raises(ValueError, match="vocab_size"):
        _run(tmp_path / "vocab-range", vocab_size=7)
    with pytest.raises(ValueError, match="frozen tokenizer"):
        _run(tmp_path / "vocab-binding", vocab_size=511)
    with pytest.raises(ValueError, match="directory name is unsafe"):
        _run(tmp_path / "unsafe-output", output_name="unsafe output")
    with pytest.raises(ValueError, match="SHA-256"):
        _run(tmp_path / "hash", train_inventory_sha256="not-a-hash")
    shared = _examples(6)
    with pytest.raises(ValueError, match="must be disjoint"):
        _run(
            tmp_path / "overlap",
            train_examples=shared,
            validation_examples=shared,
        )
    overlong = replace(
        _examples()[0],
        token_ids=(1, 4, 5, 6, 7, 8, 9, 10, 2),
        target_mask=(False, False, False, False, False, False, False, True, True),
        prompt_token_count=7,
        prompt_tokens_retained=7,
    )
    with pytest.raises(ValueError, match="exceeds the model context"):
        _run(tmp_path / "context", train_examples=(overlong,))
    out_of_vocab = replace(_examples()[0], token_ids=(1, 4, 700, 2))
    with pytest.raises(ValueError, match="out-of-vocabulary"):
        _run(tmp_path / "token", train_examples=(out_of_vocab,))
    with pytest.raises(TrainingError, match="starting timestamp"):
        _run(
            tmp_path / "clock",
            monotonic_clock=lambda: float("nan"),
        )
    assert isinstance(_run(tmp_path / "clock"), CompactTrainingResult)


def test_training_module_has_no_numpy_pickle_or_global_sampler_rng_import() -> None:
    source = Path(__file__).parents[2] / "src/reactorbench/remediation/training.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            roots.add(node.module.partition(".")[0])
    assert roots.isdisjoint({"numpy", "pickle", "random"})
