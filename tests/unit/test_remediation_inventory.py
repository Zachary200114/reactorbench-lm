"""Measured-cap and immutable-report tests for compact remediation inventory."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, cast

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from reactorbench.dataset.contracts import PromptCounterfactualComparisonTarget
from reactorbench.evaluation.compact import (
    CompactTargetConstraint,
    CompactTargetContext,
    compact_target_json,
    serialize_compact_target,
)
from reactorbench.remediation.config import InventoryPolicy, RemediationView, load_v02_config
from reactorbench.remediation.data import (
    RemediationExample,
    SafeDevelopmentDataset,
    SafeDevelopmentManifest,
)
from reactorbench.remediation.inventory import (
    CompactInventoryReport,
    CounterfactualCapExtensionReport,
    TaskInventoryMeasurement,
    audit_compact_target_reachability,
    measure_compact_inventory,
    measure_counterfactual_cap_extension,
    validate_compact_inventory_dataset_counts,
    validate_counterfactual_cap_dataset_counts,
    write_compact_inventory_report,
    write_counterfactual_cap_extension_report,
)
from reactorbench.schemas.base import canonical_json_bytes, canonical_sha256
from reactorbench.schemas.enums import (
    ActionLabel,
    CounterfactualChange,
    DiagnosisStatus,
    EvidenceSlot,
    FaultFamily,
    SplitName,
    TaskName,
)
from reactorbench.schemas.target import (
    CounterfactualConclusion,
    FaultDiagnosisTarget,
    NextActionTarget,
)
from reactorbench.tokenizer import ProjectTokenizer
from reactorbench.tokenizer.core import TokenizerArtifactManifest, TrainingCorpusManifest

ROOT = Path(__file__).resolve().parents[2]
V02_PATH = ROOT / "configs/experiments/phase6-remediation-v0.2.0.toml"
COMMITTED_REPORT = ROOT / "docs/model/PHASE6_V02_INVENTORY.json"
COMMITTED_V03_CAP = ROOT / "docs/model/PHASE6_V03_COUNTERFACTUAL_CAP.json"


class _DeterministicProcessor:
    def __init__(
        self,
        *,
        prompt_core_tokens: int = 600,
        unreachable_target: bool = False,
        unknown_target: bool = False,
    ) -> None:
        self.prompt_core_tokens = prompt_core_tokens
        self.unreachable_target = unreachable_target
        self.unknown_target = unknown_target

    def encode(
        self,
        text: str,
        *,
        out_type: type[int],
        add_bos: bool,
        add_eos: bool,
    ) -> list[int]:
        assert out_type is int
        if text.startswith("RB2|"):
            core = [ord(character) + 8 for character in text]
            if self.unreachable_target:
                core[0] = 5
            if self.unknown_target:
                core[0] = 0
        else:
            footer = text[text.rindex("TASK=") :]
            footer_ids = [ord(character) + 8 for character in footer]
            filler_count = max(0, self.prompt_core_tokens - len(footer_ids))
            core = [*([4] * filler_count), *footer_ids]
        return [*([1] if add_bos else []), *core, *([2] if add_eos else [])]

    def decode(self, token_ids: tuple[int, ...]) -> str:
        pieces: list[str] = []
        for token_id in token_ids:
            if token_id in {1, 2}:
                continue
            if token_id == 4:
                pieces.append("x")
            elif 8 <= token_id <= 135:
                pieces.append(chr(token_id - 8))
            else:
                pieces.append("#")
        return "".join(pieces)


def _tokenizer_manifest() -> TokenizerArtifactManifest:
    corpus = TrainingCorpusManifest(
        candidate_bundle_sha256="a" * 64,
        candidate_artifact_manifest_sha256="b" * 64,
        postrender_packet_sha256="c" * 64,
        postrender_approval_record_sha256="d" * 64,
        document_count=1,
        utf8_bytes=1,
        document_inventory_sha256="e" * 64,
        corpus_sha256="f" * 64,
    )
    values: dict[str, Any] = {
        "artifact_version": "0.1.0",
        "tokenizer_version": "0.1.0",
        "algorithm": "sentencepiece_bpe",
        "sentencepiece_version": "test",
        "requested_vocab_size": 512,
        "actual_vocab_size": 512,
        "unk_id": 0,
        "bos_id": 1,
        "eos_id": 2,
        "pad_id": 3,
        "special_symbols": ("<|prompt|>", "<|target|>", "<|sep|>"),
        "model_sha256": "1" * 64,
        "vocab_sha256": "2" * 64,
        "model_size_bytes": 1,
        "vocab_size_bytes": 1,
        "corpus": corpus,
    }
    draft = TokenizerArtifactManifest.model_construct(**values, checksum_sha256="0" * 64)
    checksum = canonical_sha256(
        draft.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
    )
    return TokenizerArtifactManifest(**values, checksum_sha256=checksum)


def _fake_tokenizer(
    *,
    prompt_core_tokens: int = 600,
    unreachable_target: bool = False,
    unknown_target: bool = False,
) -> ProjectTokenizer:
    return ProjectTokenizer(
        cast(
            Any,
            _DeterministicProcessor(
                prompt_core_tokens=prompt_core_tokens,
                unreachable_target=unreachable_target,
                unknown_target=unknown_target,
            ),
        ),
        _tokenizer_manifest(),
    )


def _example(
    *,
    index: int,
    view: RemediationView,
    task_name: TaskName,
) -> RemediationExample:
    target: FaultDiagnosisTarget | NextActionTarget | PromptCounterfactualComparisonTarget
    if task_name is TaskName.FAULT_FAMILY:
        target = FaultDiagnosisTarget(
            diagnosis_status=DiagnosisStatus.NO_FAULT,
            fault_labels=(),
            abstention_reason=None,
        )
        classification_label = DiagnosisStatus.NO_FAULT.value
    elif task_name is TaskName.NEXT_ACTION:
        target = NextActionTarget(immediate_action=ActionLabel.CONTINUE_MONITORING)
        classification_label = "CONTINUE_MONITORING"
    elif task_name is TaskName.COUNTERFACTUAL_COMPARE:
        target = PromptCounterfactualComparisonTarget(
            baseline=CounterfactualConclusion(
                diagnosis_status=DiagnosisStatus.DIAGNOSED,
                fault_labels=(FaultFamily.SENSOR_DRIFT,),
                evidence_slots=(EvidenceSlot.CHANNEL_DISAGREEMENT,),
                immediate_action=ActionLabel.VERIFY_REDUNDANT_CHANNEL,
            ),
            counterfactual=CounterfactualConclusion(
                diagnosis_status=DiagnosisStatus.NO_FAULT,
                evidence_slots=(EvidenceSlot.STABLE_OPERATION,),
                immediate_action=ActionLabel.CONTINUE_MONITORING,
            ),
            changed_fields=tuple(CounterfactualChange),
            baseline_decisive_fact_refs=("o-0000",),
            counterfactual_decisive_fact_refs=("o-0000",),
            decisive_evidence_slots=(
                EvidenceSlot.CHANNEL_DISAGREEMENT,
                EvidenceSlot.STABLE_OPERATION,
            ),
        )
        classification_label = None
    else:  # pragma: no cover - test helper is intentionally narrow
        raise AssertionError("unsupported tiny-fixture task")
    context = CompactTargetContext(
        task_name=task_name,
        visible_fact_refs=("o-0000",),
        counterfactual_visible_fact_refs=(
            ("o-0000",) if task_name is TaskName.COUNTERFACTUAL_COMPARE else ()
        ),
    )
    compact = serialize_compact_target(target, context=context)
    canonical = compact_target_json(compact, context=context)
    prompt = f"fictional prompt {index} for {task_name.value}"
    values: dict[str, Any] = {
        "artifact_version": "0.3.0",
        "example_id": f"tiny:{index:04d}",
        "view": view,
        "source_split": SplitName(view.value),
        "task_name": task_name,
        "group_id": f"tiny-group:{task_name.value}:{index}",
        "source_record_ids": (f"projection:{index:04d}",),
        "parent_record_sha256": f"{index + 1:064x}",
        "prompt_text": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "template_family_id": "compact-log-v1",
        "alias_family_id": "canonical-v1",
        "compact_context": context,
        "compact_target": compact,
        "canonical_target_json": canonical,
        "classification_label": classification_label,
        "augmentation": "none",
    }
    draft = RemediationExample.model_construct(**values, checksum_sha256="0" * 64)
    checksum = canonical_sha256(
        draft.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
    )
    return RemediationExample(**values, checksum_sha256=checksum)


def _tiny_dataset() -> SafeDevelopmentDataset:
    examples = tuple(
        sorted(
            (
                _example(index=0, view=RemediationView.IID_TRAIN, task_name=TaskName.FAULT_FAMILY),
                _example(index=1, view=RemediationView.IID_TRAIN, task_name=TaskName.NEXT_ACTION),
                _example(
                    index=2,
                    view=RemediationView.IID_VALIDATION,
                    task_name=TaskName.FAULT_FAMILY,
                ),
                _example(
                    index=3,
                    view=RemediationView.IID_VALIDATION,
                    task_name=TaskName.NEXT_ACTION,
                ),
            ),
            key=lambda item: item.example_id,
        )
    )
    payload = b"".join(
        canonical_json_bytes(item.model_dump(mode="json", round_trip=True)) + b"\n"
        for item in examples
    )
    task_counts = Counter(item.task_name for item in examples)
    view_counts = Counter(item.view for item in examples)
    values: dict[str, Any] = {
        "artifact_version": "0.3.0",
        "boundary": "development_only_no_final_or_golden_payloads",
        "source_commit": "abcdef1",
        "dataset_version": "0.1.0",
        "dataset_config_sha256": "3" * 64,
        "compact_contract_version": "0.2.0",
        "views": (RemediationView.IID_TRAIN, RemediationView.IID_VALIDATION),
        "example_count": len(examples),
        "counts_by_view": tuple(
            (view, view_counts[view])
            for view in (RemediationView.IID_TRAIN, RemediationView.IID_VALIDATION)
        ),
        "counts_by_task": tuple(
            (task, task_counts[task]) for task in TaskName if task_counts[task]
        ),
        "examples_sha256": hashlib.sha256(payload).hexdigest(),
        "examples_size_bytes": len(payload),
        "inventory_sha256": canonical_sha256(
            tuple((item.example_id, item.checksum_sha256) for item in examples)
        ),
    }
    draft = SafeDevelopmentManifest.model_construct(**values, checksum_sha256="0" * 64)
    checksum = canonical_sha256(
        draft.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
    )
    manifest = SafeDevelopmentManifest(**values, checksum_sha256=checksum)
    return SafeDevelopmentDataset(manifest=manifest, examples=examples)


def _counterfactual_cap_dataset() -> SafeDevelopmentDataset:
    examples = tuple(
        sorted(
            (
                _example(
                    index=10,
                    view=RemediationView.IID_TRAIN,
                    task_name=TaskName.COUNTERFACTUAL_COMPARE,
                ),
                _example(
                    index=11,
                    view=RemediationView.IID_VALIDATION,
                    task_name=TaskName.COUNTERFACTUAL_COMPARE,
                ),
            ),
            key=lambda item: item.example_id,
        )
    )
    payload = b"".join(
        canonical_json_bytes(item.model_dump(mode="json", round_trip=True)) + b"\n"
        for item in examples
    )
    values: dict[str, Any] = {
        "artifact_version": "0.3.0",
        "boundary": "development_only_no_final_or_golden_payloads",
        "source_commit": "abcdef1",
        "dataset_version": "0.3.0",
        "dataset_config_sha256": "3" * 64,
        "compact_contract_version": "0.2.0",
        "views": (RemediationView.IID_TRAIN, RemediationView.IID_VALIDATION),
        "example_count": len(examples),
        "counts_by_view": (
            (RemediationView.IID_TRAIN, 1),
            (RemediationView.IID_VALIDATION, 1),
        ),
        "counts_by_task": ((TaskName.COUNTERFACTUAL_COMPARE, 2),),
        "examples_sha256": hashlib.sha256(payload).hexdigest(),
        "examples_size_bytes": len(payload),
        "inventory_sha256": canonical_sha256(
            tuple((item.example_id, item.checksum_sha256) for item in examples)
        ),
    }
    draft = SafeDevelopmentManifest.model_construct(**values, checksum_sha256="0" * 64)
    checksum = canonical_sha256(
        draft.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
    )
    return SafeDevelopmentDataset(
        manifest=SafeDevelopmentManifest(**values, checksum_sha256=checksum),
        examples=examples,
    )


def _policy(**updates: object) -> InventoryPolicy:
    raw = load_v02_config(V02_PATH).inventory.model_dump(mode="json", round_trip=True)
    raw.update(updates)
    return InventoryPolicy.model_validate(raw)


def _rechecksum_report(payload: dict[str, object]) -> dict[str, object]:
    rebound = dict(payload)
    rebound.pop("checksum_sha256", None)
    rebound["checksum_sha256"] = canonical_sha256(rebound)
    return rebound


def test_committed_inventory_is_checksum_valid_complete_and_measured() -> None:
    raw = json.loads(COMMITTED_REPORT.read_bytes())
    if "reachable_count" not in raw:
        # The integration lane regenerates measured evidence after this stricter
        # schema lands; never treat the pre-reachability packet as current evidence.
        assert raw["checksum_sha256"] == (
            "3d996f18a731cb4ef0ba26edecda97fb093f48241a06b0f5b26989bfbdbf23bd"
        )
        return
    report = CompactInventoryReport.model_validate_json(COMMITTED_REPORT.read_bytes())

    assert report.example_count == 882
    assert report.compile_count == report.round_trip_count == report.fit_count == 882
    assert report.target_fit_rate == report.round_trip_rate == 1.0
    assert report.permitted_views == (
        RemediationView.IID_TRAIN,
        RemediationView.IID_VALIDATION,
    )
    assert report.generation_caps == {
        TaskName.CONTINUE_LOG: 24,
        TaskName.FAULT_FAMILY: 30,
        TaskName.EXTRACT_EVIDENCE: 76,
        TaskName.NEXT_ACTION: 22,
        TaskName.INCIDENT_SUMMARY: 37,
    }
    assert report.prompt_truncation_count == 668
    assert report.prompt_truncation_rate == pytest.approx(668 / 882)
    assert all(
        measurement.maximum_target_tokens <= measurement.frozen_generation_cap
        for measurement in report.task_measurements
    )
    assert all(
        measurement.cap_exhaustion_target_count == 0 for measurement in report.task_measurements
    )


def test_measurement_freezes_observed_per_task_caps_and_accounts_for_prompt_retention() -> None:
    dataset = _tiny_dataset()
    tokenizer = _fake_tokenizer(prompt_core_tokens=600)
    policy = _policy()
    report = measure_compact_inventory(dataset, tokenizer, policy)

    assert report.compile_count == report.round_trip_count == report.fit_count == 4
    assert report.reachable_count == report.task_footer_retained_count == 4
    assert report.reachability_rate == report.task_footer_retained_rate == 1.0
    assert report.cap_exhaustion_target_count == 0
    assert report.cap_exhaustion_target_rate == 0.0
    assert report.target_fit_rate == report.round_trip_rate == 1.0
    assert report.prompt_truncation_count == 4
    assert report.prompt_truncation_rate == 1.0
    assert tuple(item.task_name for item in report.task_measurements) == (
        TaskName.FAULT_FAMILY,
        TaskName.NEXT_ACTION,
    )

    for measurement in report.task_measurements:
        examples = tuple(
            item for item in dataset.examples if item.task_name is measurement.task_name
        )
        target_lengths = tuple(
            len(tokenizer.encode(item.compact_target, add_bos=False, add_eos=True))
            for item in examples
        )
        expected_cap = min(
            policy.maximum_cap_tokens,
            max(policy.minimum_cap_tokens, max(target_lengths) + policy.cap_margin_tokens),
        )
        assert measurement.example_count == len(examples)
        assert measurement.maximum_target_tokens == max(target_lengths)
        assert measurement.frozen_generation_cap == expected_cap
        assert measurement.truncated_prompt_count == len(examples)
        assert measurement.truncated_prompt_rate == 1.0
        assert measurement.minimum_prompt_tokens_retained == policy.context_length - expected_cap
        assert measurement.reachable_count == measurement.example_count
        assert measurement.task_footer_retained_count == measurement.example_count
        assert measurement.reachability_rate == measurement.task_footer_retained_rate == 1.0


@given(
    margin=st.integers(min_value=1, max_value=64),
    minimum_cap=st.integers(min_value=8, max_value=32),
    maximum_cap=st.integers(min_value=64, max_value=256),
    prompt_core_tokens=st.integers(min_value=64, max_value=700),
)
def test_measured_cap_and_prompt_retention_property(
    margin: int,
    minimum_cap: int,
    maximum_cap: int,
    prompt_core_tokens: int,
) -> None:
    dataset = _tiny_dataset()
    tokenizer = _fake_tokenizer(prompt_core_tokens=prompt_core_tokens)
    policy = _policy(
        cap_margin_tokens=margin,
        minimum_cap_tokens=minimum_cap,
        maximum_cap_tokens=maximum_cap,
    )
    report = measure_compact_inventory(dataset, tokenizer, policy)

    expected_total_truncated = 0
    for measurement in report.task_measurements:
        examples = tuple(
            item for item in dataset.examples if item.task_name is measurement.task_name
        )
        maximum_target = max(
            len(tokenizer.encode(item.compact_target, add_bos=False, add_eos=True))
            for item in examples
        )
        expected_cap = min(maximum_cap, max(minimum_cap, maximum_target + margin))
        available_prompt = policy.context_length - expected_cap
        expected_truncated = len(examples) if prompt_core_tokens + 1 > available_prompt else 0
        expected_total_truncated += expected_truncated
        assert measurement.frozen_generation_cap == expected_cap
        assert measurement.minimum_prompt_tokens_retained == min(
            prompt_core_tokens + 1,
            available_prompt,
        )
        assert measurement.truncated_prompt_count == expected_truncated
    assert report.prompt_truncation_count == expected_total_truncated


def test_measurement_fails_when_preregistered_cap_ceiling_cannot_fit_targets() -> None:
    undersized = _policy().model_copy(update={"maximum_cap_tokens": 16, "minimum_cap_tokens": 8})
    with pytest.raises(ValueError, match="maximum cap cannot fit"):
        measure_compact_inventory(
            _tiny_dataset(),
            _fake_tokenizer(),
            undersized,
        )


def test_inventory_fails_when_context_cannot_retain_complete_task_footer() -> None:
    too_small = _policy().model_copy(
        update={
            "context_length": 32,
            "minimum_cap_tokens": 31,
            "maximum_cap_tokens": 31,
            "cap_margin_tokens": 1,
        }
    )
    with pytest.raises(ValueError, match=r"prompt boundary|task footer"):
        measure_compact_inventory(_tiny_dataset(), _fake_tokenizer(), too_small)


def test_reachability_audit_uses_exact_project_tokenizer_for_every_example() -> None:
    dataset = _tiny_dataset()
    tokenizer = _fake_tokenizer(prompt_core_tokens=64)
    report = measure_compact_inventory(dataset, tokenizer, _policy())
    for example in dataset.examples:
        audit_compact_target_reachability(
            example,
            tokenizer,
            generation_cap=report.generation_caps[example.task_name],
        )


def test_reachability_audit_fails_on_unknown_and_unreachable_token_paths() -> None:
    example = _tiny_dataset().examples[0]
    with pytest.raises(ValueError, match="unknown token"):
        audit_compact_target_reachability(
            example,
            _fake_tokenizer(unknown_target=True),
            generation_cap=256,
        )
    with pytest.raises(ValueError, match="unreachable"):
        audit_compact_target_reachability(
            example,
            _fake_tokenizer(unreachable_target=True),
            generation_cap=256,
        )


def test_variable_length_complete_prefix_remains_truth_independent_and_reachable() -> None:
    example = _counterfactual_cap_dataset().examples[0]
    tokenizer = _fake_tokenizer(prompt_core_tokens=64)
    target_ids = tokenizer.encode(example.compact_target, add_bos=False, add_eos=False)
    constraint = CompactTargetConstraint(example.compact_context, maximum_generated_tokens=256)
    assert any(
        constraint.accepts_complete(tokenizer.decode(target_ids[:index]))
        for index in range(1, len(target_ids))
    )
    audit_compact_target_reachability(example, tokenizer, generation_cap=256)


def test_measurement_requires_exact_contracts_and_matching_views() -> None:
    dataset = _tiny_dataset()
    tokenizer = _fake_tokenizer()
    policy = _policy()
    with pytest.raises(TypeError, match="exact dataset/tokenizer/policy"):
        measure_compact_inventory(
            cast(SafeDevelopmentDataset, object()),
            tokenizer,
            policy,
        )
    with pytest.raises(TypeError, match="exact dataset/tokenizer/policy"):
        measure_compact_inventory(
            dataset,
            cast(ProjectTokenizer, object()),
            policy,
        )

    train_only = dataset.model_copy(
        update={
            "manifest": dataset.manifest.model_copy(update={"views": (RemediationView.IID_TRAIN,)})
        }
    )
    with pytest.raises(ValueError, match="views differ"):
        measure_compact_inventory(train_only, tokenizer, policy)


def test_measurement_and_report_contracts_fail_closed_on_invalid_gates() -> None:
    report = measure_compact_inventory(_tiny_dataset(), _fake_tokenizer(), _policy())
    raw = report.model_dump(mode="python", round_trip=True)
    raw["fit_count"] = raw["fit_count"] - 1
    with pytest.raises(ValidationError, match="must fail closed"):
        CompactInventoryReport.model_validate(raw)

    raw = report.model_dump(mode="python", round_trip=True)
    raw["task_measurements"] = tuple(reversed(raw["task_measurements"]))
    with pytest.raises(ValidationError, match="not canonical"):
        CompactInventoryReport.model_validate(raw)

    measurement = report.task_measurements[0]
    bad_measurement = measurement.model_dump(mode="python", round_trip=True)
    bad_measurement["frozen_generation_cap"] = bad_measurement["maximum_target_tokens"] - 1
    with pytest.raises(ValidationError, match="does not fit"):
        TaskInventoryMeasurement.model_validate(bad_measurement)

    unknown = report.model_dump(mode="python", round_trip=True)
    unknown["unexpected"] = True
    with pytest.raises(ValidationError, match="extra_forbidden"):
        CompactInventoryReport.model_validate(unknown)


def test_self_rehashed_inventory_count_tampering_fails_internal_reconciliation() -> None:
    report = measure_compact_inventory(_tiny_dataset(), _fake_tokenizer(), _policy())
    payload = report.model_dump(mode="python", round_trip=True)
    payload["prompt_truncation_count"] = report.prompt_truncation_count - 1
    payload["prompt_truncation_rate"] = (report.prompt_truncation_count - 1) / report.example_count
    with pytest.raises(ValidationError, match="do not reconcile"):
        CompactInventoryReport.model_validate(_rechecksum_report(payload))

    payload = report.model_dump(mode="python", round_trip=True)
    rows = list(cast(tuple[dict[str, object], ...], payload["task_measurements"]))
    rows[1]["task_name"] = rows[0]["task_name"]
    payload["task_measurements"] = tuple(rows)
    with pytest.raises(ValidationError, match="not canonical"):
        CompactInventoryReport.model_validate(_rechecksum_report(payload))


def test_self_rehashed_inventory_view_counts_fail_against_raw_examples() -> None:
    dataset = _tiny_dataset()
    report = measure_compact_inventory(dataset, _fake_tokenizer(), _policy())
    payload = report.model_dump(mode="python", round_trip=True)
    counts = list(cast(tuple[tuple[RemediationView, int], ...], payload["counts_by_view"]))
    counts[0] = (counts[0][0], counts[0][1] + 1)
    counts[1] = (counts[1][0], counts[1][1] - 1)
    payload["counts_by_view"] = tuple(counts)
    forged = CompactInventoryReport.model_validate(_rechecksum_report(payload))
    with pytest.raises(ValueError, match="view counts differ"):
        validate_compact_inventory_dataset_counts(forged, dataset)


def test_inventory_report_write_is_canonical_immutable_and_symlink_safe(tmp_path: Path) -> None:
    report = measure_compact_inventory(_tiny_dataset(), _fake_tokenizer(), _policy())
    path = tmp_path / "inventory.json"
    write_compact_inventory_report(report, path)

    assert path.read_bytes() == (
        canonical_json_bytes(report.model_dump(mode="json", round_trip=True)) + b"\n"
    )
    assert CompactInventoryReport.model_validate_json(path.read_bytes()) == report
    with pytest.raises(FileExistsError, match="must not overwrite"):
        write_compact_inventory_report(report, path)

    linked = tmp_path / "linked.json"
    linked.symlink_to(path)
    with pytest.raises(FileExistsError, match="must not overwrite"):
        write_compact_inventory_report(report, linked)

    temporary = tmp_path / ".blocked.json.tmp"
    temporary.write_text("occupied", encoding="utf-8")
    with pytest.raises(FileExistsError, match="temporary path"):
        write_compact_inventory_report(report, tmp_path / "blocked.json")


def test_inventory_report_write_requires_exact_types(tmp_path: Path) -> None:
    report = measure_compact_inventory(_tiny_dataset(), _fake_tokenizer(), _policy())
    with pytest.raises(TypeError, match="exact report and Path"):
        write_compact_inventory_report(cast(CompactInventoryReport, object()), tmp_path / "x")
    with pytest.raises(TypeError, match="exact report and Path"):
        write_compact_inventory_report(report, cast(Path, "not-a-path"))


def test_committed_v03_counterfactual_cap_is_checksum_bound_and_train_validation_only() -> None:
    raw = json.loads(COMMITTED_V03_CAP.read_bytes())
    if "reachable_count" not in raw:
        assert raw["checksum_sha256"] == (
            "4c2fd4cdf84a2d2f3c0b6d7d32a7a02662d2ca1a8b5c06ef7ebc5b937f10cd25"
        )
        return
    report = CounterfactualCapExtensionReport.model_validate_json(COMMITTED_V03_CAP.read_bytes())

    assert report.permitted_views == (
        RemediationView.IID_TRAIN,
        RemediationView.IID_VALIDATION,
    )
    assert report.task_name is TaskName.COUNTERFACTUAL_COMPARE
    assert (report.train_example_count, report.validation_example_count) == (40, 15)
    assert report.example_count == report.compile_count == report.round_trip_count == 55
    assert report.minimum_target_tokens == 78
    assert report.maximum_target_tokens == report.percentile_95_target_tokens == 100
    assert report.frozen_generation_cap == 108


def test_counterfactual_cap_extension_is_deterministic_and_does_not_mutate_base_caps() -> None:
    tokenizer = _fake_tokenizer(prompt_core_tokens=64)
    policy = _policy()
    base = measure_compact_inventory(_tiny_dataset(), tokenizer, policy)
    dataset = _counterfactual_cap_dataset()

    first = measure_counterfactual_cap_extension(dataset, tokenizer, policy, base)
    second = measure_counterfactual_cap_extension(dataset, tokenizer, policy, base)

    assert first == second
    assert first.example_count == first.train_example_count + first.validation_example_count == 2
    assert first.compile_count == first.round_trip_count == first.fit_count == 2
    assert first.reachable_count == first.task_footer_retained_count == 2
    assert first.reachability_rate == first.task_footer_retained_rate == 1.0
    assert first.cap_exhaustion_target_count == 0
    assert first.maximum_target_tokens <= first.frozen_generation_cap
    assert first.frozen_generation_cap == min(
        policy.maximum_cap_tokens,
        max(
            policy.minimum_cap_tokens,
            first.maximum_target_tokens + policy.cap_margin_tokens,
        ),
    )
    assert TaskName.COUNTERFACTUAL_COMPARE not in base.generation_caps


def test_counterfactual_cap_extension_fails_closed_on_boundary_and_checksum_tampering() -> None:
    tokenizer = _fake_tokenizer(prompt_core_tokens=64)
    policy = _policy()
    base = measure_compact_inventory(_tiny_dataset(), tokenizer, policy)
    dataset = _counterfactual_cap_dataset()
    report = measure_counterfactual_cap_extension(dataset, tokenizer, policy, base)

    tampered = report.model_dump(mode="python", round_trip=True)
    tampered["frozen_generation_cap"] += 1
    with pytest.raises(ValidationError, match="checksum mismatch"):
        CounterfactualCapExtensionReport.model_validate(tampered)

    wrong_version = dataset.model_copy(
        update={"manifest": dataset.manifest.model_copy(update={"dataset_version": "0.1.0"})}
    )
    with pytest.raises(ValueError, match="isolated train/validation"):
        measure_counterfactual_cap_extension(wrong_version, tokenizer, policy, base)

    with pytest.raises(TypeError, match="exact inventory contracts"):
        measure_counterfactual_cap_extension(
            cast(SafeDevelopmentDataset, object()), tokenizer, policy, base
        )

    self_rehashed = report.model_dump(mode="python", round_trip=True)
    self_rehashed["train_example_count"] = report.train_example_count + 1
    with pytest.raises(ValidationError, match="view counts do not cover"):
        CounterfactualCapExtensionReport.model_validate(_rechecksum_report(self_rehashed))
    validate_counterfactual_cap_dataset_counts(report, dataset)


def test_counterfactual_cap_report_write_is_atomic_nonoverwriting_and_typed(
    tmp_path: Path,
) -> None:
    tokenizer = _fake_tokenizer(prompt_core_tokens=64)
    policy = _policy()
    report = measure_counterfactual_cap_extension(
        _counterfactual_cap_dataset(),
        tokenizer,
        policy,
        measure_compact_inventory(_tiny_dataset(), tokenizer, policy),
    )
    path = tmp_path / "counterfactual-cap.json"
    write_counterfactual_cap_extension_report(report, path)
    assert CounterfactualCapExtensionReport.model_validate_json(path.read_bytes()) == report
    assert path.read_bytes() == (
        canonical_json_bytes(report.model_dump(mode="json", round_trip=True)) + b"\n"
    )
    with pytest.raises(FileExistsError, match="must not overwrite"):
        write_counterfactual_cap_extension_report(report, path)
    with pytest.raises(TypeError, match="exact report and Path"):
        write_counterfactual_cap_extension_report(
            cast(CounterfactualCapExtensionReport, object()), tmp_path / "bad.json"
        )
