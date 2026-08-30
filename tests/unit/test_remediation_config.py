"""Strict contract tests for the versioned remediation configurations."""

from __future__ import annotations

import copy
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError

from reactorbench.model.config import StrictConfigModel
from reactorbench.remediation.config import (
    MAX_REMEDIATION_CONFIG_BYTES,
    PIPELINE_STAGES,
    SHADOW_VIEWS,
    PipelineConfig,
    RemediationView,
    V02Config,
    V03Config,
    V04Config,
    config_sha256,
    load_v02_config,
    load_v03_config,
    load_v04_config,
)
from reactorbench.schemas.base import canonical_sha256
from reactorbench.schemas.enums import SplitName, TaskName

ROOT = Path(__file__).resolve().parents[2]
V02_PATH = ROOT / "configs/experiments/phase6-remediation-v0.2.0.toml"
V03_PATH = ROOT / "configs/experiments/phase6-remediation-v0.3.0.toml"
TARGETED_V03_PATH = ROOT / "configs/experiments/phase6-remediation-v0.3.1-targeted.toml"
FOCUSED_V03_PATH = ROOT / "configs/experiments/phase6-remediation-v0.3.2-focused.toml"
HIERARCHICAL_V03_PATH = ROOT / "configs/experiments/phase6-remediation-v0.3.3-hierarchical.toml"
FAULT_BOOSTED_V03_PATH = ROOT / "configs/experiments/phase6-remediation-v0.3.4-fault-boosted.toml"
TASK_WEIGHTED_V03_PATH = ROOT / "configs/experiments/phase6-remediation-v0.3.5-task-weighted.toml"
TARGETED_PIPELINE_PATH = (
    ROOT / "configs/experiments/phase6-remediation-pipeline-v0.4.0-targeted-01.toml"
)
FOCUSED_PIPELINE_PATH = (
    ROOT / "configs/experiments/phase6-remediation-pipeline-v0.4.0-targeted-02.toml"
)
HIERARCHICAL_PIPELINE_PATH = (
    ROOT / "configs/experiments/phase6-remediation-pipeline-v0.4.0-targeted-03.toml"
)
FAULT_BOOSTED_PIPELINE_PATH = (
    ROOT / "configs/experiments/phase6-remediation-pipeline-v0.4.0-targeted-04.toml"
)
TASK_WEIGHTED_PIPELINE_PATH = (
    ROOT / "configs/experiments/phase6-remediation-pipeline-v0.4.0-targeted-05.toml"
)
DIAGNOSTIC_PIPELINE_PATH = (
    ROOT / "configs/experiments/phase6-remediation-pipeline-v0.4.0-targeted-05-diagnostic-01.toml"
)
V04_PATH = ROOT / "configs/experiments/phase6-remediation-v0.4.0.toml"


def _raw(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def test_committed_v02_contract_freezes_output_reliability_gate() -> None:
    config = load_v02_config(V02_PATH)

    assert config.iteration_version == "0.2.0"
    assert config.status == "developmental"
    assert config.inventory.permitted_views == (
        RemediationView.IID_TRAIN,
        RemediationView.IID_VALIDATION,
    )
    assert config.inventory.maximum_cap_tokens == 256
    assert config.inventory.require_target_fit_rate == 1.0
    assert config.inventory.require_round_trip_rate == 1.0
    assert config.model.model_dump(mode="python") == {
        "model_version": "0.2.0",
        "layers": 8,
        "width": 384,
        "heads": 8,
        "context_length": 512,
        "feed_forward_multiplier": 4,
        "dropout": 0.1,
        "tie_embeddings": True,
        "bias": True,
    }
    assert config.decoder.compact_contract_version == "0.2.0"
    assert config.decoder.constrained_strategy == "truth_independent_greedy"
    assert config.decoder.report_both_paths is True
    assert config.decoder.maximum_decoder_cache_entries == 4096


def test_committed_v03_contract_freezes_candidates_and_semantic_gate() -> None:
    config = load_v03_config(V03_PATH)

    assert config.iteration_version == "0.3.0"
    assert config.requires_v02_gate is True
    assert tuple(candidate.candidate_id for candidate in config.candidates) == (
        "v02-uniform-control",
        "v03-task-balanced",
    )
    assert tuple(candidate.sampling for candidate in config.candidates) == (
        "uniform_control",
        "task_balanced",
    )
    assert all(candidate.exposure == "teacher_forced_only" for candidate in config.candidates)
    assert config.augmentation.preserve_group_atomicity is True
    assert config.augmentation.include_insufficient_evidence_views is True
    assert config.augmentation.include_counterfactual_pairs is True
    assert config.augmentation.prohibit_target_text_in_prompt is True
    assert (
        config.selection.constrained_schema_validity,
        config.selection.minimum_fault_margin,
        config.selection.minimum_action_margin,
        config.selection.minimum_continuation_macro_f1,
        config.selection.minimum_evidence_f1,
        config.selection.minimum_required_abstention_accuracy,
        config.selection.maximum_no_fault_false_positive_rate,
        config.selection.maximum_expected_calibration_error,
        config.selection.selective_risk_coverage,
        config.selection.maximum_selective_risk,
    ) == (1.0, 0.02, 0.02, 0.9, 0.7, 0.8, 0.1, 0.15, 0.8, 0.2)
    assert (
        config_sha256(config) == "2ca7984455aeba225d7ee5ee1cbec8a64716e748ae45b812f28fee0f51f35487"
    )


def test_targeted_v03_changes_sampling_and_calibration_but_not_thresholds() -> None:
    config = load_v03_config(TARGETED_V03_PATH)

    assert tuple(item.sampling for item in config.candidates) == (
        "task_balanced",
        "task_class_balanced",
    )
    assert tuple(item.seed for item in config.candidates) == (6301, 6302)
    assert config.training.steps == 2000
    assert config.targeted_policy is not None
    assert config.targeted_policy.sampling_metadata_required is True
    assert config.targeted_policy.calibration.calibration_example_limit == 56
    assert (
        config.selection.constrained_schema_validity,
        config.selection.minimum_fault_margin,
        config.selection.minimum_action_margin,
        config.selection.minimum_continuation_macro_f1,
        config.selection.minimum_evidence_f1,
        config.selection.minimum_required_abstention_accuracy,
        config.selection.maximum_no_fault_false_positive_rate,
        config.selection.maximum_expected_calibration_error,
        config.selection.selective_risk_coverage,
        config.selection.maximum_selective_risk,
    ) == (1.0, 0.02, 0.02, 0.9, 0.7, 0.8, 0.1, 0.15, 0.8, 0.2)


def test_targeted_prefix_reuse_has_an_exact_external_evidence_pin() -> None:
    raw = _raw(TARGETED_PIPELINE_PATH)
    config = PipelineConfig.model_validate(raw)
    assert config.reuse_v02_prefix is not None
    assert len(config.reuse_v02_prefix.evidence) == 21
    assert canonical_sha256(config.reuse_v02_prefix.evidence) == (
        config.reuse_v02_prefix.evidence_inventory_sha256
    )

    changed = copy.deepcopy(raw)
    changed["reuse_v02_prefix"]["evidence"][0][1] = "0" * 64
    with pytest.raises(ValidationError, match="inventory checksum"):
        PipelineConfig.model_validate(changed)


def test_focused_v03_is_candidate_only_and_preserves_every_threshold() -> None:
    config = load_v03_config(FOCUSED_V03_PATH)

    assert tuple(item.sampling for item in config.candidates) == ("fault_continuation_focused",)
    assert tuple(item.seed for item in config.candidates) == (6401,)
    assert config.training.steps == 2000
    assert config.training.batch_size == 6
    assert config.targeted_policy is not None
    assert config.targeted_policy.policy_version == "0.3.2-focused"
    assert config.targeted_policy.calibration.policy_version == "0.3.2-focused"
    assert (
        config.selection.constrained_schema_validity,
        config.selection.minimum_fault_margin,
        config.selection.minimum_action_margin,
        config.selection.minimum_continuation_macro_f1,
        config.selection.minimum_evidence_f1,
        config.selection.minimum_required_abstention_accuracy,
        config.selection.maximum_no_fault_false_positive_rate,
        config.selection.maximum_expected_calibration_error,
        config.selection.selective_risk_coverage,
        config.selection.maximum_selective_risk,
    ) == (1.0, 0.02, 0.02, 0.9, 0.7, 0.8, 0.1, 0.15, 0.8, 0.2)
    pipeline = PipelineConfig.model_validate(_raw(FOCUSED_PIPELINE_PATH))
    assert pipeline.run_name == "phase6-remediation-v0.4.0-targeted-02"
    assert pipeline.v03_config_sha256 == config_sha256(config)


def test_hierarchical_v03_restores_every_task_and_preserves_every_threshold() -> None:
    config = load_v03_config(HIERARCHICAL_V03_PATH)

    assert tuple(item.sampling for item in config.candidates) == (
        "hierarchical_task_label_balanced",
    )
    assert tuple(item.seed for item in config.candidates) == (6501,)
    assert config.training.steps == 2500
    assert config.training.batch_size == 6
    assert config.targeted_policy is not None
    assert config.targeted_policy.policy_version == "0.3.3-hierarchical"
    assert config.targeted_policy.calibration.policy_version == "0.3.3-hierarchical"
    assert config.selection.metric == "semantic_floor_then_validation_nll"
    assert config.selection.minimum_checkpoint_semantic_composite == 0.75
    assert (
        config.selection.constrained_schema_validity,
        config.selection.minimum_fault_margin,
        config.selection.minimum_action_margin,
        config.selection.minimum_continuation_macro_f1,
        config.selection.minimum_evidence_f1,
        config.selection.minimum_required_abstention_accuracy,
        config.selection.maximum_no_fault_false_positive_rate,
        config.selection.maximum_expected_calibration_error,
        config.selection.selective_risk_coverage,
        config.selection.maximum_selective_risk,
    ) == (1.0, 0.02, 0.02, 0.9, 0.7, 0.8, 0.1, 0.15, 0.8, 0.2)
    pipeline = PipelineConfig.model_validate(_raw(HIERARCHICAL_PIPELINE_PATH))
    assert pipeline.run_name == "phase6-remediation-v0.4.0-targeted-03"
    assert pipeline.v03_config_sha256 == config_sha256(config)
    assert (
        pipeline.reuse_v02_prefix
        == PipelineConfig.model_validate(_raw(FOCUSED_PIPELINE_PATH)).reuse_v02_prefix
    )


def test_hierarchical_sampling_and_checkpoint_policy_cannot_be_decoupled() -> None:
    raw = _raw(HIERARCHICAL_V03_PATH)
    changed = copy.deepcopy(raw)
    changed["selection"]["metric"] = "frozen_semantic_composite"
    changed["selection"].pop("minimum_checkpoint_semantic_composite")
    with pytest.raises(ValidationError, match="enabled together"):
        V03Config.model_validate(changed)
    changed = copy.deepcopy(raw)
    changed["selection"]["metric"] = "frozen_semantic_composite"
    with pytest.raises(ValidationError, match="cannot add a checkpoint floor"):
        V03Config.model_validate(changed)


def test_fault_boosted_v03_is_bounded_and_preserves_every_threshold() -> None:
    config = load_v03_config(FAULT_BOOSTED_V03_PATH)

    assert tuple(item.sampling for item in config.candidates) == ("fault_boosted_hierarchical",)
    assert tuple(item.seed for item in config.candidates) == (6701,)
    assert config.training.steps == 2500
    assert config.training.batch_size == 7
    assert config.targeted_policy is not None
    assert config.targeted_policy.policy_version == "0.3.4-fault-boosted"
    assert config.targeted_policy.calibration.policy_version == "0.3.4-fault-boosted"
    assert config.selection.metric == "semantic_floor_then_validation_nll"
    assert config.selection.minimum_checkpoint_semantic_composite == 0.75
    assert (
        config.selection.constrained_schema_validity,
        config.selection.minimum_fault_margin,
        config.selection.minimum_action_margin,
        config.selection.minimum_continuation_macro_f1,
        config.selection.minimum_evidence_f1,
        config.selection.minimum_required_abstention_accuracy,
        config.selection.maximum_no_fault_false_positive_rate,
        config.selection.maximum_expected_calibration_error,
        config.selection.selective_risk_coverage,
        config.selection.maximum_selective_risk,
    ) == (1.0, 0.02, 0.02, 0.9, 0.7, 0.8, 0.1, 0.15, 0.8, 0.2)
    pipeline = PipelineConfig.model_validate(_raw(FAULT_BOOSTED_PIPELINE_PATH))
    assert pipeline.run_name == "phase6-remediation-v0.4.0-targeted-04"
    assert pipeline.v03_config_sha256 == config_sha256(config)
    assert (
        pipeline.reuse_v02_prefix
        == PipelineConfig.model_validate(_raw(HIERARCHICAL_PIPELINE_PATH)).reuse_v02_prefix
    )


def test_task_weighted_v03_restores_class_mix_and_adds_task_selection_floors() -> None:
    config = load_v03_config(TASK_WEIGHTED_V03_PATH)

    assert tuple(item.sampling for item in config.candidates) == ("task_weighted_hierarchical",)
    assert tuple(item.seed for item in config.candidates) == (6801,)
    assert config.training.steps == 2500
    assert config.training.batch_size == 6
    assert config.targeted_policy is not None
    assert config.targeted_policy.policy_version == "0.3.5-task-weighted"
    assert config.targeted_policy.calibration.policy_version == "0.3.5-task-weighted"
    assert config.selection.metric == "task_floor_then_validation_nll"
    assert config.selection.minimum_checkpoint_semantic_composite == 0.75
    assert config.selection.minimum_checkpoint_fault_macro_f1 == 0.9
    assert config.selection.minimum_checkpoint_continuation_macro_f1 == 0.9
    assert config_sha256(config) == (
        "87e1b5f9730d0ed2515607e1adf3cc5ae0d26448df563a577a2500cd5d79e04e"
    )
    assert (
        config.selection.minimum_fault_margin,
        config.selection.minimum_continuation_macro_f1,
    ) == (0.02, 0.9)
    pipeline = PipelineConfig.model_validate(_raw(TASK_WEIGHTED_PIPELINE_PATH))
    assert pipeline.run_name == "phase6-remediation-v0.4.0-targeted-05"
    assert pipeline.v03_config_sha256 == config_sha256(config)
    assert config_sha256(pipeline) == (
        "bfdb3833fce80fd31e018126b1ae225e04cf85d7fcbd9ef0fcada69a69f4a354"
    )
    assert (
        pipeline.reuse_v02_prefix
        == PipelineConfig.model_validate(_raw(FAULT_BOOSTED_PIPELINE_PATH)).reuse_v02_prefix
    )


def test_diagnostic_pipeline_has_a_separate_identity_and_preserves_official_hash() -> None:
    official = PipelineConfig.model_validate(_raw(TASK_WEIGHTED_PIPELINE_PATH))
    diagnostic = PipelineConfig.model_validate(_raw(DIAGNOSTIC_PIPELINE_PATH))

    assert official.diagnostic_mode is None
    assert config_sha256(official) == (
        "bfdb3833fce80fd31e018126b1ae225e04cf85d7fcbd9ef0fcada69a69f4a354"
    )
    assert diagnostic.run_name == "phase6-remediation-v0.4.0-targeted-05-diagnostic-01"
    assert diagnostic.diagnostic_mode == "collect_scientific_failures"
    assert diagnostic.stop_before_final_evaluation is True
    assert diagnostic.v03_config_sha256 == official.v03_config_sha256
    assert diagnostic.reuse_v02_prefix == official.reuse_v02_prefix

    changed = _raw(DIAGNOSTIC_PIPELINE_PATH)
    changed["run_name"] = "unsafe-diagnostic-name"
    with pytest.raises(ValidationError, match="exact non-overwriting run identity"):
        PipelineConfig.model_validate(changed)


def test_task_weighted_objective_and_selection_cannot_be_decoupled() -> None:
    raw = _raw(TASK_WEIGHTED_V03_PATH)
    changed = copy.deepcopy(raw)
    changed["selection"]["metric"] = "semantic_floor_then_validation_nll"
    changed["selection"].pop("minimum_checkpoint_fault_macro_f1")
    changed["selection"].pop("minimum_checkpoint_continuation_macro_f1")
    with pytest.raises(ValidationError, match="must match"):
        V03Config.model_validate(changed)

    changed = copy.deepcopy(raw)
    changed["selection"]["minimum_checkpoint_fault_macro_f1"] = 0.89
    with pytest.raises(ValidationError, match="differ from targeted-05"):
        V03Config.model_validate(changed)

    changed = copy.deepcopy(raw)
    changed["candidates"][0]["sampling"] = "fault_boosted_hierarchical"
    with pytest.raises(ValidationError, match="task-weighted candidate"):
        V03Config.model_validate(changed)


def test_committed_v04_contract_freezes_shadow_and_final_access_boundaries() -> None:
    config = load_v04_config(V04_PATH)

    assert config.iteration_version == "0.4.0"
    assert config.requires_v03_gate is True
    assert config.shadow.required_views == SHADOW_VIEWS
    assert set(config.shadow.required_views).isdisjoint(
        {RemediationView.IID_TRAIN, RemediationView.IID_VALIDATION}
    )
    assert config.shadow.worst_split_rule == "all_required_views_must_pass"
    assert config.shadow.source_groups_must_be_disjoint is True
    assert config.shadow.content_checksums_must_be_disjoint is True
    assert config.development_dataset_config_path != config.final_dataset_config_path
    assert config.variants.default_context_length == 512
    assert config.variants.optional_context_length == 1024
    assert config.variants.maximum_capacity_variants == 1
    assert config.variants.context_candidate_selection_rule == (
        "all_gates_then_highest_min_view_composite_then_iid_composite_then_shorter_context"
    )
    assert config.pilot.batch_sizes == (1, 2, 4)
    assert config.training.batch_size in config.pilot.batch_sizes
    assert config.final_access.automatically_run_final_evaluation is False
    assert config.final_access.require_ready_marker is True
    assert config.final_access.require_owner_review is True
    assert config.final_access.require_explicit_confirm_flag is True
    assert config.final_access.one_access_only is True
    assert config.final_access.historical_golden_packet_permitted is False


@pytest.mark.parametrize(
    ("model", "path", "loader"),
    [
        (V02Config, V02_PATH, load_v02_config),
        (V03Config, V03_PATH, load_v03_config),
        (V04Config, V04_PATH, load_v04_config),
    ],
)
def test_unknown_top_level_fields_are_rejected(
    model: type[V02Config] | type[V03Config] | type[V04Config],
    path: Path,
    loader: Callable[[Path], object],
    tmp_path: Path,
) -> None:
    raw = _raw(path)
    raw["unexpected"] = "forbidden"
    with pytest.raises(ValidationError, match="extra_forbidden"):
        model.model_validate(raw)

    candidate = tmp_path / path.name
    candidate.write_text(path.read_text(encoding="utf-8") + '\nunexpected = "forbidden"\n')
    with pytest.raises(ValidationError, match="extra_forbidden"):
        loader(candidate)


@pytest.mark.parametrize("bad_path", ["/absolute", "../escape", "a//b", "a\\b", "./a"])
def test_every_remediation_path_is_canonical_and_project_relative(bad_path: str) -> None:
    raw = _raw(V02_PATH)
    raw["paths"]["run_root"] = bad_path
    with pytest.raises(ValidationError, match="project paths"):
        V02Config.model_validate(raw)

    v04 = _raw(V04_PATH)
    v04["compact_contract_path"] = bad_path
    with pytest.raises(ValidationError, match="project paths"):
        V04Config.model_validate(v04)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("training", "steps", "1500"),
        ("training", "batch_size", True),
        ("training", "learning_rate", 1),
        ("inventory", "require_target_fit_rate", 1),
        ("decoder", "report_both_paths", 1),
    ],
)
def test_v02_strict_types_do_not_coerce(section: str, field: str, value: object) -> None:
    raw = _raw(V02_PATH)
    raw[section][field] = value
    with pytest.raises(ValidationError):
        V02Config.model_validate(raw)


def test_v02_architecture_inventory_and_schedule_gates_fail_closed() -> None:
    architecture = _raw(V02_PATH)
    architecture["model"]["width"] = 512
    with pytest.raises(ValidationError, match="control architecture"):
        V02Config.model_validate(architecture)

    fit = _raw(V02_PATH)
    fit["inventory"]["require_target_fit_rate"] = 0.99
    with pytest.raises(ValidationError, match=r"must remain 1\.0"):
        V02Config.model_validate(fit)

    views = _raw(V02_PATH)
    views["inventory"]["permitted_views"] = ["iid_validation", "iid_train"]
    with pytest.raises(ValidationError, match="only IID train and validation"):
        V02Config.model_validate(views)

    schedule = _raw(V02_PATH)
    schedule["training"]["steps"] = 1501
    with pytest.raises(ValidationError, match="must align"):
        V02Config.model_validate(schedule)


def test_v03_candidate_matrix_augmentation_and_thresholds_fail_closed() -> None:
    missing = _raw(V03_PATH)
    missing["candidates"] = missing["candidates"][:1]
    with pytest.raises(ValidationError, match="freezes the control and task-balanced"):
        V03Config.model_validate(missing)

    reversed_candidates = _raw(V03_PATH)
    reversed_candidates["candidates"] = list(reversed(reversed_candidates["candidates"]))
    with pytest.raises(ValidationError, match="freezes the control and task-balanced"):
        V03Config.model_validate(reversed_candidates)

    duplicate_family = _raw(V03_PATH)
    duplicate_family["augmentation"]["train_alias_families"] = [
        "canonical-v1",
        "canonical-v1",
    ]
    with pytest.raises(ValidationError, match="non-empty and unique"):
        V03Config.model_validate(duplicate_family)

    threshold = _raw(V03_PATH)
    threshold["selection"]["minimum_evidence_f1"] = 0.69
    with pytest.raises(ValidationError, match="differ from the preregistration"):
        V03Config.model_validate(threshold)


def test_v04_shadow_order_conditional_variants_and_access_fail_closed() -> None:
    shadow = _raw(V04_PATH)
    shadow["shadow"]["required_views"] = list(reversed(shadow["shadow"]["required_views"]))
    with pytest.raises(ValidationError, match="preregistered order"):
        V04Config.model_validate(shadow)

    same_path = _raw(V04_PATH)
    same_path["final_dataset_config_path"] = same_path["development_dataset_config_path"]
    with pytest.raises(ValidationError, match="must be distinct"):
        V04Config.model_validate(same_path)

    automatic = _raw(V04_PATH)
    automatic["final_access"]["automatically_run_final_evaluation"] = True
    with pytest.raises(ValidationError):
        V04Config.model_validate(automatic)

    unconditional = _raw(V04_PATH)
    unconditional["variants"]["longer_context_enabled_only_if"] = "always"
    with pytest.raises(ValidationError):
        V04Config.model_validate(unconditional)

    post_hoc_selection = _raw(V04_PATH)
    post_hoc_selection["variants"]["context_candidate_selection_rule"] = "best_iid_only"
    with pytest.raises(ValidationError):
        V04Config.model_validate(post_hoc_selection)

    untested_main_batch = _raw(V04_PATH)
    untested_main_batch["pilot"]["batch_sizes"] = [1, 2]
    with pytest.raises(ValidationError):
        V04Config.model_validate(untested_main_batch)


def test_pipeline_graph_and_stop_boundary_are_exact() -> None:
    payload: dict[str, Any] = {
        "pipeline_version": "0.4.0",
        "run_name": "phase6-remediation-local",
        "run_root": "runs",
        "v02_config_path": V02_PATH.relative_to(ROOT).as_posix(),
        "v02_config_sha256": "a" * 64,
        "v03_config_path": V03_PATH.relative_to(ROOT).as_posix(),
        "v03_config_sha256": "b" * 64,
        "v04_config_path": V04_PATH.relative_to(ROOT).as_posix(),
        "v04_config_sha256": "c" * 64,
        "stage_order": list(PIPELINE_STAGES),
        "heartbeat_interval_seconds": 60,
        "maximum_status_bytes": 4096,
        "maximum_event_log_bytes": 8192,
        "maximum_pipeline_seconds": 3600,
        "maximum_run_bytes": 1024 * 1024,
        "maximum_process_rss_bytes": 256 * 1024**2,
        "stop_before_final_evaluation": True,
    }
    config = PipelineConfig.model_validate(payload)
    assert config.stage_order == PIPELINE_STAGES
    assert config.stop_before_final_evaluation is True

    reordered = copy.deepcopy(payload)
    reordered["stage_order"][0], reordered["stage_order"][1] = (
        reordered["stage_order"][1],
        reordered["stage_order"][0],
    )
    with pytest.raises(ValidationError, match="preregistered graph"):
        PipelineConfig.model_validate(reordered)

    unsafe = copy.deepcopy(payload)
    unsafe["stop_before_final_evaluation"] = False
    with pytest.raises(ValidationError):
        PipelineConfig.model_validate(unsafe)


def test_config_loaders_reject_nonfiles_symlinks_and_oversize(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="regular non-symlink"):
        load_v02_config(tmp_path / "missing.toml")
    with pytest.raises(ValueError, match="regular non-symlink"):
        load_v02_config(cast(Path, "not-a-path"))

    symlink = tmp_path / "linked.toml"
    symlink.symlink_to(V02_PATH)
    with pytest.raises(ValueError, match="regular non-symlink"):
        load_v02_config(symlink)

    oversized = tmp_path / "oversized.toml"
    oversized.write_bytes(b"x" * (MAX_REMEDIATION_CONFIG_BYTES + 1))
    with pytest.raises(ValueError, match="size bound"):
        load_v02_config(oversized)


def test_config_hash_is_stable_and_requires_a_strict_contract() -> None:
    first = load_v02_config(V02_PATH)
    second = load_v02_config(V02_PATH)
    assert config_sha256(first) == config_sha256(second)
    assert len(config_sha256(first)) == 64
    with pytest.raises(TypeError, match="strict remediation contract"):
        config_sha256(cast(StrictConfigModel, {}))


def test_view_source_vocabulary_excludes_a_final_iid_view() -> None:
    assert not any(view.value == SplitName.IID_TEST.value for view in RemediationView)
    assert tuple(TaskName) == (
        TaskName.CONTINUE_LOG,
        TaskName.FAULT_FAMILY,
        TaskName.EXTRACT_EVIDENCE,
        TaskName.NEXT_ACTION,
        TaskName.INCIDENT_SUMMARY,
        TaskName.COUNTERFACTUAL_COMPARE,
    )
