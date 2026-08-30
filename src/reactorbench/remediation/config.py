"""Strict, versioned configuration for the Phase 6 remediation program."""

from __future__ import annotations

import hashlib
import tomllib
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from reactorbench.model.config import StrictConfigModel, TransformerConfig, _relative_project_path
from reactorbench.schemas.base import ContractId, canonical_json_bytes, canonical_sha256
from reactorbench.schemas.enums import SplitName

MAX_REMEDIATION_CONFIG_BYTES = 256 * 1024
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
PositiveInt = Annotated[StrictInt, Field(gt=0)]
PositiveFloat = Annotated[StrictFloat, Field(gt=0.0, allow_inf_nan=False)]
Probability = Annotated[StrictFloat, Field(ge=0.0, le=1.0, allow_inf_nan=False)]


class RemediationView(StrEnum):
    """Development view identities that cannot be mistaken for final-test splits."""

    IID_TRAIN = "iid_train"
    IID_VALIDATION = "iid_validation"
    SHADOW_RENDERER = "shadow_renderer"
    SHADOW_COMPONENT = "shadow_component"
    SHADOW_SEVERITY = "shadow_severity"
    SHADOW_COMPOSITION = "shadow_composition"
    SHADOW_COUNTERFACTUAL = "shadow_counterfactual"
    SHADOW_NOISE = "shadow_noise"


SHADOW_VIEWS: tuple[RemediationView, ...] = (
    RemediationView.SHADOW_RENDERER,
    RemediationView.SHADOW_COMPONENT,
    RemediationView.SHADOW_SEVERITY,
    RemediationView.SHADOW_COMPOSITION,
    RemediationView.SHADOW_COUNTERFACTUAL,
    RemediationView.SHADOW_NOISE,
)

VIEW_SOURCE_SPLIT: dict[RemediationView, SplitName] = {
    RemediationView.IID_TRAIN: SplitName.IID_TRAIN,
    RemediationView.IID_VALIDATION: SplitName.IID_VALIDATION,
    RemediationView.SHADOW_RENDERER: SplitName.TEMPLATE_TEST,
    RemediationView.SHADOW_COMPONENT: SplitName.COMPONENT_TEST,
    RemediationView.SHADOW_SEVERITY: SplitName.SEVERITY_TEST,
    RemediationView.SHADOW_COMPOSITION: SplitName.COMPOSITION_TEST,
    RemediationView.SHADOW_COUNTERFACTUAL: SplitName.COUNTERFACTUAL_TEST,
    RemediationView.SHADOW_NOISE: SplitName.NOISE_TEST,
}


class RemediationPaths(StrictConfigModel):
    dataset_config_path: str
    tokenizer_path: str
    compact_contract_path: str
    run_root: str

    @field_validator(
        "dataset_config_path", "tokenizer_path", "compact_contract_path", "run_root", mode="after"
    )
    @classmethod
    def paths_are_project_relative(cls, value: str) -> str:
        return _relative_project_path(value)


class InventoryPolicy(StrictConfigModel):
    policy_version: Literal["0.2.0"]
    permitted_views: tuple[Literal[RemediationView.IID_TRAIN, RemediationView.IID_VALIDATION], ...]
    cap_policy: Literal["observed_max_plus_margin"]
    cap_margin_tokens: Annotated[StrictInt, Field(ge=1, le=64)]
    minimum_cap_tokens: Annotated[StrictInt, Field(ge=8, le=256)]
    maximum_cap_tokens: Annotated[StrictInt, Field(ge=32, le=512)]
    context_length: Literal[512]
    maximum_prompt_utf8_bytes: Annotated[StrictInt, Field(ge=1024, le=1024 * 1024)]
    require_target_fit_rate: Probability
    require_round_trip_rate: Probability

    @field_validator("require_target_fit_rate", "require_round_trip_rate", mode="before")
    @classmethod
    def probabilities_are_exact_floats(cls, value: object) -> object:
        if type(value) is not float:
            raise ValueError("inventory probabilities must be exact floats")
        return value

    @field_validator("permitted_views", mode="before")
    @classmethod
    def list_becomes_tuple(cls, value: object) -> object:
        if type(value) is not list:
            raise ValueError("permitted_views must be a TOML array")
        return tuple(RemediationView(item) for item in value)

    @model_validator(mode="after")
    def boundary_is_exact(self) -> InventoryPolicy:
        if self.permitted_views != (
            RemediationView.IID_TRAIN,
            RemediationView.IID_VALIDATION,
        ):
            raise ValueError("v0.2 inventory may contain only IID train and validation")
        if self.minimum_cap_tokens > self.maximum_cap_tokens:
            raise ValueError("minimum generation cap cannot exceed maximum generation cap")
        if self.require_target_fit_rate != 1.0 or self.require_round_trip_rate != 1.0:
            raise ValueError("compact target fit and round-trip requirements must remain 1.0")
        return self


class RemediationTraining(StrictConfigModel):
    seed: Annotated[StrictInt, Field(ge=0, le=4_294_967_295)]
    device: Literal["cpu", "mps"]
    allow_cpu_fallback: StrictBool
    steps: Annotated[StrictInt, Field(ge=1, le=50_000)]
    batch_size: Annotated[StrictInt, Field(ge=1, le=128)]
    learning_rate: PositiveFloat
    weight_decay: Annotated[StrictFloat, Field(ge=0.0, allow_inf_nan=False)]
    gradient_clip_norm: PositiveFloat
    evaluation_interval: Annotated[StrictInt, Field(ge=1, le=5_000)]
    durable_checkpoint_interval: Annotated[StrictInt, Field(ge=1, le=10_000)]

    @field_validator("learning_rate", "weight_decay", "gradient_clip_norm", mode="before")
    @classmethod
    def scalar_hyperparameters_are_exact_floats(cls, value: object) -> object:
        if type(value) is not float:
            raise ValueError("training scalar hyperparameters must be exact floats")
        return value

    @model_validator(mode="after")
    def schedules_are_aligned(self) -> RemediationTraining:
        if self.steps % self.evaluation_interval or self.steps % self.durable_checkpoint_interval:
            raise ValueError("training steps must align with evaluation and checkpoint intervals")
        return self


class DecoderPolicy(StrictConfigModel):
    compact_contract_version: Literal["0.2.0"]
    constrained_strategy: Literal["truth_independent_greedy"]
    unconstrained_strategy: Literal["greedy"]
    report_both_paths: Literal[True]
    confidence: Literal["geometric_mean_selected_token_probability"]
    calibration_bins: Literal[10]
    maximum_decoder_cache_entries: Literal[4096]

    @field_validator("report_both_paths", mode="before")
    @classmethod
    def report_both_paths_is_exact_boolean(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("report_both_paths must be an exact boolean")
        return value


class V02Config(StrictConfigModel):
    iteration_version: Literal["0.2.0"]
    status: Literal["developmental"]
    paths: RemediationPaths
    inventory: InventoryPolicy
    model: TransformerConfig
    training: RemediationTraining
    decoder: DecoderPolicy
    inventory_report_path: str
    inventory_report_checksum_sha256: Sha256

    @field_validator("inventory_report_path", mode="after")
    @classmethod
    def inventory_report_is_project_relative(cls, value: str) -> str:
        return _relative_project_path(value)

    @model_validator(mode="after")
    def control_model_is_frozen(self) -> V02Config:
        expected = {
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
        if self.model.model_dump(mode="python") != expected:
            raise ValueError("v0.2 must preserve the 8x384 context-512 control architecture")
        return self


class AugmentationPolicy(StrictConfigModel):
    policy_version: Literal["0.3.0"]
    train_template_families: tuple[ContractId, ...]
    train_alias_families: tuple[ContractId, ...]
    renderer_variants_per_projection: Annotated[StrictInt, Field(ge=1, le=9)]
    preserve_group_atomicity: Literal[True]
    include_insufficient_evidence_views: Literal[True]
    include_counterfactual_pairs: Literal[True]
    prohibit_target_text_in_prompt: Literal[True]

    @field_validator(
        "preserve_group_atomicity",
        "include_insufficient_evidence_views",
        "include_counterfactual_pairs",
        "prohibit_target_text_in_prompt",
        mode="before",
    )
    @classmethod
    def flags_are_exact_booleans(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("augmentation flags must be exact booleans")
        return value

    @field_validator("train_template_families", "train_alias_families", mode="before")
    @classmethod
    def arrays_become_tuples(cls, value: object) -> object:
        if type(value) is not list:
            raise ValueError("augmentation families must be TOML arrays")
        return tuple(value)

    @model_validator(mode="after")
    def families_are_unique(self) -> AugmentationPolicy:
        for values in (self.train_template_families, self.train_alias_families):
            if not values or len(values) != len(set(values)):
                raise ValueError("augmentation families must be non-empty and unique")
        return self


class CandidatePolicy(StrictConfigModel):
    candidate_id: ContractId
    seed: Annotated[StrictInt, Field(ge=0, le=4_294_967_295)]
    sampling: Literal[
        "uniform_control",
        "task_balanced",
        "task_class_balanced",
        "fault_continuation_focused",
        "hierarchical_task_label_balanced",
        "fault_boosted_hierarchical",
        "task_weighted_hierarchical",
    ]
    exposure: Literal["teacher_forced_only"]
    enabled: Literal[True]

    @field_validator("enabled", mode="before")
    @classmethod
    def enabled_is_exact_boolean(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("candidate enabled flag must be an exact boolean")
        return value


class SemanticSelectionPolicy(StrictConfigModel):
    metric: Literal[
        "frozen_semantic_composite",
        "semantic_floor_then_validation_nll",
        "task_floor_then_validation_nll",
    ]
    tie_break: Literal["lower_validation_nll_then_earlier_step"]
    checkpoint_selection_uses_test: Literal[False]
    minimum_checkpoint_semantic_composite: Probability | None = None
    minimum_checkpoint_fault_macro_f1: Probability | None = None
    minimum_checkpoint_continuation_macro_f1: Probability | None = None
    constrained_schema_validity: Probability
    minimum_fault_margin: Probability
    minimum_action_margin: Probability
    minimum_continuation_macro_f1: Probability
    minimum_evidence_f1: Probability
    minimum_required_abstention_accuracy: Probability
    maximum_no_fault_false_positive_rate: Probability
    maximum_expected_calibration_error: Probability
    selective_risk_coverage: Probability
    maximum_selective_risk: Probability

    @field_validator(
        "minimum_fault_margin",
        "minimum_action_margin",
        "minimum_continuation_macro_f1",
        "minimum_evidence_f1",
        "minimum_required_abstention_accuracy",
        "maximum_no_fault_false_positive_rate",
        "maximum_expected_calibration_error",
        "selective_risk_coverage",
        "maximum_selective_risk",
        "constrained_schema_validity",
        mode="before",
    )
    @classmethod
    def thresholds_are_exact_floats(cls, value: object) -> object:
        if type(value) is not float:
            raise ValueError("semantic thresholds must be exact floats")
        return value

    @field_validator(
        "minimum_checkpoint_semantic_composite",
        "minimum_checkpoint_fault_macro_f1",
        "minimum_checkpoint_continuation_macro_f1",
        mode="before",
    )
    @classmethod
    def checkpoint_floor_is_an_exact_optional_float(cls, value: object) -> object:
        if value is not None and type(value) is not float:
            raise ValueError("checkpoint semantic floor must be an exact float")
        return value

    @model_validator(mode="after")
    def thresholds_are_frozen(self) -> SemanticSelectionPolicy:
        observed = (
            self.constrained_schema_validity,
            self.minimum_fault_margin,
            self.minimum_action_margin,
            self.minimum_continuation_macro_f1,
            self.minimum_evidence_f1,
            self.minimum_required_abstention_accuracy,
            self.maximum_no_fault_false_positive_rate,
            self.maximum_expected_calibration_error,
            self.selective_risk_coverage,
            self.maximum_selective_risk,
        )
        expected = (1.0, 0.02, 0.02, 0.9, 0.7, 0.8, 0.1, 0.15, 0.8, 0.2)
        if observed != expected:
            raise ValueError("v0.3 semantic thresholds differ from the preregistration")
        if self.metric == "frozen_semantic_composite":
            if any(
                value is not None
                for value in (
                    self.minimum_checkpoint_semantic_composite,
                    self.minimum_checkpoint_fault_macro_f1,
                    self.minimum_checkpoint_continuation_macro_f1,
                )
            ):
                raise ValueError("historical semantic selection cannot add a checkpoint floor")
        elif self.metric == "semantic_floor_then_validation_nll":
            if (
                self.minimum_checkpoint_semantic_composite != 0.75
                or self.minimum_checkpoint_fault_macro_f1 is not None
                or self.minimum_checkpoint_continuation_macro_f1 is not None
            ):
                raise ValueError("hierarchical checkpoint semantic floor must remain 0.75")
        elif (
            self.minimum_checkpoint_semantic_composite,
            self.minimum_checkpoint_fault_macro_f1,
            self.minimum_checkpoint_continuation_macro_f1,
        ) != (0.75, 0.9, 0.9):
            raise ValueError("task-aware checkpoint floors differ from targeted-05")
        return self


class CalibrationPolicy(StrictConfigModel):
    """Frozen, validation-only temperature-calibration declaration."""

    policy_version: Literal[
        "0.3.1-targeted",
        "0.3.2-focused",
        "0.3.3-hierarchical",
        "0.3.4-fault-boosted",
        "0.3.5-task-weighted",
    ]
    calibration_example_limit: Literal[56]
    grid_start: StrictFloat
    grid_stop: StrictFloat
    grid_step: StrictFloat
    selection_excludes_semantic_subset: Literal[True]
    calibration_is_validation_only: Literal[True]

    @model_validator(mode="after")
    def grid_is_preregistered(self) -> CalibrationPolicy:
        if (self.grid_start, self.grid_stop, self.grid_step) != (0.5, 5.0, 0.05):
            raise ValueError("calibration temperature grid differs from preregistration")
        return self


class TargetedV03Policy(StrictConfigModel):
    """Narrow opt-in for the non-historical semantic remediation attempt."""

    policy_version: Literal[
        "0.3.1-targeted",
        "0.3.2-focused",
        "0.3.3-hierarchical",
        "0.3.4-fault-boosted",
        "0.3.5-task-weighted",
    ]
    sampling_metadata_required: Literal[True]
    calibration: CalibrationPolicy

    @model_validator(mode="after")
    def versions_match(self) -> TargetedV03Policy:
        if self.calibration.policy_version != self.policy_version:
            raise ValueError("targeted sampling and calibration policy versions differ")
        return self


class V03Config(StrictConfigModel):
    iteration_version: Literal["0.3.0"]
    requires_v02_gate: Literal[True]
    paths: RemediationPaths
    augmentation: AugmentationPolicy
    candidates: tuple[CandidatePolicy, ...]
    training: RemediationTraining
    selection: SemanticSelectionPolicy
    baseline_config_path: str
    baseline_config_sha256: Sha256
    counterfactual_cap_report_path: str
    counterfactual_cap_report_checksum_sha256: Sha256
    semantic_selection_example_limit: Literal[48]
    targeted_policy: TargetedV03Policy | None = None

    @field_validator("baseline_config_path", "counterfactual_cap_report_path", mode="after")
    @classmethod
    def report_paths_are_project_relative(cls, value: str) -> str:
        return _relative_project_path(value)

    @field_validator("requires_v02_gate", mode="before")
    @classmethod
    def prerequisite_is_exact_boolean(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("v0.3 prerequisite flag must be an exact boolean")
        return value

    @field_validator("candidates", mode="before")
    @classmethod
    def candidates_become_tuple(cls, value: object) -> object:
        if type(value) is not list:
            raise ValueError("candidates must be a TOML array of tables")
        return tuple(value)

    @model_validator(mode="after")
    def candidate_matrix_is_bounded(self) -> V03Config:
        if not 1 <= len(self.candidates) <= 3:
            raise ValueError("v0.3 permits one control and at most two variants")
        ids = tuple(item.candidate_id for item in self.candidates)
        if len(ids) != len(set(ids)):
            raise ValueError("v0.3 candidate IDs must be unique")
        historical = ("uniform_control", "task_balanced")
        targeted = ("task_balanced", "task_class_balanced")
        focused = ("fault_continuation_focused",)
        hierarchical = ("hierarchical_task_label_balanced",)
        fault_boosted = ("fault_boosted_hierarchical",)
        task_weighted = ("task_weighted_hierarchical",)
        sampling = tuple(item.sampling for item in self.candidates)
        if self.targeted_policy is None and sampling != historical:
            raise ValueError("historical v0.3 freezes the control and task-balanced candidates")
        if (
            self.targeted_policy is not None
            and self.targeted_policy.policy_version == "0.3.1-targeted"
            and sampling != targeted
        ):
            raise ValueError(
                "targeted v0.3 requires task-balanced and task-class-balanced candidates"
            )
        if (
            self.targeted_policy is not None
            and self.targeted_policy.policy_version == "0.3.2-focused"
            and sampling != focused
        ):
            raise ValueError("focused v0.3 requires exactly one focused candidate")
        if (
            self.targeted_policy is not None
            and self.targeted_policy.policy_version == "0.3.3-hierarchical"
            and sampling != hierarchical
        ):
            raise ValueError("hierarchical v0.3 requires exactly one hierarchical candidate")
        if (
            self.targeted_policy is not None
            and self.targeted_policy.policy_version == "0.3.4-fault-boosted"
            and sampling != fault_boosted
        ):
            raise ValueError("fault-boosted v0.3 requires exactly one fault-boosted candidate")
        if (
            self.targeted_policy is not None
            and self.targeted_policy.policy_version == "0.3.5-task-weighted"
            and sampling != task_weighted
        ):
            raise ValueError("task-weighted v0.3 requires exactly one task-weighted candidate")
        if self.targeted_policy is not None:
            hierarchical_policy = self.targeted_policy.policy_version in {
                "0.3.3-hierarchical",
                "0.3.4-fault-boosted",
                "0.3.5-task-weighted",
            }
            hierarchical_selection = self.selection.metric in {
                "semantic_floor_then_validation_nll",
                "task_floor_then_validation_nll",
            }
            if hierarchical_policy != hierarchical_selection:
                raise ValueError(
                    "hierarchical sampling and checkpoint selection must be enabled together"
                )
            if (self.targeted_policy.policy_version == "0.3.5-task-weighted") != (
                self.selection.metric == "task_floor_then_validation_nll"
            ):
                raise ValueError("task-weighted objective and task-aware selection must match")
        return self


class ShadowPolicy(StrictConfigModel):
    policy_version: Literal["0.4.0"]
    required_views: tuple[RemediationView, ...]
    worst_split_rule: Literal["all_required_views_must_pass"]
    source_groups_must_be_disjoint: Literal[True]
    content_checksums_must_be_disjoint: Literal[True]

    @field_validator(
        "source_groups_must_be_disjoint", "content_checksums_must_be_disjoint", mode="before"
    )
    @classmethod
    def separation_flags_are_exact_booleans(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("shadow separation flags must be exact booleans")
        return value

    @field_validator("required_views", mode="before")
    @classmethod
    def views_become_tuple(cls, value: object) -> object:
        if type(value) is not list:
            raise ValueError("required_views must be a TOML array")
        return tuple(RemediationView(item) for item in value)

    @model_validator(mode="after")
    def views_are_exact(self) -> ShadowPolicy:
        if self.required_views != SHADOW_VIEWS:
            raise ValueError("v0.4 shadow views must match the preregistered order")
        return self


class ConditionalVariantPolicy(StrictConfigModel):
    longer_context_enabled_only_if: Literal["v03_prompt_truncation_is_material"]
    capacity_variant_enabled_only_if: Literal["v03_learning_curves_show_underfitting"]
    default_context_length: Literal[512]
    optional_context_length: Literal[1024]
    maximum_capacity_variants: Literal[1]
    require_new_mps_pilot_before_activation: Literal[True]
    material_prompt_truncation_rate: Probability
    capacity_variant_status: Literal["not_activated_without_measured_underfitting"]
    context_candidate_selection_rule: Literal[
        "all_gates_then_highest_min_view_composite_then_iid_composite_then_shorter_context"
    ]

    @field_validator("material_prompt_truncation_rate", mode="before")
    @classmethod
    def material_rate_is_exact_float(cls, value: object) -> object:
        if type(value) is not float:
            raise ValueError("material prompt-truncation rate must be an exact float")
        return value

    @field_validator("require_new_mps_pilot_before_activation", mode="before")
    @classmethod
    def pilot_flag_is_exact_boolean(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("MPS pilot flag must be an exact boolean")
        return value

    @model_validator(mode="after")
    def materiality_threshold_is_frozen(self) -> ConditionalVariantPolicy:
        if self.material_prompt_truncation_rate != 0.10:
            raise ValueError("v0.4 material prompt-truncation threshold must remain 0.10")
        return self


class MpsPilotPolicy(StrictConfigModel):
    pilot_version: Literal["0.4.0"]
    candidate_id: Literal["v04-context-1024"]
    steps: Literal[10]
    batch_sizes: tuple[Literal[1], Literal[2], Literal[4]]
    require_finite_loss: Literal[True]
    require_checkpoint_reload: Literal[True]

    @field_validator("batch_sizes", mode="before")
    @classmethod
    def batch_sizes_become_tuple(cls, value: object) -> object:
        if type(value) is not list:
            raise ValueError("MPS pilot batch_sizes must be a TOML array")
        return tuple(value)

    @field_validator("require_finite_loss", "require_checkpoint_reload", mode="before")
    @classmethod
    def pilot_flags_are_exact_booleans(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("MPS pilot flags must be exact booleans")
        return value

    @model_validator(mode="after")
    def pilot_matrix_is_exact(self) -> MpsPilotPolicy:
        if self.batch_sizes != (1, 2, 4):
            raise ValueError("v0.4 MPS pilot batch-size matrix must remain [1,2,4]")
        return self


class FinalAccessPolicy(StrictConfigModel):
    automatically_run_final_evaluation: Literal[False]
    require_ready_marker: Literal[True]
    require_owner_review: Literal[True]
    require_explicit_confirm_flag: Literal[True]
    one_access_only: Literal[True]
    historical_golden_packet_permitted: Literal[False]

    @field_validator(
        "automatically_run_final_evaluation",
        "require_ready_marker",
        "require_owner_review",
        "require_explicit_confirm_flag",
        "one_access_only",
        "historical_golden_packet_permitted",
        mode="before",
    )
    @classmethod
    def access_flags_are_exact_booleans(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("final-access flags must be exact booleans")
        return value


class V04Config(StrictConfigModel):
    iteration_version: Literal["0.4.0"]
    requires_v03_gate: Literal[True]
    development_dataset_config_path: str
    final_dataset_config_path: str
    compact_contract_path: str
    run_root: str
    shadow: ShadowPolicy
    variants: ConditionalVariantPolicy
    longer_context_model: TransformerConfig
    training: RemediationTraining
    pilot: MpsPilotPolicy
    final_access: FinalAccessPolicy

    @field_validator("requires_v03_gate", mode="before")
    @classmethod
    def prerequisite_is_exact_boolean(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("v0.4 prerequisite flag must be an exact boolean")
        return value

    @field_validator(
        "development_dataset_config_path",
        "final_dataset_config_path",
        "compact_contract_path",
        "run_root",
        mode="after",
    )
    @classmethod
    def paths_are_project_relative(cls, value: str) -> str:
        return _relative_project_path(value)

    @model_validator(mode="after")
    def dataset_configs_are_distinct(self) -> V04Config:
        if self.development_dataset_config_path == self.final_dataset_config_path:
            raise ValueError("development shadow and final dataset policies must be distinct")
        if self.longer_context_model.model_dump(mode="python") != {
            "model_version": "0.4.0",
            "layers": 8,
            "width": 384,
            "heads": 8,
            "context_length": 1024,
            "feed_forward_multiplier": 4,
            "dropout": 0.1,
            "tie_embeddings": True,
            "bias": True,
        }:
            raise ValueError("v0.4 longer-context candidate differs from its frozen architecture")
        return self


PIPELINE_STAGES: tuple[str, ...] = (
    "preflight",
    "v02_inventory_and_caps",
    "v02_smoke",
    "v02_development_training",
    "v02_development_gate",
    "v03_data_audit",
    "v03_smoke",
    "v03_candidate_training",
    "v03_development_evaluation",
    "v03_gate",
    "v04_shadow_freeze",
    "v04_pilot",
    "v04_candidate_training",
    "v04_shadow_evaluation",
    "v04_gate_and_final_policy_freeze",
    "review_bundle",
)


class V02PrefixReusePolicy(StrictConfigModel):
    """Checksum-pinned read-only prefix source for the targeted pipeline only."""

    policy_version: Literal["0.3.1-targeted"]
    source_run_root: str
    source_run_manifest_sha256: Sha256
    source_commit: Annotated[str, Field(pattern=r"^[0-9a-f]{7,64}$")]
    evidence_inventory_sha256: Sha256
    evidence: tuple[tuple[StrictStr, Sha256], ...] = Field(min_length=21, max_length=21)
    verify_only_stages: tuple[
        Literal[
            "v02_inventory_and_caps",
            "v02_smoke",
            "v02_development_training",
            "v02_development_gate",
        ],
        Literal[
            "v02_inventory_and_caps",
            "v02_smoke",
            "v02_development_training",
            "v02_development_gate",
        ],
        Literal[
            "v02_inventory_and_caps",
            "v02_smoke",
            "v02_development_training",
            "v02_development_gate",
        ],
        Literal[
            "v02_inventory_and_caps",
            "v02_smoke",
            "v02_development_training",
            "v02_development_gate",
        ],
    ]

    @field_validator("source_run_root", mode="after")
    @classmethod
    def source_is_project_relative(cls, value: str) -> str:
        return _relative_project_path(value)

    @field_validator("verify_only_stages", mode="before")
    @classmethod
    def stages_become_tuple(cls, value: object) -> object:
        return tuple(value) if type(value) is list else value

    @field_validator("evidence", mode="before")
    @classmethod
    def evidence_becomes_exact_tuples(cls, value: object) -> object:
        if type(value) is not list:
            return value
        return tuple(tuple(item) if type(item) is list else item for item in value)

    @model_validator(mode="after")
    def stages_are_exact(self) -> V02PrefixReusePolicy:
        if self.verify_only_stages != PIPELINE_STAGES[1:5]:
            raise ValueError("v0.2 reuse must verify exactly the four compute-heavy prefix stages")
        paths = tuple(path for path, _checksum in self.evidence)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("v0.2 reuse evidence must contain 21 unique paths in canonical order")
        if canonical_sha256(self.evidence) != self.evidence_inventory_sha256:
            raise ValueError("v0.2 reuse evidence inventory checksum mismatch")
        return self


class PipelineConfig(StrictConfigModel):
    pipeline_version: Literal["0.4.0"]
    run_name: ContractId
    run_root: str
    v02_config_path: str
    v02_config_sha256: Sha256
    v03_config_path: str
    v03_config_sha256: Sha256
    v04_config_path: str
    v04_config_sha256: Sha256
    stage_order: tuple[str, ...]
    heartbeat_interval_seconds: Annotated[StrictInt, Field(ge=5, le=60)]
    maximum_status_bytes: Annotated[StrictInt, Field(ge=1024, le=4 * 1024 * 1024)]
    maximum_event_log_bytes: Annotated[StrictInt, Field(ge=1024, le=256 * 1024 * 1024)]
    maximum_pipeline_seconds: Annotated[StrictInt, Field(ge=3600, le=7 * 24 * 3600)]
    maximum_run_bytes: Annotated[StrictInt, Field(ge=1024 * 1024, le=32 * 1024**3)]
    maximum_process_rss_bytes: Annotated[StrictInt, Field(ge=256 * 1024**2, le=128 * 1024**3)]
    stop_before_final_evaluation: Literal[True]
    reuse_v02_prefix: V02PrefixReusePolicy | None = None

    @field_validator("stop_before_final_evaluation", mode="before")
    @classmethod
    def final_stop_is_exact_boolean(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("final-evaluation stop flag must be an exact boolean")
        return value

    @field_validator(
        "run_root", "v02_config_path", "v03_config_path", "v04_config_path", mode="after"
    )
    @classmethod
    def paths_are_project_relative(cls, value: str) -> str:
        return _relative_project_path(value)

    @field_validator("stage_order", mode="before")
    @classmethod
    def stage_list_becomes_tuple(cls, value: object) -> object:
        if type(value) is not list or any(type(item) is not str for item in value):
            raise ValueError("stage_order must be a TOML string array")
        return tuple(value)

    @model_validator(mode="after")
    def stage_graph_is_exact(self) -> PipelineConfig:
        if self.stage_order != PIPELINE_STAGES:
            raise ValueError("pipeline stage order differs from the preregistered graph")
        return self


def _load_config(path: Path, model: type[StrictConfigModel]) -> StrictConfigModel:
    if not isinstance(path, Path) or path.is_symlink() or not path.is_file():
        raise ValueError("remediation config must be a regular non-symlink file")
    if not 0 < path.stat().st_size <= MAX_REMEDIATION_CONFIG_BYTES:
        raise ValueError("remediation config exceeds its size bound")
    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    return model.model_validate(raw)


def load_v02_config(path: Path) -> V02Config:
    return V02Config.model_validate(_load_config(path, V02Config))


def load_v03_config(path: Path) -> V03Config:
    return V03Config.model_validate(_load_config(path, V03Config))


def load_v04_config(path: Path) -> V04Config:
    return V04Config.model_validate(_load_config(path, V04Config))


def load_pipeline_config(path: Path) -> PipelineConfig:
    return PipelineConfig.model_validate(_load_config(path, PipelineConfig))


def config_sha256(config: StrictConfigModel) -> str:
    if not isinstance(config, StrictConfigModel):
        raise TypeError("config must use the strict remediation contract")
    return hashlib.sha256(
        canonical_json_bytes(config.model_dump(mode="json", round_trip=True, exclude_none=True))
    ).hexdigest()


__all__ = [
    "PIPELINE_STAGES",
    "SHADOW_VIEWS",
    "VIEW_SOURCE_SPLIT",
    "AugmentationPolicy",
    "CalibrationPolicy",
    "CandidatePolicy",
    "DecoderPolicy",
    "FinalAccessPolicy",
    "InventoryPolicy",
    "MpsPilotPolicy",
    "PipelineConfig",
    "RemediationTraining",
    "RemediationView",
    "SemanticSelectionPolicy",
    "ShadowPolicy",
    "TargetedV03Policy",
    "V02Config",
    "V02PrefixReusePolicy",
    "V03Config",
    "V04Config",
    "config_sha256",
    "load_pipeline_config",
    "load_v02_config",
    "load_v03_config",
    "load_v04_config",
]
