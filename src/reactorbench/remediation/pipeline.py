"""Concrete development-only execution layer for the Phase 6 remediation graph.

The orchestration module owns crash recovery and immutable stage commits.  This
module supplies the scientific actions for that frozen graph, verifies every
development input before use, and deliberately provides no final-evaluation action.
All dataset construction is explicitly scoped to IID development or preregistered
shadow views; the future fresh extension remains a separate owner-reviewed surface.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import resource
import shutil
import subprocess
import time
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Literal, Never, cast

import torch
from pydantic import Field, StrictBool, StrictFloat, StrictInt, StrictStr, model_validator

from reactorbench.dataset.config import (
    DevelopmentDatasetConfig,
    load_development_dataset_config,
)
from reactorbench.evaluation.compact import compact_output_contract
from reactorbench.evaluation.config import BaselineConfig, load_phase5_config
from reactorbench.model import TransformerConfig, TransformerLM
from reactorbench.model.checkpoint import CheckpointManifest, load_checkpoint
from reactorbench.schemas.base import (
    ContractId,
    ContractModel,
    canonical_json_bytes,
    canonical_sha256,
)
from reactorbench.schemas.enums import TaskName
from reactorbench.tokenizer import ProjectTokenizer

from .acceptance import (
    DevelopmentArtifactBinding,
    DevelopmentView,
    V03AcceptanceResult,
    V04AcceptanceResult,
    evaluate_v03_acceptance,
    evaluate_v04_acceptance,
)
from .audit import audit_safe_development_dataset
from .baselines import RemediationBaselineReport, run_remediation_baselines
from .calibration import (
    CalibrationObservation,
    TemperatureCalibrationReport,
    apply_temperature,
    fit_temperature,
)
from .config import (
    PIPELINE_STAGES,
    SHADOW_VIEWS,
    PipelineConfig,
    RemediationTraining,
    RemediationView,
    SemanticSelectionPolicy,
    V02Config,
    V03Config,
    V04Config,
    config_sha256,
    load_v02_config,
    load_v03_config,
    load_v04_config,
)
from .data import (
    FrozenV03IIDMaterial,
    RemediationExample,
    SafeDevelopmentDataset,
    SafeDevelopmentManifest,
    TaskScopedStructuredFingerprint,
    build_frozen_v03_iid_material,
    build_safe_development_dataset,
    build_safe_development_dataset_with_structured_fingerprints,
    load_safe_development_artifact,
    write_safe_development_artifact,
)
from .decoding import (
    DualPathCompactPrediction,
    decode_compact_examples,
)
from .inventory import (
    CompactInventoryReport,
    CounterfactualCapExtensionReport,
    measure_compact_inventory,
    measure_counterfactual_cap_extension,
)
from .metrics import (
    SemanticEvaluationReport,
    canonical_prediction_jsonl_bytes,
    evaluate_semantic_predictions,
    prediction_artifact_byte_sha256,
    semantic_composite_score,
)
from .orchestration import (
    ArtifactReference,
    PipelineEngine,
    PipelineState,
    PipelineStore,
    RunManifest,
    StageAction,
    StageCompletionMarker,
    StageContext,
    StageMetric,
    StageOutcome,
    StageStatus,
)
from .progress import PROGRESS_EVENT_LOG_FILENAME, ProgressMetric, ProgressSnapshot
from .sampling import (
    SamplingMetadataRecord,
    sampling_metadata_inventory_sha256,
    task_balanced_batch_indices,
)
from .selection import (
    CalibrationSelectionManifest,
    SemanticSelectionManifest,
    build_calibration_selection_manifest,
    build_semantic_selection_manifest,
    resolve_calibration_selection_examples,
    resolve_semantic_selection_examples,
)
from .serialization import CompactTokenizedExample, tokenize_compact_example
from .training import (
    TARGETED_SAMPLING_BINDING_FILENAME,
    CompactTrainingOutcome,
    CompactTrainingResult,
    CompactTrainingStopped,
    DeviceResolution,
    EvaluationCallback,
    TargetedSamplingBinding,
    TrainingProgress,
    bind_targeted_sampling,
    durable_training_state_upper_bound_bytes,
    ensure_targeted_sampling_binding,
    latest_committed_training_state,
    load_targeted_sampling_binding,
    retire_superseded_training_states,
    selected_checkpoint_upper_bound_bytes,
    tokenized_inventory_sha256,
    train_compact_model,
)

PIPELINE_EXECUTION_VERSION: Literal["0.4.0"] = "0.4.0"
STOP_REQUEST_FILENAME = "STOP_REQUESTED"
STOP_ARCHIVE_DIRECTORY = "stop-requests"
FINAL_EVALUATION_READY_FILENAME = "FINAL_EVALUATION_READY.json"
OWNER_REVIEW_APPROVED_FILENAME = "OWNER_REVIEW_APPROVED.json"
FRESH_EXTENSION_MANIFEST_FILENAME = "FRESH_EXTENSION_MANIFEST.json"
MAX_PIPELINE_JSON_BYTES = 64 * 1024 * 1024
MAX_PREDICTION_ROW_BYTES = 256 * 1024
MAX_RUN_FILES = 200_000
MAX_TRANSIENT_DURABLE_STATES = 3
COOPERATIVE_DECODE_CHUNK_SIZE: Literal[1] = 1
DECODE_PROGRESS_REPORT_INTERVAL: Literal[16] = 16
SMOKE_STEPS = 2
SMOKE_EXAMPLES_PER_VIEW = 24
V02_MAXIMUM_CAP_EXHAUSTION_RATE = 0.01
V01_PROMPT_TRUNCATION_COUNT: Literal[689] = 689
V01_PROMPT_TRUNCATION_EXAMPLE_COUNT: Literal[882] = 882
V02_FROZEN_PROMPT_TRUNCATION_COUNT: Literal[668] = 668
FINAL_ACCESS_LEDGER_FILENAME = "FINAL_EVALUATION_ACCESS.json"
FINAL_RESULT_FILENAME = "FINAL_EVALUATION_RESULT.json"
FINAL_REVIEW_FILENAME = "FINAL_EVALUATION_REVIEW.md"
TERMINAL_REVIEW_DIRECTORY = "terminal-reviews"
TERMINAL_REVIEW_MANIFEST_FILENAME = "terminal-review-bundle.json"
TERMINAL_REVIEW_SUMMARY_FILENAME = "TERMINAL_REVIEW.md"


def _semantic_checkpoint_selection_score(
    policy: SemanticSelectionPolicy,
    *,
    composite: float,
) -> float:
    """Convert semantic evidence to the checkpoint selector's minimized score."""

    if type(policy) is not SemanticSelectionPolicy:
        raise TypeError("checkpoint selection requires the exact semantic policy")
    if type(composite) is not float or not math.isfinite(composite) or not 0.0 <= composite <= 1.0:
        raise ValueError("semantic checkpoint composite must be a finite probability")
    if policy.metric == "frozen_semantic_composite":
        return 1.0 - composite
    floor = policy.minimum_checkpoint_semantic_composite
    if floor is None:
        raise PipelineExecutionError("semantic-floor checkpoint selection lacks its frozen floor")
    return 0.0 if composite >= floor else 1.0 + (floor - composite)


def _select_v03_candidate(
    policy: SemanticSelectionPolicy,
    candidates: tuple[CandidateScore, ...],
) -> CandidateScore:
    """Reconstruct the frozen candidate decision under the configured policy."""

    if type(policy) is not SemanticSelectionPolicy:
        raise TypeError("candidate selection requires the exact semantic policy")
    if not candidates:
        raise PipelineExecutionError("v0.3 candidate selection inventory is empty")
    if policy.metric == "semantic_floor_then_validation_nll" and len(candidates) != 1:
        raise PipelineExecutionError(
            "hierarchical v0.3 selection requires its single frozen candidate"
        )
    if policy.metric == "frozen_semantic_composite":
        return min(
            candidates,
            key=lambda item: (
                -item.semantic_composite,
                item.selected_validation_nll,
                item.selected_step,
                item.candidate_id,
            ),
        )
    return min(
        candidates,
        key=lambda item: (
            _semantic_checkpoint_selection_score(policy, composite=item.semantic_composite),
            item.selected_validation_nll,
            item.selected_step,
            item.candidate_id,
        ),
    )


TRUSTED_GIT = "/usr/bin/git"

_DEVELOPMENT_VIEW_BY_REMEDIATION: Mapping[RemediationView, DevelopmentView] = MappingProxyType(
    {
        RemediationView.IID_VALIDATION: DevelopmentView.IID_VALIDATION,
        RemediationView.SHADOW_RENDERER: DevelopmentView.RENDERER_SHADOW,
        RemediationView.SHADOW_COMPONENT: DevelopmentView.COMPONENT_ROLE_SHADOW,
        RemediationView.SHADOW_SEVERITY: DevelopmentView.SEVERITY_SHADOW,
        RemediationView.SHADOW_COMPOSITION: DevelopmentView.COMPOSITION_SHADOW,
        RemediationView.SHADOW_COUNTERFACTUAL: DevelopmentView.COUNTERFACTUAL_SHADOW,
        RemediationView.SHADOW_NOISE: DevelopmentView.NOISE_SHADOW,
    }
)

Sha256 = Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{64}$")]
GitCommit = Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{7,64}$")]
Probability = Annotated[StrictFloat, Field(ge=0.0, le=1.0, allow_inf_nan=False)]


class PipelineExecutionError(RuntimeError):
    """Safe public error for an invalid scientific execution boundary."""


class PipelineResourceLimitError(PipelineExecutionError):
    """Raised before a configured resource boundary can be exceeded further."""


class FinalEvaluationBlockedError(PipelineExecutionError):
    """Raised when the separate future final-access prerequisites are incomplete."""


class PipelineStopRequest(ContractModel):
    request_version: Literal["0.4.0"] = PIPELINE_EXECUTION_VERSION
    requested_at: str
    process_id: StrictInt = Field(ge=1)
    checksum_sha256: Sha256

    @model_validator(mode="after")
    def timestamp_and_checksum_match(self) -> PipelineStopRequest:
        _canonical_utc(self.requested_at)
        expected = canonical_sha256(
            self.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        if self.checksum_sha256 != expected:
            raise ValueError("pipeline stop request checksum mismatch")
        return self


class ExecutionPreflightReport(ContractModel):
    report_version: Literal["0.4.0"] = PIPELINE_EXECUTION_VERSION
    runner_source_commit: GitCommit
    runner_worktree_clean: Literal[True]
    pipeline_config_sha256: Sha256
    v02_config_sha256: Sha256
    v03_config_sha256: Sha256
    v04_config_sha256: Sha256
    frozen_data_source_commit: GitCommit
    tokenizer_manifest_sha256: Sha256
    compact_contract_sha256: Sha256
    v02_inventory_report_sha256: Sha256
    v03_counterfactual_cap_report_sha256: Sha256
    final_evaluation_automatic: Literal[False] = False
    development_only: Literal[True] = True
    checksum_sha256: Sha256

    @model_validator(mode="after")
    def checksum_matches(self) -> ExecutionPreflightReport:
        expected = canonical_sha256(
            self.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        if self.checksum_sha256 != expected:
            raise ValueError("execution preflight report checksum mismatch")
        return self


class V02PrefixReuseReport(ContractModel):
    """New-run reference proving one v0.2 stage was independently reused, not run."""

    report_version: Literal["0.3.1-targeted"] = "0.3.1-targeted"
    stage_name: Literal[
        "v02_inventory_and_caps",
        "v02_smoke",
        "v02_development_training",
        "v02_development_gate",
    ]
    source_run_manifest_sha256: Sha256
    source_commit: GitCommit
    evidence: tuple[tuple[StrictStr, Sha256], ...] = Field(min_length=21, max_length=21)
    checksum_sha256: Sha256

    @model_validator(mode="after")
    def checksum_matches(self) -> V02PrefixReuseReport:
        if tuple(path for path, _hash in self.evidence) != tuple(
            sorted(path for path, _hash in self.evidence)
        ):
            raise ValueError("prefix reuse evidence must use canonical path order")
        expected = canonical_sha256(
            self.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        if self.checksum_sha256 != expected:
            raise ValueError("prefix reuse report checksum mismatch")
        return self


class PredictionArtifactManifest(ContractModel):
    artifact_version: Literal["0.4.0"] = PIPELINE_EXECUTION_VERSION
    view: RemediationView
    example_count: StrictInt = Field(ge=1, le=1_000_000)
    example_inventory_sha256: Sha256
    prediction_inventory_sha256: Sha256
    predictions_sha256: Sha256
    predictions_size_bytes: StrictInt = Field(ge=1, le=4 * 1024**3)
    checksum_sha256: Sha256

    @model_validator(mode="after")
    def checksum_matches(self) -> PredictionArtifactManifest:
        expected = canonical_sha256(
            self.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        if self.checksum_sha256 != expected:
            raise ValueError("prediction artifact manifest checksum mismatch")
        return self


class V02DevelopmentGateReport(ContractModel):
    report_version: Literal["0.2.0"] = "0.2.0"
    inventory_report_sha256: Sha256
    prediction_manifest_sha256: Sha256
    training_result_sha256: Sha256
    checkpoint_manifest_sha256: Sha256
    checkpoint_weights_sha256: Sha256
    example_count: StrictInt = Field(ge=1)
    constrained_parse_rate: Probability
    constrained_schema_validity_rate: Probability
    constrained_exact_semantic_match_rate: Probability
    constrained_mean_latency_seconds: StrictFloat = Field(ge=0.0, allow_inf_nan=False)
    unconstrained_parse_rate: Probability
    unconstrained_schema_validity_rate: Probability
    unconstrained_exact_semantic_match_rate: Probability
    unconstrained_mean_latency_seconds: StrictFloat = Field(ge=0.0, allow_inf_nan=False)
    generation_cap_exhaustion_rate: Probability
    process_peak_rss_bytes: StrictInt = Field(ge=1)
    mps_peak_current_allocated_bytes: StrictInt = Field(ge=0)
    mps_peak_driver_allocated_bytes: StrictInt = Field(ge=0)
    checkpoint_size_bytes: StrictInt = Field(ge=1)
    inventory_example_count: StrictInt = Field(ge=1)
    prompt_truncation_count: StrictInt = Field(ge=0)
    prompt_truncation_rate: Probability
    target_fit_rate: Probability
    round_trip_rate: Probability
    reachability_rate: Probability
    task_footer_retained_rate: Probability
    cap_exhaustion_target_rate: Probability
    v01_prompt_truncation_count: Literal[689] = V01_PROMPT_TRUNCATION_COUNT
    v01_example_count: Literal[882] = V01_PROMPT_TRUNCATION_EXAMPLE_COUNT
    frozen_prompt_truncation_count: Literal[668] = V02_FROZEN_PROMPT_TRUNCATION_COUNT
    prompt_truncation_materially_lower: Literal[False] = False
    v04_context_pilot_required: Literal[True] = True
    maximum_generation_cap_exhaustion_rate: StrictFloat = Field(
        default=V02_MAXIMUM_CAP_EXHAUSTION_RATE,
        ge=V02_MAXIMUM_CAP_EXHAUSTION_RATE,
        le=V02_MAXIMUM_CAP_EXHAUSTION_RATE,
        allow_inf_nan=False,
    )
    advancement_allowed: StrictBool
    checksum_sha256: Sha256

    @model_validator(mode="after")
    def gate_and_checksum_match(self) -> V02DevelopmentGateReport:
        expected_gate = (
            self.constrained_parse_rate == 1.0
            and self.constrained_schema_validity_rate == 1.0
            and self.generation_cap_exhaustion_rate <= self.maximum_generation_cap_exhaustion_rate
            and self.prompt_truncation_count == self.frozen_prompt_truncation_count
            and self.prompt_truncation_rate
            == self.prompt_truncation_count / self.inventory_example_count
            and self.target_fit_rate == 1.0
            and self.round_trip_rate == 1.0
            and self.reachability_rate == 1.0
            and self.task_footer_retained_rate == 1.0
            and self.cap_exhaustion_target_rate == 0.0
        )
        if self.advancement_allowed is not expected_gate:
            raise ValueError("v0.2 advancement differs from its structural gate")
        expected = canonical_sha256(
            self.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        if self.checksum_sha256 != expected:
            raise ValueError("v0.2 development gate checksum mismatch")
        return self


class StructuredFingerprintViewInventory(ContractModel):
    view: RemediationView
    example_count: StrictInt = Field(ge=1)
    distinct_task_scoped_fingerprint_count: StrictInt = Field(ge=1)
    inventory_sha256: Sha256

    @model_validator(mode="after")
    def counts_are_possible(self) -> StructuredFingerprintViewInventory:
        if self.distinct_task_scoped_fingerprint_count > self.example_count:
            raise ValueError("distinct structured fingerprints exceed the view inventory")
        return self


class StructuredFingerprintViewOverlap(ContractModel):
    first_view: RemediationView
    second_view: RemediationView
    overlap_count: StrictInt = Field(ge=0)


class TaskScopedStructuredSeparationReport(ContractModel):
    report_version: Literal["0.4.0"] = PIPELINE_EXECUTION_VERSION
    views: tuple[RemediationView, ...] = Field(min_length=2, max_length=len(RemediationView))
    inventories: tuple[StructuredFingerprintViewInventory, ...] = Field(min_length=2)
    pairwise_overlaps: tuple[StructuredFingerprintViewOverlap, ...] = Field(min_length=1)
    overlap_count: StrictInt = Field(ge=0)
    passed: StrictBool
    checksum_sha256: Sha256

    @model_validator(mode="after")
    def inventory_pairing_and_checksum_match(self) -> TaskScopedStructuredSeparationReport:
        canonical_views = tuple(view for view in RemediationView if view in set(self.views))
        if self.views != canonical_views:
            raise ValueError("structured separation views must be unique and canonical")
        if tuple(item.view for item in self.inventories) != self.views:
            raise ValueError("structured fingerprint inventories differ from declared views")
        expected_pairs = tuple(
            (first, second)
            for first_index, first in enumerate(self.views)
            for second in self.views[first_index + 1 :]
        )
        if (
            tuple((item.first_view, item.second_view) for item in self.pairwise_overlaps)
            != expected_pairs
        ):
            raise ValueError("structured overlap pairs are incomplete or noncanonical")
        expected_overlap = sum(item.overlap_count for item in self.pairwise_overlaps)
        if self.overlap_count != expected_overlap or self.passed is not (expected_overlap == 0):
            raise ValueError("structured separation result differs from pairwise findings")
        expected = canonical_sha256(
            self.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        if self.checksum_sha256 != expected:
            raise ValueError("structured separation checksum mismatch")
        return self


class V03RemovedDuplicateBinding(ContractModel):
    """One removed raw row and the exact retained row that supersedes it."""

    binding_version: Literal["0.3.0"] = "0.3.0"
    removed_example_id: ContractId
    removed_example_sha256: Sha256
    task_name: TaskName
    view: RemediationView
    prompt_sha256: Sha256
    canonical_target_sha256: Sha256
    retained_example_id: ContractId
    retained_example_sha256: Sha256
    retained_task_name: TaskName
    retained_view: RemediationView
    retained_prompt_sha256: Sha256
    retained_canonical_target_sha256: Sha256
    checksum_sha256: Sha256

    @model_validator(mode="after")
    def exact_duplicate_and_checksum_match(self) -> V03RemovedDuplicateBinding:
        if (
            self.view is not RemediationView.IID_TRAIN
            or self.retained_view is not RemediationView.IID_TRAIN
            or self.removed_example_id == self.retained_example_id
            or self.removed_example_sha256 == self.retained_example_sha256
            or self.task_name is TaskName.COUNTERFACTUAL_COMPARE
            or self.retained_task_name is not self.task_name
            or self.retained_prompt_sha256 != self.prompt_sha256
            or self.retained_canonical_target_sha256 != self.canonical_target_sha256
        ):
            raise ValueError("v0.3 removed-row binding is not an IID duplicate pair")
        expected = canonical_sha256(
            self.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        if self.checksum_sha256 != expected:
            raise ValueError("v0.3 removed-row binding checksum mismatch")
        return self


class V03CounterfactualCapCompatibilityReport(ContractModel):
    """Exact bridge from the historical raw cap material to deduplicated IID data."""

    report_version: Literal["0.3.0"] = "0.3.0"
    frozen_cap_report_sha256: Sha256
    raw_cap_report_sha256: Sha256
    deduplicated_cap_report_sha256: Sha256
    frozen_cap_dataset_manifest_sha256: Sha256
    raw_dataset_manifest_sha256: Sha256
    deduplicated_dataset_manifest_sha256: Sha256
    raw_example_inventory_sha256: Sha256
    deduplicated_example_inventory_sha256: Sha256
    removed_example_inventory_sha256: Sha256
    removed_examples: tuple[V03RemovedDuplicateBinding, ...] = Field(
        min_length=24,
        max_length=24,
    )
    raw_counterfactual_inventory_sha256: Sha256
    deduplicated_counterfactual_inventory_sha256: Sha256
    raw_counterfactual_evidence_sha256: Sha256
    deduplicated_counterfactual_evidence_sha256: Sha256
    raw_example_count: Literal[5859]
    deduplicated_example_count: Literal[5835]
    removed_example_count: Literal[24]
    counterfactual_train_count: Literal[40]
    counterfactual_validation_count: Literal[15]
    retained_rows_bit_exact: Literal[True]
    removed_rows_verified: Literal[True]
    frozen_cap_reproduced: Literal[True]
    passed: StrictBool
    checksum_sha256: Sha256

    @model_validator(mode="after")
    def bridge_and_checksum_match(self) -> V03CounterfactualCapCompatibilityReport:
        removed_ids = tuple(item.removed_example_id for item in self.removed_examples)
        if (
            removed_ids != tuple(sorted(removed_ids))
            or len(removed_ids) != len(set(removed_ids))
            or len(self.removed_examples) != self.removed_example_count
            or self.removed_example_inventory_sha256
            != canonical_sha256(
                tuple(
                    item.model_dump(mode="json", round_trip=True) for item in self.removed_examples
                )
            )
        ):
            raise ValueError("v0.3 removed-row inventory is incomplete or noncanonical")
        expected_pass = (
            self.frozen_cap_report_sha256 == self.raw_cap_report_sha256
            and self.frozen_cap_dataset_manifest_sha256 == self.raw_dataset_manifest_sha256
            and self.raw_example_count - self.deduplicated_example_count
            == self.removed_example_count
            and self.raw_counterfactual_inventory_sha256
            == self.deduplicated_counterfactual_inventory_sha256
            and self.raw_counterfactual_evidence_sha256
            == self.deduplicated_counterfactual_evidence_sha256
            and self.retained_rows_bit_exact
            and self.removed_rows_verified
            and self.frozen_cap_reproduced
        )
        if self.passed is not expected_pass:
            raise ValueError("v0.3 cap compatibility state differs from its exact bridge")
        expected = canonical_sha256(
            self.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        if self.checksum_sha256 != expected:
            raise ValueError("v0.3 cap compatibility checksum mismatch")
        return self


class DevelopmentSeparationReport(ContractModel):
    report_version: Literal["0.4.0"] = PIPELINE_EXECUTION_VERSION
    iid_dataset_manifest_sha256: Sha256
    shadow_dataset_manifest_sha256: Sha256
    iid_example_count: StrictInt = Field(ge=1)
    shadow_example_count: StrictInt = Field(ge=1)
    group_overlap_count: StrictInt = Field(ge=0)
    example_checksum_overlap_count: StrictInt = Field(ge=0)
    prompt_checksum_overlap_count: StrictInt = Field(ge=0)
    structured_separation: TaskScopedStructuredSeparationReport
    passed: StrictBool
    checksum_sha256: Sha256

    @model_validator(mode="after")
    def result_and_checksum_match(self) -> DevelopmentSeparationReport:
        if self.structured_separation.views != tuple(RemediationView):
            raise ValueError("development separation must cover every development view")
        expected_pass = (
            self.group_overlap_count
            + self.example_checksum_overlap_count
            + self.prompt_checksum_overlap_count
            == 0
            and self.structured_separation.passed
        )
        if self.passed is not expected_pass:
            raise ValueError("development separation pass state differs from findings")
        expected = canonical_sha256(
            self.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        if self.checksum_sha256 != expected:
            raise ValueError("development separation checksum mismatch")
        return self


class CandidateScore(ContractModel):
    candidate_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,95}$")
    checkpoint_manifest_sha256: Sha256
    semantic_composite: Probability
    selected_validation_nll: StrictFloat = Field(ge=0.0, allow_inf_nan=False)
    selected_step: StrictInt = Field(ge=0, le=50_000)
    evaluation_report_sha256: Sha256


class CandidateSelectionReport(ContractModel):
    report_version: Literal["0.3.0"] = "0.3.0"
    selection_manifest_sha256: Sha256
    candidates: tuple[CandidateScore, ...] = Field(min_length=1, max_length=3)
    selected_candidate_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,95}$")
    selected_checkpoint_manifest_sha256: Sha256
    checksum_sha256: Sha256

    @model_validator(mode="after")
    def selection_and_checksum_match(self) -> CandidateSelectionReport:
        ids = tuple(item.candidate_id for item in self.candidates)
        if len(ids) != len(set(ids)) or ids != tuple(sorted(ids)):
            raise ValueError("candidate selection inventory must be unique and sorted")
        selected = min(
            self.candidates,
            key=lambda item: (
                -item.semantic_composite,
                item.selected_validation_nll,
                item.selected_step,
                item.candidate_id,
            ),
        )
        if (
            self.selected_candidate_id != selected.candidate_id
            or self.selected_checkpoint_manifest_sha256 != selected.checkpoint_manifest_sha256
        ):
            raise ValueError("selected candidate differs from the frozen ranking")
        expected = canonical_sha256(
            self.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        if self.checksum_sha256 != expected:
            raise ValueError("candidate selection report checksum mismatch")
        return self


class TargetedV03GateBinding(ContractModel):
    """Bind calibration, raw evidence, calibrated evidence, and unchanged thresholds."""

    report_version: Literal["0.3.1-targeted"] = "0.3.1-targeted"
    candidate_selection_sha256: Sha256
    calibration_selection_sha256: Sha256
    temperature_calibration_sha256: Sha256
    raw_evaluation_sha256: Sha256
    calibrated_evaluation_sha256: Sha256
    acceptance_sha256: Sha256
    raw_prediction_artifact_sha256: Sha256
    calibrated_prediction_artifact_sha256: Sha256
    outputs_bit_exact: Literal[True]
    thresholds_unchanged: Literal[True]
    checksum_sha256: Sha256

    @model_validator(mode="after")
    def invariance_and_checksum_match(self) -> TargetedV03GateBinding:
        if self.raw_prediction_artifact_sha256 != self.calibrated_prediction_artifact_sha256:
            raise ValueError("calibration must not alter the prediction artifact")
        expected = canonical_sha256(
            self.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        if self.checksum_sha256 != expected:
            raise ValueError("targeted v0.3 gate binding checksum mismatch")
        return self


class V03GateReplayCertification(ContractModel):
    """Immutable certificate for a no-training reconstruction of a preserved gate."""

    report_version: Literal["0.4.1-gate-replay"] = "0.4.1-gate-replay"
    source_run_name: Literal["phase6-remediation-v0.4.0-targeted-03"]
    replay_name: Literal["phase6-remediation-v0.4.0-targeted-03-gate-replay-01"]
    source_run_manifest_sha256: Sha256
    source_pipeline_state_file_sha256: Sha256
    source_pipeline_state_contract_sha256: Sha256
    source_commit: GitCommit
    replay_source_commit: GitCommit
    pipeline_config_sha256: Sha256
    training_completion_marker_sha256: Sha256
    evaluation_completion_marker_sha256: Sha256
    acceptance_sha256: Sha256
    targeted_gate_binding_sha256: Sha256
    passed_check_count: StrictInt = Field(ge=0, le=10)
    total_check_count: Literal[10]
    advancement_allowed: StrictBool
    thresholds_unchanged: Literal[True]
    retraining_performed: Literal[False]
    final_evaluation_accessed: Literal[False]
    checksum_sha256: Sha256

    @model_validator(mode="after")
    def certification_matches(self) -> V03GateReplayCertification:
        if self.advancement_allowed is not (self.passed_check_count == self.total_check_count):
            raise ValueError("gate replay advancement differs from reconstructed checks")
        expected = canonical_sha256(
            self.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        if self.checksum_sha256 != expected:
            raise ValueError("gate replay certification checksum mismatch")
        return self


class V04PilotMeasurement(ContractModel):
    batch_size: StrictInt = Field(ge=1, le=128)
    training_result_sha256: Sha256
    training_config_sha256: Sha256
    model_config_sha256: Sha256
    train_tokenized_sha256: Sha256
    validation_tokenized_sha256: Sha256
    tokenizer_manifest_sha256: Sha256
    checkpoint_manifest_sha256: Sha256
    device: DeviceResolution
    train_example_count: StrictInt = Field(ge=1)
    validation_example_count: StrictInt = Field(ge=1)
    pilot_train_example_count: StrictInt = Field(ge=1)
    pilot_validation_example_count: StrictInt = Field(ge=1)
    train_length_inventory_sha256: Sha256
    validation_length_inventory_sha256: Sha256
    maximum_train_sequence_tokens: StrictInt = Field(ge=1, le=1024)
    maximum_validation_sequence_tokens: StrictInt = Field(ge=1, le=1024)
    mean_train_sequence_tokens: StrictFloat = Field(gt=0.0, le=1024.0, allow_inf_nan=False)
    mean_validation_sequence_tokens: StrictFloat = Field(gt=0.0, le=1024.0, allow_inf_nan=False)
    maximum_train_sequence_exercised: Literal[True]
    finite_loss: Literal[True]
    checkpoint_reloaded: Literal[True]
    elapsed_seconds: StrictFloat = Field(gt=0.0, allow_inf_nan=False)
    process_peak_rss_bytes: StrictInt = Field(ge=1)

    @model_validator(mode="after")
    def length_evidence_is_possible(self) -> V04PilotMeasurement:
        if (
            self.mean_train_sequence_tokens > self.maximum_train_sequence_tokens
            or self.mean_validation_sequence_tokens > self.maximum_validation_sequence_tokens
        ):
            raise ValueError("v0.4 pilot mean sequence length exceeds its maximum")
        return self


class V04PilotReport(ContractModel):
    report_version: Literal["0.4.0"] = PIPELINE_EXECUTION_VERSION
    candidate_id: str
    requested_device: Literal["mps"]
    required_resolved_device: Literal["mps"]
    mandatory_batch_resolved_device: Literal["cpu", "mps"] | None
    frozen_v02_prompt_truncation_count: Literal[668] = V02_FROZEN_PROMPT_TRUNCATION_COUNT
    frozen_v02_example_count: Literal[882] = V01_PROMPT_TRUNCATION_EXAMPLE_COUNT
    prompt_truncation_rate: Probability
    v03_train_prompt_truncation_rate: Probability
    material_truncation_threshold: Probability
    activated: StrictBool
    measurements: tuple[V04PilotMeasurement, ...] = Field(max_length=3)
    passed: StrictBool
    checksum_sha256: Sha256

    @model_validator(mode="after")
    def activation_and_checksum_match(self) -> V04PilotReport:
        frozen_rate = self.frozen_v02_prompt_truncation_count / self.frozen_v02_example_count
        if self.prompt_truncation_rate != frozen_rate:
            raise ValueError("v0.4 pilot does not reproduce the D-073 truncation rate")
        expected_activation = (
            frozen_rate >= self.material_truncation_threshold
            and self.v03_train_prompt_truncation_rate >= self.material_truncation_threshold
        )
        if self.activated is not expected_activation:
            raise ValueError("v0.4 pilot activation differs from its two measured conditions")
        if self.activated:
            if tuple(item.batch_size for item in self.measurements) != (1, 2, 4):
                raise ValueError("activated v0.4 pilot must cover batches 1, 2, and 4")
            if any(item.device.requested != self.requested_device for item in self.measurements):
                raise ValueError("activated v0.4 pilot requested-device evidence differs")
            profile = self.measurements[0]
            profile_fields = (
                "train_example_count",
                "validation_example_count",
                "pilot_train_example_count",
                "pilot_validation_example_count",
                "model_config_sha256",
                "train_tokenized_sha256",
                "validation_tokenized_sha256",
                "tokenizer_manifest_sha256",
                "train_length_inventory_sha256",
                "validation_length_inventory_sha256",
                "maximum_train_sequence_tokens",
                "maximum_validation_sequence_tokens",
                "mean_train_sequence_tokens",
                "mean_validation_sequence_tokens",
            )
            if any(
                getattr(item, field) != getattr(profile, field)
                for item in self.measurements[1:]
                for field in profile_fields
            ):
                raise ValueError("v0.4 pilot measurements use different length profiles")
            mandatory_measurement = self.measurements[-1]
            if self.mandatory_batch_resolved_device != mandatory_measurement.device.resolved:
                raise ValueError("v0.4 pilot mandatory resolved-device evidence differs")
            expected_pass = (
                all(
                    item.finite_loss
                    and item.checkpoint_reloaded
                    and item.device.resolved == self.required_resolved_device
                    and not item.device.fallback_used
                    and item.maximum_train_sequence_exercised
                    for item in self.measurements
                )
                and mandatory_measurement.batch_size == 4
                and mandatory_measurement.device.resolved == self.required_resolved_device
                and not mandatory_measurement.device.fallback_used
            )
            if self.passed is not expected_pass:
                raise ValueError("activated v0.4 pilot pass state differs from native-MPS evidence")
        elif (
            self.measurements or self.mandatory_batch_resolved_device is not None or not self.passed
        ):
            raise ValueError("non-activated v0.4 pilot must be a passing no-op")
        expected = canonical_sha256(
            self.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        if self.checksum_sha256 != expected:
            raise ValueError("v0.4 pilot report checksum mismatch")
        return self


class V04CandidateTrainingReport(ContractModel):
    report_version: Literal["0.4.0"] = PIPELINE_EXECUTION_VERSION
    activated: StrictBool
    candidate_id: StrictStr = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,95}$")
    reused_v03_candidate: StrictBool
    source_stage: Literal["v03_candidate_training", "v04_candidate_training"]
    training_result_sha256: Sha256
    checkpoint_manifest_sha256: Sha256
    checksum_sha256: Sha256

    @model_validator(mode="after")
    def source_and_checksum_match(self) -> V04CandidateTrainingReport:
        expected_reuse = not self.activated
        if self.reused_v03_candidate is not expected_reuse or self.source_stage != (
            "v03_candidate_training" if expected_reuse else "v04_candidate_training"
        ):
            raise ValueError("v0.4 candidate source differs from pilot activation")
        expected = canonical_sha256(
            self.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        if self.checksum_sha256 != expected:
            raise ValueError("v0.4 candidate-training report checksum mismatch")
        return self


class V04CandidateEvaluation(ContractModel):
    candidate_id: StrictStr = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,95}$")
    context_length: StrictInt = Field(ge=512, le=1024)
    checkpoint_manifest_sha256: Sha256
    iid_report_sha256: Sha256
    iid_acceptance_sha256: Sha256
    shadow_reports: tuple[tuple[RemediationView, Sha256], ...]
    v04_acceptance_sha256: Sha256
    all_required_gates_passed: StrictBool
    worst_view_semantic_composite: Probability
    iid_semantic_composite: Probability

    @model_validator(mode="after")
    def inventory_is_complete(self) -> V04CandidateEvaluation:
        if tuple(view for view, _ in self.shadow_reports) != SHADOW_VIEWS:
            raise ValueError("v0.4 candidate evaluation lacks the required shadow order")
        return self


class V04EvaluationIndex(ContractModel):
    report_version: Literal["0.4.0"] = PIPELINE_EXECUTION_VERSION
    candidates: tuple[V04CandidateEvaluation, ...] = Field(min_length=1, max_length=2)
    selected_candidate_id: StrictStr = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,95}$")
    checkpoint_manifest_sha256: Sha256
    iid_report_sha256: Sha256
    shadow_reports: tuple[tuple[RemediationView, Sha256], ...]
    v04_acceptance_sha256: Sha256
    checksum_sha256: Sha256

    @model_validator(mode="after")
    def inventory_and_checksum_match(self) -> V04EvaluationIndex:
        if tuple(view for view, _ in self.shadow_reports) != SHADOW_VIEWS:
            raise ValueError("v0.4 evaluation index lacks the required shadow view order")
        candidate_ids = tuple(item.candidate_id for item in self.candidates)
        if candidate_ids != tuple(sorted(candidate_ids)) or len(candidate_ids) != len(
            set(candidate_ids)
        ):
            raise ValueError("v0.4 candidate evaluation inventory must be unique and sorted")
        selected = _select_v04_candidate(self.candidates)
        if (
            self.selected_candidate_id != selected.candidate_id
            or self.checkpoint_manifest_sha256 != selected.checkpoint_manifest_sha256
            or self.iid_report_sha256 != selected.iid_report_sha256
            or self.shadow_reports != selected.shadow_reports
            or self.v04_acceptance_sha256 != selected.v04_acceptance_sha256
        ):
            raise ValueError("v0.4 selected evidence differs from the frozen ranking")
        expected = canonical_sha256(
            self.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        if self.checksum_sha256 != expected:
            raise ValueError("v0.4 evaluation index checksum mismatch")
        return self


def _select_v04_candidate(
    candidates: tuple[V04CandidateEvaluation, ...],
) -> V04CandidateEvaluation:
    if type(candidates) is not tuple or not 1 <= len(candidates) <= 2:
        raise ValueError("v0.4 selection requires one control and at most one variant")
    if any(type(candidate) is not V04CandidateEvaluation for candidate in candidates):
        raise TypeError("v0.4 selection requires exact candidate-evaluation contracts")
    return min(
        candidates,
        key=lambda item: (
            not item.all_required_gates_passed,
            -item.worst_view_semantic_composite,
            -item.iid_semantic_composite,
            item.context_length,
            item.candidate_id,
        ),
    )


class FinalEvaluationPolicyFreeze(ContractModel):
    policy_version: Literal["0.4.0"] = PIPELINE_EXECUTION_VERSION
    v04_acceptance_sha256: Sha256
    development_gate_passed: StrictBool
    automatic_final_evaluation: Literal[False] = False
    requires_fresh_extension_manifest: Literal[True] = True
    requires_owner_review_record: Literal[True] = True
    requires_explicit_confirmation: Literal[True] = True
    one_access_only: Literal[True] = True
    historical_extension_permitted: Literal[False] = False
    status: Literal[
        "locked_pending_owner_reviewed_fresh_extension",
        "locked_development_gate_failed",
    ]
    checksum_sha256: Sha256

    @model_validator(mode="after")
    def status_and_checksum_match(self) -> FinalEvaluationPolicyFreeze:
        expected_status = (
            "locked_pending_owner_reviewed_fresh_extension"
            if self.development_gate_passed
            else "locked_development_gate_failed"
        )
        if self.status != expected_status:
            raise ValueError("final evaluation policy status differs from the development gate")
        expected = canonical_sha256(
            self.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        if self.checksum_sha256 != expected:
            raise ValueError("final evaluation policy checksum mismatch")
        return self


class ReviewStageBinding(ContractModel):
    stage: str
    outcome: ArtifactReference


class ReviewBundleManifest(ContractModel):
    bundle_version: Literal["0.4.0"] = PIPELINE_EXECUTION_VERSION
    run_name: str
    source_commit: GitCommit
    pipeline_config_sha256: Sha256
    stages: tuple[ReviewStageBinding, ...] = Field(min_length=len(PIPELINE_STAGES) - 1)
    final_policy_sha256: Sha256
    final_evaluation_status: Literal["locked_pending_owner_reviewed_fresh_extension"] = (
        "locked_pending_owner_reviewed_fresh_extension"
    )
    contains_final_payload: Literal[False] = False
    contains_historical_extension_payload: Literal[False] = False
    summary_relative_path: str
    summary_sha256: Sha256
    checksum_sha256: Sha256

    @model_validator(mode="after")
    def graph_and_checksum_match(self) -> ReviewBundleManifest:
        if tuple(item.stage for item in self.stages) != PIPELINE_STAGES[:-1]:
            raise ValueError("review bundle stage inventory differs from the frozen graph")
        expected = canonical_sha256(
            self.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        if self.checksum_sha256 != expected:
            raise ValueError("review bundle checksum mismatch")
        return self


class TerminalReviewStage(ContractModel):
    stage: str
    status: Literal["completed", "blocked", "failed", "stopped"]
    latest_attempt_path: str
    outcome: ArtifactReference | None

    @model_validator(mode="after")
    def terminal_shape_matches(self) -> TerminalReviewStage:
        path = Path(self.latest_attempt_path)
        if (
            not self.latest_attempt_path
            or "\\" in self.latest_attempt_path
            or path.is_absolute()
            or ".." in path.parts
            or path.as_posix() != self.latest_attempt_path
        ):
            raise ValueError("terminal review attempt path is unsafe")
        if self.status in {"completed", "blocked"} and self.outcome is None:
            raise ValueError("published terminal stage lacks its immutable outcome")
        if self.status in {"failed", "stopped"} and self.outcome is not None:
            raise ValueError("unsuccessful terminal stage cannot publish an outcome")
        return self


class TerminalReviewBundleManifest(ContractModel):
    bundle_version: Literal["0.4.0"] = PIPELINE_EXECUTION_VERSION
    bundle_kind: Literal["terminal_prefix"] = "terminal_prefix"
    run_name: str
    source_commit: GitCommit
    pipeline_config_sha256: Sha256
    pipeline_state_sha256: Sha256
    pipeline_status: Literal["completed", "blocked", "failed", "stopped"]
    stages: tuple[TerminalReviewStage, ...] = Field(max_length=len(PIPELINE_STAGES))
    completed_prefix_length: StrictInt = Field(ge=0, le=len(PIPELINE_STAGES))
    final_evaluation_accessed: Literal[False] = False
    summary_relative_path: str
    summary_sha256: Sha256
    checksum_sha256: Sha256

    @model_validator(mode="after")
    def prefix_and_checksum_match(self) -> TerminalReviewBundleManifest:
        observed = tuple(item.stage for item in self.stages)
        if observed != PIPELINE_STAGES[: len(observed)]:
            raise ValueError("terminal review stages are not a contiguous graph prefix")
        completed_prefix = tuple(
            item.status for item in self.stages[: self.completed_prefix_length]
        )
        if completed_prefix != ("completed",) * self.completed_prefix_length:
            raise ValueError("terminal review completed-prefix count mismatch")
        suffix = self.stages[self.completed_prefix_length :]
        if self.pipeline_status == "completed":
            if self.completed_prefix_length != len(PIPELINE_STAGES) or suffix:
                raise ValueError("completed terminal review does not cover the full graph")
        elif self.pipeline_status in {"blocked", "failed"}:
            if len(suffix) != 1 or suffix[0].status != self.pipeline_status:
                raise ValueError("terminal review status differs from its terminal stage")
        elif suffix:
            if len(suffix) != 1 or suffix[0].status != "stopped":
                raise ValueError("attempted stop has an invalid terminal stage")
        elif self.completed_prefix_length >= len(PIPELINE_STAGES):
            raise ValueError("pre-stage stop must leave an incomplete graph")
        expected = canonical_sha256(
            self.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        if self.checksum_sha256 != expected:
            raise ValueError("terminal review bundle checksum mismatch")
        return self


@dataclass(frozen=True, slots=True)
class ReviewBundleOutput:
    manifest_path: Path
    summary_path: Path
    manifest: ReviewBundleManifest | TerminalReviewBundleManifest


class FreshFinalExtensionManifest(ContractModel):
    manifest_version: Literal["future-1.0.0"]
    extension_id: StrictStr = Field(
        min_length=1,
        max_length=96,
        pattern=r"^[A-Za-z][A-Za-z0-9._:-]*$",
    )
    created_at: str
    generated_after_policy_sha256: Sha256
    final_dataset_config_sha256: Sha256
    frozen_final_payload_relative_path: str
    frozen_final_payload_sha256: Sha256
    fresh_extension_payload_relative_path: str
    fresh_extension_payload_sha256: Sha256
    case_ids: tuple[StrictStr, ...] = Field(min_length=1, max_length=256)
    historical_case_ids: tuple[StrictStr, ...] = ()
    checksum_sha256: Sha256

    @model_validator(mode="after")
    def freshness_and_checksum_match(self) -> FreshFinalExtensionManifest:
        _canonical_utc(self.created_at)
        for path_text in (
            self.frozen_final_payload_relative_path,
            self.fresh_extension_payload_relative_path,
        ):
            path = Path(path_text)
            if (
                path.is_absolute()
                or ".." in path.parts
                or path.as_posix() != path_text
                or not path_text
            ):
                raise ValueError("fresh-extension payload path escapes its run")
        if (
            len(self.case_ids) != len(set(self.case_ids))
            or tuple(sorted(self.case_ids)) != self.case_ids
            or self.historical_case_ids
            or any(
                re.fullmatch(r"G(?:0[1-9]|1[0-5])", item, re.IGNORECASE) for item in self.case_ids
            )
        ):
            raise ValueError("fresh extension contains historical G01-G15 identity")
        expected = canonical_sha256(
            self.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        if self.checksum_sha256 != expected:
            raise ValueError("fresh final-extension manifest checksum mismatch")
        return self


class FreshExtensionReview(ContractModel):
    review_version: Literal["future-1.0.0"]
    fresh_extension_manifest_sha256: Sha256
    owner_review_record_sha256: Sha256
    owner_approved: Literal[True]
    generated_after_development_freeze: Literal[True]
    historical_payload_used: Literal[False]
    checksum_sha256: Sha256

    @model_validator(mode="after")
    def checksum_matches(self) -> FreshExtensionReview:
        expected = canonical_sha256(
            self.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        if self.checksum_sha256 != expected:
            raise ValueError("fresh-extension review checksum mismatch")
        return self


class FinalAccessLedger(ContractModel):
    ledger_version: Literal["future-1.0.0"]
    status: Literal["claimed", "completed", "failed"]
    source_commit: GitCommit
    authorization_sha256: Sha256
    claimed_at: str
    completed_at: str | None
    result_sha256: Sha256 | None
    failure_code: StrictStr | None = Field(default=None, max_length=128)
    checksum_sha256: Sha256

    @model_validator(mode="after")
    def lifecycle_and_checksum_match(self) -> FinalAccessLedger:
        _canonical_utc(self.claimed_at)
        if self.completed_at is not None:
            _canonical_utc(self.completed_at)
        if self.status == "claimed" and any(
            value is not None
            for value in (self.completed_at, self.result_sha256, self.failure_code)
        ):
            raise ValueError("claimed final-access ledger contains terminal fields")
        if self.status == "completed" and (
            self.completed_at is None or self.result_sha256 is None or self.failure_code is not None
        ):
            raise ValueError("completed final-access ledger shape is invalid")
        if self.status == "failed" and (
            self.completed_at is None or self.result_sha256 is not None or self.failure_code is None
        ):
            raise ValueError("failed final-access ledger shape is invalid")
        expected = canonical_sha256(
            self.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        if self.checksum_sha256 != expected:
            raise ValueError("final-access ledger checksum mismatch")
        return self


class FinalEvaluationResult(ContractModel):
    result_version: Literal["future-1.0.0"]
    authorization_sha256: Sha256
    source_commit: GitCommit
    final_dataset_config_sha256: Sha256
    frozen_final_payload_sha256: Sha256
    fresh_extension_payload_sha256: Sha256
    selected_checkpoint_sha256: Sha256
    final_acceptance_sha256: Sha256
    final_acceptance_passed: StrictBool
    completed_at: str
    review_summary_relative_path: str
    review_summary_sha256: Sha256
    checksum_sha256: Sha256

    @model_validator(mode="after")
    def checksum_matches(self) -> FinalEvaluationResult:
        _canonical_utc(self.completed_at)
        expected = canonical_sha256(
            self.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        if self.checksum_sha256 != expected:
            raise ValueError("final-evaluation result checksum mismatch")
        return self


class FinalEvaluationRequest(ContractModel):
    request_version: Literal["future-1.0.0"]
    policy_sha256: Sha256
    review_bundle_sha256: Sha256
    fresh_extension_review_sha256: Sha256
    explicit_confirmation: Literal[True]
    one_access_nonce_sha256: Sha256
    checksum_sha256: Sha256

    @model_validator(mode="after")
    def checksum_matches(self) -> FinalEvaluationRequest:
        expected = canonical_sha256(
            self.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        if self.checksum_sha256 != expected:
            raise ValueError("final evaluation request checksum mismatch")
        return self


class FinalEvaluationAuthorization(ContractModel):
    authorization_version: Literal["future-1.0.0"]
    policy_sha256: Sha256
    review_bundle_sha256: Sha256
    fresh_extension_manifest_sha256: Sha256
    owner_review_record_sha256: Sha256
    one_access_nonce_sha256: Sha256
    checksum_sha256: Sha256

    @model_validator(mode="after")
    def checksum_matches(self) -> FinalEvaluationAuthorization:
        expected = canonical_sha256(
            self.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        if self.checksum_sha256 != expected:
            raise ValueError("final evaluation authorization checksum mismatch")
        return self


def _canonical_utc(value: str) -> datetime:
    if type(value) is not str:
        raise ValueError("pipeline timestamp must be a string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError("pipeline timestamp is invalid") from error
    offset = parsed.utcoffset()
    if parsed.tzinfo is None or offset is None or offset.total_seconds() != 0:
        raise ValueError("pipeline timestamp must use UTC")
    if value != parsed.astimezone(UTC).isoformat(timespec="seconds"):
        raise ValueError("pipeline timestamp must use canonical second precision")
    return parsed


def _utc_now() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")


def _bound_model[ModelT: ContractModel](draft: ModelT, model_type: type[ModelT]) -> ModelT:
    payload = draft.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
    payload["checksum_sha256"] = canonical_sha256(payload)
    return model_type.model_validate_json(canonical_json_bytes(payload), strict=True)


def _task_scoped_structured_separation(
    records: tuple[TaskScopedStructuredFingerprint, ...],
    *,
    views: tuple[RemediationView, ...],
) -> TaskScopedStructuredSeparationReport:
    """Checksum-bind every view and reject truth-independent structure reuse."""

    canonical_views = tuple(view for view in RemediationView if view in set(views))
    if (
        type(records) is not tuple
        or any(type(item) is not TaskScopedStructuredFingerprint for item in records)
        or type(views) is not tuple
        or views != canonical_views
        or len(views) < 2
        or any(item.view not in set(views) for item in records)
    ):
        raise TypeError("structured separation requires canonical in-memory inventories")
    records_by_view = {view: tuple(item for item in records if item.view is view) for view in views}
    if any(not items for items in records_by_view.values()):
        raise ValueError("every separated development view must have fingerprint records")
    inventories = tuple(
        StructuredFingerprintViewInventory(
            view=view,
            example_count=len(records_by_view[view]),
            distinct_task_scoped_fingerprint_count=len(
                {item.separation_key for item in records_by_view[view]}
            ),
            inventory_sha256=canonical_sha256(
                tuple(
                    sorted(
                        (
                            item.example_id,
                            item.task_name.value,
                            item.structured_fingerprint_sha256,
                        )
                        for item in records_by_view[view]
                    )
                )
            ),
        )
        for view in views
    )
    keys_by_view = {view: {item.separation_key for item in records_by_view[view]} for view in views}
    overlaps = tuple(
        StructuredFingerprintViewOverlap(
            first_view=first,
            second_view=second,
            overlap_count=len(keys_by_view[first] & keys_by_view[second]),
        )
        for first_index, first in enumerate(views)
        for second in views[first_index + 1 :]
    )
    overlap_count = sum(item.overlap_count for item in overlaps)
    draft = TaskScopedStructuredSeparationReport.model_construct(
        views=views,
        inventories=inventories,
        pairwise_overlaps=overlaps,
        overlap_count=overlap_count,
        passed=overlap_count == 0,
        checksum_sha256="0" * 64,
    )
    return _bound_model(draft, TaskScopedStructuredSeparationReport)


def _counterfactual_inventory_sha256(dataset: SafeDevelopmentDataset) -> str:
    return canonical_sha256(
        tuple(
            (item.example_id, item.checksum_sha256, item.view.value)
            for item in dataset.examples
            if item.task_name is TaskName.COUNTERFACTUAL_COMPARE
        )
    )


def _counterfactual_cap_evidence_sha256(report: CounterfactualCapExtensionReport) -> str:
    if type(report) is not CounterfactualCapExtensionReport:
        raise TypeError("counterfactual cap evidence requires an exact report")
    return canonical_sha256(
        report.model_dump(
            mode="json",
            round_trip=True,
            exclude={"dataset_manifest_sha256", "checksum_sha256"},
        )
    )


def _v03_cap_compatibility_report(
    material: FrozenV03IIDMaterial,
    *,
    frozen_cap: CounterfactualCapExtensionReport,
    raw_cap: CounterfactualCapExtensionReport,
    deduplicated_cap: CounterfactualCapExtensionReport,
) -> V03CounterfactualCapCompatibilityReport:
    """Bind exact frozen-cap reproduction to the row-safe deduplication proof."""

    if type(material) is not FrozenV03IIDMaterial or any(
        type(report) is not CounterfactualCapExtensionReport
        for report in (frozen_cap, raw_cap, deduplicated_cap)
    ):
        raise TypeError("v0.3 cap bridge requires exact material and cap reports")
    if raw_cap != frozen_cap:
        raise PipelineExecutionError("v0.3 raw cap material did not reproduce its frozen report")
    raw_counterfactual = _counterfactual_inventory_sha256(material.raw_dataset)
    deduplicated_counterfactual = _counterfactual_inventory_sha256(material.dataset)
    raw_evidence = _counterfactual_cap_evidence_sha256(raw_cap)
    deduplicated_evidence = _counterfactual_cap_evidence_sha256(deduplicated_cap)
    retained_by_signature: dict[
        tuple[TaskName, str, str],
        list[RemediationExample],
    ] = {}
    for retained in material.dataset.examples:
        retained_by_signature.setdefault(
            (retained.task_name, retained.prompt_sha256, retained.canonical_target_json),
            [],
        ).append(retained)
    removed_bindings: list[V03RemovedDuplicateBinding] = []
    for removed in sorted(material.removed_examples, key=lambda item: item.example_id):
        counterparts = retained_by_signature.get(
            (removed.task_name, removed.prompt_sha256, removed.canonical_target_json),
            [],
        )
        if len(counterparts) != 1:
            raise PipelineExecutionError(
                "v0.3 removed row does not bind exactly one retained counterpart"
            )
        retained = counterparts[0]
        binding_draft = V03RemovedDuplicateBinding.model_construct(
            removed_example_id=removed.example_id,
            removed_example_sha256=removed.checksum_sha256,
            task_name=removed.task_name,
            view=removed.view,
            prompt_sha256=removed.prompt_sha256,
            canonical_target_sha256=canonical_sha256(removed.canonical_target_json),
            retained_example_id=retained.example_id,
            retained_example_sha256=retained.checksum_sha256,
            retained_task_name=retained.task_name,
            retained_view=retained.view,
            retained_prompt_sha256=retained.prompt_sha256,
            retained_canonical_target_sha256=canonical_sha256(retained.canonical_target_json),
            checksum_sha256="0" * 64,
        )
        removed_bindings.append(_bound_model(binding_draft, V03RemovedDuplicateBinding))
    removed_examples = tuple(removed_bindings)
    removed_inventory = canonical_sha256(
        tuple(item.model_dump(mode="json", round_trip=True) for item in removed_examples)
    )
    draft = V03CounterfactualCapCompatibilityReport.model_construct(
        frozen_cap_report_sha256=frozen_cap.checksum_sha256,
        raw_cap_report_sha256=raw_cap.checksum_sha256,
        deduplicated_cap_report_sha256=deduplicated_cap.checksum_sha256,
        frozen_cap_dataset_manifest_sha256=frozen_cap.dataset_manifest_sha256,
        raw_dataset_manifest_sha256=material.raw_dataset.manifest.checksum_sha256,
        deduplicated_dataset_manifest_sha256=material.dataset.manifest.checksum_sha256,
        raw_example_inventory_sha256=material.raw_dataset.manifest.inventory_sha256,
        deduplicated_example_inventory_sha256=material.dataset.manifest.inventory_sha256,
        removed_example_inventory_sha256=removed_inventory,
        removed_examples=removed_examples,
        raw_counterfactual_inventory_sha256=raw_counterfactual,
        deduplicated_counterfactual_inventory_sha256=deduplicated_counterfactual,
        raw_counterfactual_evidence_sha256=raw_evidence,
        deduplicated_counterfactual_evidence_sha256=deduplicated_evidence,
        raw_example_count=len(material.raw_dataset.examples),
        deduplicated_example_count=len(material.dataset.examples),
        removed_example_count=len(material.removed_examples),
        counterfactual_train_count=sum(
            item.task_name is TaskName.COUNTERFACTUAL_COMPARE
            and item.view is RemediationView.IID_TRAIN
            for item in material.dataset.examples
        ),
        counterfactual_validation_count=sum(
            item.task_name is TaskName.COUNTERFACTUAL_COMPARE
            and item.view is RemediationView.IID_VALIDATION
            for item in material.dataset.examples
        ),
        retained_rows_bit_exact=True,
        removed_rows_verified=True,
        frozen_cap_reproduced=True,
        passed=(
            raw_counterfactual == deduplicated_counterfactual
            and raw_evidence == deduplicated_evidence
        ),
        checksum_sha256="0" * 64,
    )
    return _bound_model(draft, V03CounterfactualCapCompatibilityReport)


def _development_separation_report(
    iid: SafeDevelopmentDataset,
    shadow: SafeDevelopmentDataset,
    structured_separation: TaskScopedStructuredSeparationReport,
) -> DevelopmentSeparationReport:
    """Bind legacy rendered checks and the stronger cross-view structured gate."""

    iid_groups = {item.group_id for item in iid.examples}
    shadow_groups = {item.group_id for item in shadow.examples}
    iid_checksums = {item.checksum_sha256 for item in iid.examples}
    shadow_checksums = {item.checksum_sha256 for item in shadow.examples}
    iid_prompts = {item.prompt_sha256 for item in iid.examples}
    shadow_prompts = {item.prompt_sha256 for item in shadow.examples}
    draft = DevelopmentSeparationReport.model_construct(
        iid_dataset_manifest_sha256=iid.manifest.checksum_sha256,
        shadow_dataset_manifest_sha256=shadow.manifest.checksum_sha256,
        iid_example_count=len(iid.examples),
        shadow_example_count=len(shadow.examples),
        group_overlap_count=len(iid_groups & shadow_groups),
        example_checksum_overlap_count=len(iid_checksums & shadow_checksums),
        prompt_checksum_overlap_count=len(iid_prompts & shadow_prompts),
        structured_separation=structured_separation,
        passed=not (
            iid_groups & shadow_groups
            or iid_checksums & shadow_checksums
            or iid_prompts & shadow_prompts
        )
        and structured_separation.passed,
        checksum_sha256="0" * 64,
    )
    return _bound_model(draft, DevelopmentSeparationReport)


def _require_exact_regenerated_iid(
    regenerated: SafeDevelopmentDataset,
    committed: SafeDevelopmentDataset,
) -> None:
    """Fail closed unless v0.4 reproduced the exact committed v0.3 material."""

    if regenerated != committed:
        raise PipelineExecutionError(
            "regenerated IID material differs from the committed v0.3 stage artifact"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json(payload: bytes) -> object:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise ValueError("pipeline execution JSON contains a duplicate key")
            result[key] = value
        return result

    return json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"pipeline execution JSON contains non-finite data: {value}")
        ),
    )


def _write_bytes(path: Path, payload: bytes) -> None:
    if (
        not isinstance(path, Path)
        or path.parent.is_symlink()
        or not path.parent.is_dir()
        or path.exists()
        or path.is_symlink()
    ):
        raise FileExistsError("pipeline execution output must be a new regular file")
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        parent_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        if path.exists() and not path.is_symlink():
            path.unlink()
        raise


def _write_contract(path: Path, model: ContractModel) -> None:
    payload = canonical_json_bytes(model.model_dump(mode="json", round_trip=True)) + b"\n"
    if len(payload) > MAX_PIPELINE_JSON_BYTES:
        raise ValueError("pipeline execution contract exceeds its byte bound")
    _write_bytes(path, payload)


def _read_contract[ModelT: ContractModel](
    path: Path,
    model_type: type[ModelT],
    *,
    maximum_bytes: int = MAX_PIPELINE_JSON_BYTES,
) -> ModelT:
    if (
        not isinstance(path, Path)
        or path.is_symlink()
        or not path.is_file()
        or not 0 < path.stat().st_size <= maximum_bytes
    ):
        raise ValueError("pipeline execution contract is missing, unsafe, or oversized")
    payload = path.read_bytes()
    _strict_json(payload)
    model = model_type.model_validate_json(payload, strict=True)
    canonical = canonical_json_bytes(model.model_dump(mode="json", round_trip=True)) + b"\n"
    if payload != canonical:
        raise ValueError("pipeline execution contract is not canonical JSON")
    return model


def _artifact_reference(path: Path, *, run_directory: Path) -> ArtifactReference:
    root = run_directory.resolve(strict=True)
    if path.is_symlink() or not path.is_file():
        raise ValueError("pipeline execution artifact must be a regular file")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError("pipeline execution artifact escapes the run directory")
    cursor = run_directory
    for part in path.relative_to(run_directory).parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError("pipeline execution artifact traverses a symlink")
    return ArtifactReference(
        relative_path=resolved.relative_to(root).as_posix(),
        sha256=_sha256(path),
        size_bytes=path.stat().st_size,
    )


def _safe_input_path(
    project_root: Path,
    relative: str,
    *,
    kind: Literal["file", "directory"],
) -> Path:
    if type(relative) is not str or not relative or "\\" in relative:
        raise ValueError("pipeline input path must be a non-empty POSIX relative path")
    candidate_relative = Path(relative)
    if candidate_relative.is_absolute() or ".." in candidate_relative.parts:
        raise ValueError("pipeline input path escapes the project")
    prohibited = ("golden", "heldout", "iid_test", "final")
    if any(any(token in part.lower() for token in prohibited) for part in candidate_relative.parts):
        raise ValueError("development pipeline input path crosses a prohibited boundary")
    candidate = project_root / candidate_relative
    cursor = project_root
    for part in candidate_relative.parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError("pipeline input path traverses a symlink")
    if (kind == "file" and not candidate.is_file()) or (
        kind == "directory" and not candidate.is_dir()
    ):
        raise ValueError("required pipeline input is missing")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(project_root.resolve(strict=True)):
        raise ValueError("pipeline input resolves outside the project")
    return resolved


def _verify_compact_contract(contract_path: Path) -> str:
    """Verify the committed compiler snapshot and its complete sibling manifest."""

    expected_contract = compact_output_contract()
    if (
        expected_contract.get("contract_version") != "0.2.0"
        or expected_contract.get("frozen") is not True
    ):
        raise PipelineExecutionError("runtime compact-output contract is incompatible")
    expected_bytes = canonical_json_bytes(expected_contract) + b"\n"
    if contract_path.read_bytes() != expected_bytes:
        raise PipelineExecutionError("committed compact-output contract differs from runtime")
    manifest_path = contract_path.parent / "manifest.json"
    readme_path = contract_path.parent / "README.md"
    if (
        manifest_path.is_symlink()
        or not manifest_path.is_file()
        or not 0 < manifest_path.stat().st_size <= 64 * 1024
        or readme_path.is_symlink()
        or not readme_path.is_file()
        or readme_path.stat().st_size <= 0
    ):
        raise PipelineExecutionError("compact-output snapshot manifest is missing or unsafe")
    manifest_payload = manifest_path.read_bytes()
    raw = _strict_json(manifest_payload)
    if type(raw) is not dict or manifest_payload != canonical_json_bytes(raw) + b"\n":
        raise PipelineExecutionError("compact-output manifest is not canonical JSON")
    files = raw.get("files")
    if (
        set(raw) != {"contract_version", "files", "manifest_version", "snapshot_sha256"}
        or raw.get("contract_version") != "0.2.0"
        or raw.get("manifest_version") != "0.1.0"
        or type(files) is not dict
        or set(files) != {"README.md", "contract.json"}
    ):
        raise PipelineExecutionError("compact-output manifest inventory is incompatible")
    observed_files = {
        "README.md": _sha256(readme_path),
        "contract.json": _sha256(contract_path),
    }
    if files != observed_files or raw.get("snapshot_sha256") != canonical_sha256(
        {"files": observed_files, "contract_version": "0.2.0"}
    ):
        raise PipelineExecutionError("compact-output snapshot checksum mismatch")
    return observed_files["contract.json"]


def pipeline_stop_file(*, project_root: Path, config: PipelineConfig) -> Path:
    if not isinstance(project_root, Path) or type(config) is not PipelineConfig:
        raise TypeError("pipeline stop path requires exact project/config contracts")
    root = project_root.resolve(strict=True)
    path = root / config.run_root / config.run_name / STOP_REQUEST_FILENAME
    if not path.is_relative_to(root):
        raise ValueError("pipeline stop path escapes the project")
    return path


def request_pipeline_stop(path: Path) -> PipelineStopRequest:
    if not isinstance(path, Path) or path.name != STOP_REQUEST_FILENAME:
        raise TypeError("pipeline stop request requires the canonical marker Path")
    draft = PipelineStopRequest.model_construct(
        requested_at=_utc_now(),
        process_id=os.getpid(),
        checksum_sha256="0" * 64,
    )
    request = _bound_model(draft, PipelineStopRequest)
    _write_contract(path, request)
    return request


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def archive_pipeline_stop(path: Path) -> Path | None:
    if not isinstance(path, Path) or path.name != STOP_REQUEST_FILENAME:
        raise TypeError("pipeline stop archive requires the canonical marker Path")
    if not path.exists() and not path.is_symlink():
        return None
    request = _read_contract(path, PipelineStopRequest, maximum_bytes=16 * 1024)
    archive_root = path.parent / STOP_ARCHIVE_DIRECTORY
    archive_root.mkdir(mode=0o750, exist_ok=True)
    if archive_root.is_symlink() or not archive_root.is_dir():
        raise ValueError("pipeline stop archive directory is unsafe")
    destination = archive_root / f"stop-{request.checksum_sha256}.json"
    if destination.exists() or destination.is_symlink():
        try:
            archived_request = _read_contract(
                destination,
                PipelineStopRequest,
                maximum_bytes=16 * 1024,
            )
        except (TypeError, ValueError) as error:
            raise FileExistsError("conflicting pipeline stop archive already exists") from error
        if archived_request != request:
            raise FileExistsError("conflicting pipeline stop archive already exists")
        path.unlink()
        _fsync_directory(path.parent)
        return destination
    os.replace(path, destination)
    _fsync_directory(archive_root)
    _fsync_directory(path.parent)
    return destination


def build_stop_requested(*, project_root: Path, config: PipelineConfig) -> Callable[[], bool]:
    path = pipeline_stop_file(project_root=project_root, config=config)

    def requested() -> bool:
        if not path.exists() and not path.is_symlink():
            return False
        _read_contract(path, PipelineStopRequest, maximum_bytes=16 * 1024)
        return True

    return requested


def verify_final_evaluation_prerequisites(
    *,
    policy: FinalEvaluationPolicyFreeze,
    review_bundle: ReviewBundleManifest,
    fresh_extension_review: FreshExtensionReview,
    request: FinalEvaluationRequest,
) -> FinalEvaluationAuthorization:
    """Validate data-only future prerequisites; this function never runs evaluation."""

    if (
        type(policy) is not FinalEvaluationPolicyFreeze
        or type(review_bundle) is not ReviewBundleManifest
        or type(fresh_extension_review) is not FreshExtensionReview
        or type(request) is not FinalEvaluationRequest
    ):
        raise TypeError("final evaluation prerequisites require exact review contracts")
    if not policy.development_gate_passed:
        raise FinalEvaluationBlockedError("development gate did not permit future final access")
    if (
        request.policy_sha256 != policy.checksum_sha256
        or request.review_bundle_sha256 != review_bundle.checksum_sha256
        or request.fresh_extension_review_sha256 != fresh_extension_review.checksum_sha256
        or review_bundle.final_policy_sha256 != policy.checksum_sha256
    ):
        raise FinalEvaluationBlockedError("future final-access prerequisite checksums differ")
    draft = FinalEvaluationAuthorization.model_construct(
        authorization_version="future-1.0.0",
        policy_sha256=policy.checksum_sha256,
        review_bundle_sha256=review_bundle.checksum_sha256,
        fresh_extension_manifest_sha256=(fresh_extension_review.fresh_extension_manifest_sha256),
        owner_review_record_sha256=fresh_extension_review.owner_review_record_sha256,
        one_access_nonce_sha256=request.one_access_nonce_sha256,
        checksum_sha256="0" * 64,
    )
    return _bound_model(draft, FinalEvaluationAuthorization)


@dataclass(frozen=True, slots=True)
class _ExecutionInputs:
    v02: V02Config
    v03: V03Config
    v04: V04Config
    v02_dataset_config: DevelopmentDatasetConfig
    v03_dataset_config: DevelopmentDatasetConfig
    baseline_config: BaselineConfig
    tokenizer: ProjectTokenizer
    compact_contract_sha256: str
    frozen_v02_inventory: CompactInventoryReport
    frozen_v03_counterfactual_cap: CounterfactualCapExtensionReport
    frozen_data_source_commit: str

    @property
    def generation_caps(self) -> dict[TaskName, int]:
        caps = self.frozen_v02_inventory.generation_caps
        caps[self.frozen_v03_counterfactual_cap.task_name] = (
            self.frozen_v03_counterfactual_cap.frozen_generation_cap
        )
        return caps


@dataclass(frozen=True, slots=True)
class _EvaluationCandidate:
    candidate_id: str
    result: CompactTrainingResult
    model: TransformerLM
    checkpoint: CheckpointManifest
    device: torch.device


def _run_process(arguments: tuple[str, ...], *, project_root: Path) -> str:
    if (
        type(arguments) is not tuple
        or not arguments
        or arguments[0] != TRUSTED_GIT
        or any(type(argument) is not str or not argument for argument in arguments)
    ):
        raise TypeError("source-control command must use the trusted Git executable")
    git = Path(TRUSTED_GIT)
    if git.is_symlink() or not git.is_file():
        raise PipelineExecutionError("trusted Git executable is unavailable")
    try:
        result = subprocess.run(  # noqa: S603 - fixed absolute executable and bounded argv
            arguments,
            cwd=project_root,
            stdin=subprocess.DEVNULL,
            check=False,
            capture_output=True,
            text=True,
            timeout=30.0,
            env={
                "PATH": "/usr/bin:/bin",
                "LC_ALL": "C",
                "LANG": "C",
            },
        )
    except (OSError, subprocess.SubprocessError):
        raise PipelineExecutionError("source-control provenance check failed safely") from None
    if result.returncode != 0:
        raise PipelineExecutionError("source-control provenance check failed safely")
    output = result.stdout.strip()
    if any(character < " " and character not in "\n\t" for character in output):
        raise PipelineExecutionError("source-control output crossed its text boundary")
    return output


def _verify_runner_source(
    project_root: Path,
    *,
    source_commit: str,
    run_root: str,
) -> str:
    top = Path(
        _run_process(
            (TRUSTED_GIT, "rev-parse", "--show-toplevel"),
            project_root=project_root,
        )
    )
    if top.resolve(strict=True) != project_root.resolve(strict=True):
        raise PipelineExecutionError("project root is not the exact Git worktree root")
    head = _run_process((TRUSTED_GIT, "rev-parse", "HEAD"), project_root=project_root)
    if not re.fullmatch(r"[0-9a-f]{40,64}", head) or not head.startswith(source_commit):
        raise PipelineExecutionError("runner Git revision differs from the run binding")
    status = _run_process(
        (TRUSTED_GIT, "status", "--porcelain=v1", "--untracked-files=all"),
        project_root=project_root,
    )
    run_prefix = run_root.rstrip("/") + "/"
    dirty = []
    for line in status.splitlines():
        if len(line) < 4:
            raise PipelineExecutionError("Git worktree status was malformed")
        path_text = line[3:]
        if " -> " in path_text:
            dirty.append(line)
            continue
        if path_text == run_root.rstrip("/") or path_text.startswith(run_prefix):
            continue
        dirty.append(line)
    if dirty:
        raise PipelineExecutionError("runner Git worktree contains uncommitted source changes")
    return head


def _load_execution_inputs(
    *,
    project_root: Path,
    config: PipelineConfig,
) -> _ExecutionInputs:
    v02_path = _safe_input_path(project_root, config.v02_config_path, kind="file")
    v03_path = _safe_input_path(project_root, config.v03_config_path, kind="file")
    v04_path = _safe_input_path(project_root, config.v04_config_path, kind="file")
    v02 = load_v02_config(v02_path)
    v03 = load_v03_config(v03_path)
    v04 = load_v04_config(v04_path)
    observed = (config_sha256(v02), config_sha256(v03), config_sha256(v04))
    expected = (
        config.v02_config_sha256,
        config.v03_config_sha256,
        config.v04_config_sha256,
    )
    if observed != expected:
        raise PipelineExecutionError("referenced remediation config checksum mismatch")
    if not (
        v02.paths.tokenizer_path == v03.paths.tokenizer_path
        and v02.paths.compact_contract_path == v03.paths.compact_contract_path
        and v04.compact_contract_path == v03.paths.compact_contract_path
    ):
        raise PipelineExecutionError("iteration inputs do not share the frozen tokenizer/contract")
    if (
        not config.stop_before_final_evaluation
        or v04.final_access.automatically_run_final_evaluation
        or not v04.final_access.require_ready_marker
        or not v04.final_access.require_owner_review
        or not v04.final_access.require_explicit_confirm_flag
        or not v04.final_access.one_access_only
        or v04.final_access.historical_golden_packet_permitted
    ):
        raise PipelineExecutionError("final-access boundary differs from the frozen policy")

    tokenizer_path = _safe_input_path(project_root, v02.paths.tokenizer_path, kind="directory")
    tokenizer = ProjectTokenizer.load(tokenizer_path)
    compact_contract = _safe_input_path(project_root, v02.paths.compact_contract_path, kind="file")
    compact_contract_sha256 = _verify_compact_contract(compact_contract)
    v02_dataset_path = _safe_input_path(project_root, v02.paths.dataset_config_path, kind="file")
    v03_dataset_path = _safe_input_path(project_root, v03.paths.dataset_config_path, kind="file")
    v04_dataset_path = _safe_input_path(
        project_root,
        v04.development_dataset_config_path,
        kind="file",
    )
    if v04_dataset_path != v03_dataset_path:
        raise PipelineExecutionError(
            "v0.4 development dataset recipe differs from the frozen v0.3 input"
        )
    v02_dataset_config = load_development_dataset_config(v02_dataset_path)
    v03_dataset_config = load_development_dataset_config(v03_dataset_path)
    if v03_dataset_config.dataset.dataset_version != "0.3.0":
        raise PipelineExecutionError("v0.3 development dataset policy version mismatch")

    baseline_path = _safe_input_path(project_root, v03.baseline_config_path, kind="file")
    baseline_config = load_phase5_config(baseline_path).baselines
    if (
        canonical_sha256(baseline_config.model_dump(mode="json", round_trip=True))
        != v03.baseline_config_sha256
    ):
        raise PipelineExecutionError("baseline config checksum mismatch")

    inventory_path = _safe_input_path(project_root, v02.inventory_report_path, kind="file")
    inventory = _read_contract(inventory_path, CompactInventoryReport)
    if inventory.checksum_sha256 != v02.inventory_report_checksum_sha256:
        raise PipelineExecutionError("frozen v0.2 inventory report checksum mismatch")
    counterfactual_path = _safe_input_path(
        project_root, v03.counterfactual_cap_report_path, kind="file"
    )
    counterfactual = _read_contract(
        counterfactual_path,
        CounterfactualCapExtensionReport,
    )
    if (
        counterfactual.checksum_sha256 != v03.counterfactual_cap_report_checksum_sha256
        or counterfactual.base_inventory_report_sha256 != inventory.checksum_sha256
        or counterfactual.source_commit != inventory.source_commit
        or inventory.tokenizer_manifest_sha256 != tokenizer.manifest.checksum_sha256
        or counterfactual.tokenizer_manifest_sha256 != tokenizer.manifest.checksum_sha256
    ):
        raise PipelineExecutionError("frozen cap reports have incompatible provenance")
    return _ExecutionInputs(
        v02=v02,
        v03=v03,
        v04=v04,
        v02_dataset_config=v02_dataset_config,
        v03_dataset_config=v03_dataset_config,
        baseline_config=baseline_config,
        tokenizer=tokenizer,
        compact_contract_sha256=compact_contract_sha256,
        frozen_v02_inventory=inventory,
        frozen_v03_counterfactual_cap=counterfactual,
        frozen_data_source_commit=inventory.source_commit,
    )


def verify_v02_prefix_reuse(project_root: Path, config: PipelineConfig) -> tuple[Path, ...]:
    """Reopen the targeted run's immutable v0.2 evidence without copying or locking it."""

    policy = config.reuse_v02_prefix
    if policy is None:
        return ()
    root = _safe_input_path(project_root, policy.source_run_root, kind="directory")
    manifest_path = root / "run-manifest.json"
    manifest = _read_contract(manifest_path, RunManifest)
    if (
        manifest.checksum_sha256 != policy.source_run_manifest_sha256
        or manifest.source_commit != policy.source_commit
        or manifest.v02_config_sha256 != config.v02_config_sha256
    ):
        raise PipelineExecutionError("preserved v0.2 run provenance differs")

    def verified_reference(reference: ArtifactReference) -> Path:
        path = root / reference.relative_path
        cursor = root
        for part in Path(reference.relative_path).parts:
            cursor /= part
            if cursor.is_symlink():
                raise PipelineExecutionError("preserved v0.2 evidence traverses a symlink")
        if (
            not path.is_file()
            or not path.resolve(strict=True).is_relative_to(root)
            or path.stat().st_size != reference.size_bytes
            or hashlib.sha256(path.read_bytes()).hexdigest() != reference.sha256
        ):
            raise PipelineExecutionError("preserved v0.2 artifact checksum or size differs")
        return path

    stage_prefix = (
        "v02_inventory_and_caps",
        "v02_smoke",
        "v02_development_training",
        "v02_development_gate",
    )
    checked: list[Path] = [manifest_path]
    for ordinal, stage in enumerate(stage_prefix, start=1):
        marker_path = root / f"stages/{ordinal:02d}-{stage}/completed.json"
        marker = _read_contract(marker_path, StageCompletionMarker)
        if (
            marker.run_name != manifest.run_name
            or marker.pipeline_config_sha256 != manifest.pipeline_config_sha256
            or marker.source_commit != manifest.source_commit
            or marker.stage != stage
            or marker.ordinal != ordinal
            or marker.attempt != 1
        ):
            raise PipelineExecutionError("preserved v0.2 completion prefix differs")
        outcome_path = verified_reference(marker.outcome)
        outcome = _read_contract(outcome_path, StageOutcome)
        if not outcome.advancement_allowed:
            raise PipelineExecutionError("preserved v0.2 stage did not advance")
        checked.extend((marker_path, outcome_path))
        for reference in outcome.artifacts:
            checked.append(verified_reference(reference))

    if len(checked) != 21 or len(checked) != len(set(checked)):
        raise PipelineExecutionError("preserved v0.2 evidence inventory is invalid")
    for path in checked:
        if (
            path.is_symlink()
            or not path.is_file()
            or not path.resolve(strict=True).is_relative_to(root)
        ):
            raise PipelineExecutionError("preserved v0.2 evidence path is unsafe")
    observed = tuple(
        sorted(
            (
                path.relative_to(root).as_posix(),
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            for path in checked
        )
    )
    if (
        observed != policy.evidence
        or canonical_sha256(observed) != policy.evidence_inventory_sha256
    ):
        raise PipelineExecutionError("preserved v0.2 evidence differs from its external pin")
    return tuple(checked)


def _v02_reuse_outcome(
    context: StageContext, config: PipelineConfig, stage_name: str
) -> StageOutcome:
    """Write only a small new reference packet for one verified v0.2 stage."""

    policy = config.reuse_v02_prefix
    if policy is None or stage_name not in policy.verify_only_stages:
        raise PipelineExecutionError("v0.2 reuse stage is not enabled by the targeted policy")
    evidence = verify_v02_prefix_reuse(context.project_root, config)
    root = context.project_root / policy.source_run_root
    rows = tuple(
        sorted(
            (
                path.relative_to(root).as_posix(),
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            for path in evidence
        )
    )
    draft = V02PrefixReuseReport.model_construct(
        stage_name=stage_name,
        source_run_manifest_sha256=policy.source_run_manifest_sha256,
        source_commit=policy.source_commit,
        evidence=rows,
        checksum_sha256="0" * 64,
    )
    report = _bound_model(draft, V02PrefixReuseReport)
    artifact = _contract_artifact(context, f"{stage_name}-reuse.json", report)
    return _stage_outcome(
        "Verified immutable v0.2 evidence was reused; no training or decoding was invoked.",
        artifacts=(artifact,),
        metrics=(
            StageMetric(name="verified_evidence_files", value=float(len(rows)), unit="files"),
        ),
    )


def _process_peak_rss_bytes() -> int:
    observed = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system() != "Darwin":
        observed *= 1024
    return observed


def _run_size_bytes(run_directory: Path) -> int:
    if run_directory.is_symlink() or not run_directory.is_dir():
        raise PipelineResourceLimitError("pipeline run directory is unsafe")
    total = 0
    count = 0
    pending = [run_directory]
    while pending:
        directory = pending.pop()
        for child in directory.iterdir():
            count += 1
            if count > MAX_RUN_FILES:
                raise PipelineResourceLimitError("pipeline run file-count bound was exceeded")
            if child.is_symlink():
                raise PipelineResourceLimitError("pipeline run contains a symlink")
            if child.is_dir():
                pending.append(child)
            elif child.is_file():
                total += child.stat().st_size
            else:
                raise PipelineResourceLimitError("pipeline run contains a non-file entry")
    return total


class _ResourceGuard:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._last_resource_poll = 0.0

    def _elapsed_seconds(self, context: StageContext) -> float:
        snapshot = context.progress.snapshot()
        if type(snapshot) is not ProgressSnapshot:
            raise PipelineExecutionError("pipeline active-runtime evidence is invalid")
        return float(snapshot.elapsed_seconds)

    def resource_stop_required(self, context: StageContext, *, force: bool) -> bool:
        now = time.monotonic()
        if not force and now - self._last_resource_poll < 5.0:
            return False
        self._last_resource_poll = now
        event_log = context.run_directory / PROGRESS_EVENT_LOG_FILENAME
        event_bytes = event_log.stat().st_size if event_log.is_file() else 0
        return (
            self._elapsed_seconds(context) >= self.config.maximum_pipeline_seconds
            or _process_peak_rss_bytes() >= self.config.maximum_process_rss_bytes
            or _run_size_bytes(context.run_directory) >= self.config.maximum_run_bytes
            or event_bytes >= self.config.maximum_event_log_bytes
        )

    def stop_required(self, context: StageContext, *, force_resources: bool = False) -> bool:
        external = context.stop_requested()
        if type(external) is not bool:
            raise TypeError("stage stop callback must return an exact boolean")
        return external or self.resource_stop_required(context, force=force_resources)

    def enforce_start(self, context: StageContext) -> None:
        if self.stop_required(context, force_resources=True):
            raise KeyboardInterrupt

    def enforce_projected_write(
        self,
        context: StageContext,
        *,
        reservation_bytes: int,
    ) -> None:
        """Refuse before bounded scientific writes could reach the run ceiling."""

        if type(reservation_bytes) is not int or reservation_bytes < 1:
            raise TypeError("projected-write reservation must be a positive integer")
        observed = _run_size_bytes(context.run_directory)
        if observed >= self.config.maximum_run_bytes - reservation_bytes:
            raise PipelineResourceLimitError(
                "pipeline lacks capacity for the bounded scientific write"
            )

    def enforce_end(self, context: StageContext) -> None:
        if self.resource_stop_required(context, force=True):
            raise PipelineResourceLimitError("pipeline reached a configured resource boundary")


def _stage_outcome(
    summary: str,
    *,
    advancement_allowed: bool = True,
    artifacts: tuple[ArtifactReference, ...] = (),
    metrics: tuple[StageMetric, ...] = (),
    warnings: tuple[str, ...] = (),
) -> StageOutcome:
    return StageOutcome(
        summary=summary,
        advancement_allowed=advancement_allowed,
        artifacts=tuple(sorted(artifacts, key=lambda item: item.relative_path)),
        metrics=tuple(sorted(metrics, key=lambda item: item.name)),
        warnings=warnings,
    )


def _contract_artifact(
    context: StageContext,
    filename: str,
    model: ContractModel,
) -> ArtifactReference:
    path = context.attempt_directory / filename
    _write_contract(path, model)
    return _artifact_reference(path, run_directory=context.run_directory)


def _directory_artifacts(
    directory: Path,
    *,
    run_directory: Path,
) -> tuple[ArtifactReference, ...]:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("pipeline artifact directory is unsafe")
    files: list[Path] = []
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise ValueError("pipeline artifact directory contains a symlink")
        if path.is_file():
            files.append(path)
        elif not path.is_dir():
            raise ValueError("pipeline artifact directory contains a non-file entry")
    return tuple(
        _artifact_reference(path, run_directory=run_directory)
        for path in sorted(files, key=lambda item: item.relative_to(run_directory).as_posix())
    )


def _upstream_attempt(
    context: StageContext,
    config: PipelineConfig,
    stage_name: str,
) -> Path:
    if stage_name not in PIPELINE_STAGES:
        raise ValueError("upstream stage is outside the frozen graph")
    store = PipelineStore(
        context.run_directory,
        maximum_state_bytes=config.maximum_status_bytes,
    )
    state = store.load_state()
    record = state.stages[PIPELINE_STAGES.index(stage_name)]
    if record.status is not StageStatus.COMPLETED or record.latest_attempt_path is None:
        raise PipelineExecutionError("required upstream stage is not durably completed")
    path = context.run_directory / record.latest_attempt_path
    if path.is_symlink() or not path.is_dir():
        raise PipelineExecutionError("upstream attempt directory is unsafe")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(context.run_directory.resolve(strict=True)):
        raise PipelineExecutionError("upstream attempt escapes the run directory")
    return resolved


def _tokenized_inventory_sha256(
    examples: tuple[CompactTokenizedExample, ...],
) -> str:
    """Delegate reconstruction to the training contract's single canonical hash."""

    return tokenized_inventory_sha256(examples)


def _tokenize_examples(
    examples: tuple[RemediationExample, ...],
    tokenizer: ProjectTokenizer,
    *,
    context_length: int,
    generation_caps: Mapping[TaskName, int],
) -> tuple[CompactTokenizedExample, ...]:
    return tuple(
        tokenize_compact_example(
            example,
            tokenizer,
            context_length=context_length,
            generation_caps=generation_caps,
        )
        for example in examples
    )


def _subset_dataset(
    dataset: SafeDevelopmentDataset,
    examples: tuple[RemediationExample, ...],
    *,
    dataset_version: str,
) -> SafeDevelopmentDataset:
    ordered = tuple(sorted(examples, key=lambda item: item.example_id))
    if not ordered or len({item.example_id for item in ordered}) != len(ordered):
        raise ValueError("safe dataset subset must be non-empty and unique")
    payload = b"".join(
        canonical_json_bytes(item.model_dump(mode="json", round_trip=True)) + b"\n"
        for item in ordered
    )
    views = tuple(view for view in RemediationView if any(item.view is view for item in ordered))
    view_counts = Counter(item.view for item in ordered)
    task_counts = Counter(item.task_name for item in ordered)
    draft = SafeDevelopmentManifest.model_construct(
        artifact_version="0.3.0",
        boundary="development_only_no_final_or_golden_payloads",
        source_commit=dataset.manifest.source_commit,
        dataset_version=dataset_version,
        dataset_config_sha256=dataset.manifest.dataset_config_sha256,
        compact_contract_version="0.2.0",
        views=views,
        example_count=len(ordered),
        counts_by_view=tuple((view, view_counts[view]) for view in views),
        counts_by_task=tuple((task, task_counts[task]) for task in TaskName if task_counts[task]),
        examples_sha256=hashlib.sha256(payload).hexdigest(),
        examples_size_bytes=len(payload),
        inventory_sha256=canonical_sha256(
            tuple((item.example_id, item.checksum_sha256) for item in ordered)
        ),
        checksum_sha256="0" * 64,
    )
    manifest = _bound_model(draft, SafeDevelopmentManifest)
    return SafeDevelopmentDataset(manifest=manifest, examples=ordered)


def _write_predictions(
    context: StageContext,
    *,
    stem: str,
    view: RemediationView,
    examples: tuple[RemediationExample, ...],
    predictions: tuple[DualPathCompactPrediction, ...],
) -> tuple[PredictionArtifactManifest, tuple[ArtifactReference, ArtifactReference]]:
    if (
        not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,95}", stem)
        or not examples
        or len(examples) != len(predictions)
    ):
        raise ValueError("prediction artifact inputs are invalid")
    ordered_examples = tuple(sorted(examples, key=lambda item: item.example_id))
    ordered_predictions = tuple(sorted(predictions, key=lambda item: item.example_id))
    for example, prediction in zip(ordered_examples, ordered_predictions, strict=True):
        if (
            example.view is not view
            or prediction.example_id != example.example_id
            or prediction.example_checksum_sha256 != example.checksum_sha256
        ):
            raise ValueError("prediction artifact provenance mismatch")
    predictions_path = context.attempt_directory / f"{stem}.jsonl"
    if predictions_path.exists() or predictions_path.is_symlink():
        raise FileExistsError("prediction artifact must not overwrite")
    prediction_bytes = canonical_prediction_jsonl_bytes(ordered_predictions)
    rows = prediction_bytes.splitlines(keepends=True)
    if len(rows) != len(ordered_predictions) or any(
        len(row) > MAX_PREDICTION_ROW_BYTES for row in rows
    ):
        raise ValueError("one prediction row exceeds its byte bound")
    prediction_sha256 = prediction_artifact_byte_sha256(ordered_predictions)
    if hashlib.sha256(prediction_bytes).hexdigest() != prediction_sha256:
        raise PipelineExecutionError("canonical prediction byte contract is inconsistent")
    _write_bytes(predictions_path, prediction_bytes)
    draft = PredictionArtifactManifest.model_construct(
        view=view,
        example_count=len(ordered_examples),
        example_inventory_sha256=canonical_sha256(
            tuple((item.example_id, item.checksum_sha256) for item in ordered_examples)
        ),
        prediction_inventory_sha256=canonical_sha256(
            tuple((item.example_id, item.checksum_sha256) for item in ordered_predictions)
        ),
        predictions_sha256=prediction_sha256,
        predictions_size_bytes=len(prediction_bytes),
        checksum_sha256="0" * 64,
    )
    manifest = _bound_model(draft, PredictionArtifactManifest)
    manifest_path = context.attempt_directory / f"{stem}-manifest.json"
    _write_contract(manifest_path, manifest)
    return manifest, (
        _artifact_reference(manifest_path, run_directory=context.run_directory),
        _artifact_reference(predictions_path, run_directory=context.run_directory),
    )


def _read_predictions(
    *,
    manifest_path: Path,
    predictions_path: Path,
    view: RemediationView,
    examples: tuple[RemediationExample, ...],
) -> tuple[PredictionArtifactManifest, tuple[DualPathCompactPrediction, ...]]:
    """Reopen one immutable prediction artifact and prove its exact example binding."""

    if not examples:
        raise ValueError("prediction evidence requires a non-empty example inventory")
    manifest = _read_contract(manifest_path, PredictionArtifactManifest)
    if (
        predictions_path.is_symlink()
        or not predictions_path.is_file()
        or not 0 < predictions_path.stat().st_size <= MAX_PIPELINE_JSON_BYTES
        or predictions_path.stat().st_size != manifest.predictions_size_bytes
    ):
        raise ValueError("prediction evidence is missing, unsafe, or oversized")
    payload = predictions_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != manifest.predictions_sha256:
        raise PipelineExecutionError("prediction evidence byte checksum differs")
    rows = payload.splitlines(keepends=True)
    if (
        len(rows) != len(examples)
        or len(rows) != manifest.example_count
        or any(not row.endswith(b"\n") or len(row) > MAX_PREDICTION_ROW_BYTES for row in rows)
    ):
        raise PipelineExecutionError("prediction evidence row inventory differs")
    predictions = tuple(
        DualPathCompactPrediction.model_validate_json(row, strict=True) for row in rows
    )
    ordered_examples = tuple(sorted(examples, key=lambda item: item.example_id))
    ordered_predictions = tuple(sorted(predictions, key=lambda item: item.example_id))
    if (
        predictions != ordered_predictions
        or canonical_prediction_jsonl_bytes(predictions) != payload
    ):
        raise PipelineExecutionError("prediction evidence is not canonical identity-ordered JSONL")
    for example, prediction in zip(ordered_examples, ordered_predictions, strict=True):
        if (
            example.view is not view
            or prediction.example_id != example.example_id
            or prediction.example_checksum_sha256 != example.checksum_sha256
        ):
            raise PipelineExecutionError("prediction evidence example provenance differs")
    if (
        manifest.view is not view
        or manifest.example_inventory_sha256
        != canonical_sha256(
            tuple((item.example_id, item.checksum_sha256) for item in ordered_examples)
        )
        or manifest.prediction_inventory_sha256
        != canonical_sha256(
            tuple((item.example_id, item.checksum_sha256) for item in ordered_predictions)
        )
        or prediction_artifact_byte_sha256(ordered_predictions) != manifest.predictions_sha256
    ):
        raise PipelineExecutionError("prediction evidence manifest differs from its payload")
    return manifest, ordered_predictions


def _decode_examples(
    model: TransformerLM,
    tokenizer: ProjectTokenizer,
    examples: tuple[RemediationExample, ...],
    *,
    generation_caps: Mapping[TaskName, int],
    device: torch.device,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[DualPathCompactPrediction, ...]:
    """Decode at per-example atomic boundaries with optional cooperative progress."""

    if type(examples) is not tuple:
        raise TypeError("decode examples must be an exact tuple")
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("decode progress callback must be callable")
    example_ids = tuple(getattr(item, "example_id", None) for item in examples)
    if any(type(example_id) is not str or not example_id for example_id in example_ids) or len(
        example_ids
    ) != len(set(example_ids)):
        raise ValueError("decode example IDs must be non-empty and globally unique")
    results: list[DualPathCompactPrediction] = []
    total = len(examples)
    for start in range(0, total, COOPERATIVE_DECODE_CHUNK_SIZE):
        results.extend(
            decode_compact_examples(
                model,
                tokenizer,
                examples[start : start + COOPERATIVE_DECODE_CHUNK_SIZE],
                generation_caps=generation_caps,
                device=device,
            )
        )
        if progress_callback is not None:
            progress_callback(
                min(start + COOPERATIVE_DECODE_CHUNK_SIZE, total),
                total,
            )
    return tuple(results)


def _raise_if_stop(context: StageContext, guard: _ResourceGuard) -> None:
    if guard.stop_required(context):
        raise KeyboardInterrupt


def _guarded_decode_examples(
    context: StageContext,
    *,
    guard: _ResourceGuard,
    model: TransformerLM,
    tokenizer: ProjectTokenizer,
    examples: tuple[RemediationExample, ...],
    generation_caps: Mapping[TaskName, int],
    device: torch.device,
    progress_message: str,
) -> tuple[DualPathCompactPrediction, ...]:
    """Decode one complete view while polling after every atomic example."""

    if not examples:
        raise PipelineExecutionError("guarded decoding requires at least one example")
    _raise_if_stop(context, guard)
    context.progress.report(
        message=progress_message,
        completed_units=0,
        total_units=len(examples),
    )

    def progress(completed_units: int, total_units: int) -> None:
        if completed_units % DECODE_PROGRESS_REPORT_INTERVAL == 0 or completed_units == total_units:
            context.progress.report(
                message=progress_message,
                completed_units=completed_units,
                total_units=total_units,
            )
        _raise_if_stop(context, guard)

    predictions = _decode_examples(
        model,
        tokenizer,
        examples,
        generation_caps=generation_caps,
        device=device,
        progress_callback=progress,
    )
    _raise_if_stop(context, guard)
    return predictions


def _semantic_report_composite(report: SemanticEvaluationReport) -> float:
    """Recompute the frozen ranking score from immutable semantic report fields."""

    if type(report) is not SemanticEvaluationReport:
        raise TypeError("semantic report composite requires an exact report")
    metrics = report.view_metrics.metrics
    if report.constrained.schema_validity_rate != metrics.constrained_schema_validity_rate.estimate:
        raise ValueError("semantic report schema-validity fields disagree")
    if report.constrained.schema_validity_rate != 1.0:
        return 0.0
    values = [report.constrained.exact_match_rate]
    supported_quality = (
        metrics.fault_family_macro_f1,
        metrics.next_action_macro_f1,
        metrics.continuation_macro_f1,
        metrics.evidence_f1,
        metrics.required_abstention_accuracy,
    )
    values.extend(item.estimate for item in supported_quality if item.support > 0)
    if metrics.no_fault_false_positive_rate.support > 0:
        values.append(1.0 - metrics.no_fault_false_positive_rate.estimate)
    values.extend(
        (
            1.0 - metrics.expected_calibration_error.estimate,
            1.0 - metrics.selective_risk_at_80_percent_coverage.estimate,
        )
    )
    score = sum(values) / len(values)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise PipelineExecutionError("semantic report composite is outside its bound")
    return score


def _semantic_reports_differ_only_in_confidence(
    raw: SemanticEvaluationReport,
    calibrated: SemanticEvaluationReport,
) -> bool:
    """Compare reports after removing only the two confidence-derived estimates."""

    payloads: list[dict[str, object]] = []
    for report in (raw, calibrated):
        payload = cast(dict[str, object], report.model_dump(mode="json", round_trip=True))
        view_metrics = cast(dict[str, object], payload["view_metrics"])
        metrics = cast(dict[str, object], view_metrics["metrics"])
        metrics.pop("expected_calibration_error")
        metrics.pop("selective_risk_at_80_percent_coverage")
        view_metrics.pop("checksum_sha256")
        payload.pop("checksum_sha256")
        payloads.append(payload)
    return payloads[0] == payloads[1]


def _calibration_observations_by_identity(
    examples: tuple[RemediationExample, ...],
    predictions: tuple[DualPathCompactPrediction, ...],
) -> tuple[CalibrationObservation, ...]:
    """Bind calibration truth and predictions by immutable example identity."""

    ordered_examples = tuple(sorted(examples, key=lambda item: item.example_id))
    ordered_predictions = tuple(sorted(predictions, key=lambda item: item.example_id))
    if len(ordered_examples) != len(ordered_predictions):
        raise PipelineExecutionError("calibration example and prediction counts differ")
    observations: list[CalibrationObservation] = []
    for example, prediction in zip(ordered_examples, ordered_predictions, strict=True):
        if (
            prediction.example_id != example.example_id
            or prediction.example_checksum_sha256 != example.checksum_sha256
        ):
            raise PipelineExecutionError("calibration prediction identity binding differs")
        observations.append(
            CalibrationObservation(
                example_id=example.example_id,
                raw_confidence=float(
                    prediction.constrained.selected_token_geometric_mean_probability
                ),
                exact_match=(
                    prediction.constrained.canonical_target_json == example.canonical_target_json
                ),
            )
        )
    return tuple(observations)


def _reconstruct_targeted_v03_gate(
    *,
    selection: CandidateSelectionReport,
    calibration_selection: CalibrationSelectionManifest,
    calibration: TemperatureCalibrationReport,
    calibration_examples: tuple[RemediationExample, ...],
    calibration_prediction_manifest: PredictionArtifactManifest,
    calibration_predictions: tuple[DualPathCompactPrediction, ...],
    gate_examples: tuple[RemediationExample, ...],
    raw_prediction_manifest: PredictionArtifactManifest,
    raw_predictions: tuple[DualPathCompactPrediction, ...],
    raw_baseline: RemediationBaselineReport,
    expected_artifacts: DevelopmentArtifactBinding,
    saved_raw_evaluation: SemanticEvaluationReport,
    saved_calibrated_evaluation: SemanticEvaluationReport,
) -> tuple[V03AcceptanceResult, TargetedV03GateBinding]:
    """Independently reconstruct every targeted acceptance input from raw evidence."""

    observations = _calibration_observations_by_identity(
        calibration_examples,
        calibration_predictions,
    )
    reconstructed_calibration = fit_temperature(
        observations,
        calibration_selection_manifest_sha256=calibration_selection.checksum_sha256,
        calibration_prediction_manifest_sha256=(calibration_prediction_manifest.checksum_sha256),
        calibration_predictions_sha256=calibration_prediction_manifest.predictions_sha256,
        selected_checkpoint_manifest_sha256=selection.selected_checkpoint_manifest_sha256,
    )
    if calibration != reconstructed_calibration:
        raise PipelineExecutionError(
            "temperature calibration differs from independently reopened evidence"
        )
    reconstructed_raw = evaluate_semantic_predictions(
        view=RemediationView.IID_VALIDATION,
        examples=gate_examples,
        predictions=raw_predictions,
        baseline_report=raw_baseline,
        artifacts=expected_artifacts,
    )
    reconstructed_calibrated = evaluate_semantic_predictions(
        view=RemediationView.IID_VALIDATION,
        examples=gate_examples,
        predictions=raw_predictions,
        baseline_report=raw_baseline,
        artifacts=expected_artifacts,
        confidence_transform=lambda value: apply_temperature(
            value, calibration.selected_temperature
        ),
    )
    if (
        saved_raw_evaluation != reconstructed_raw
        or saved_calibrated_evaluation != reconstructed_calibrated
        or raw_prediction_manifest.predictions_sha256 != saved_raw_evaluation.predictions_sha256
        or not _semantic_reports_differ_only_in_confidence(
            saved_raw_evaluation, saved_calibrated_evaluation
        )
    ):
        raise PipelineExecutionError(
            "targeted gate reports differ from independently reopened evidence"
        )
    acceptance = evaluate_v03_acceptance(reconstructed_calibrated.view_metrics)
    draft_binding = TargetedV03GateBinding.model_construct(
        candidate_selection_sha256=selection.checksum_sha256,
        calibration_selection_sha256=calibration_selection.checksum_sha256,
        temperature_calibration_sha256=calibration.checksum_sha256,
        raw_evaluation_sha256=reconstructed_raw.checksum_sha256,
        calibrated_evaluation_sha256=reconstructed_calibrated.checksum_sha256,
        acceptance_sha256=acceptance.checksum_sha256,
        raw_prediction_artifact_sha256=raw_prediction_manifest.predictions_sha256,
        calibrated_prediction_artifact_sha256=raw_prediction_manifest.predictions_sha256,
        outputs_bit_exact=True,
        thresholds_unchanged=True,
        checksum_sha256="0" * 64,
    )
    return acceptance, _bound_model(draft_binding, TargetedV03GateBinding)


def _require_semantic_report_scope(
    report: SemanticEvaluationReport,
    *,
    view: RemediationView,
    example_count: int,
    dataset_manifest_sha256: str,
    source_commit: str,
    config_sha256_value: str,
    tokenizer_manifest_sha256: str,
    output_contract_sha256: str,
    checkpoint_manifest_sha256: str,
) -> None:
    """Require one semantic report to cover an exact frozen development view."""

    if type(report) is not SemanticEvaluationReport:
        raise TypeError("semantic scope verification requires an exact report")
    artifacts = report.view_metrics.artifacts
    if (
        report.evaluation_view is not view
        or report.view_metrics.view is not _DEVELOPMENT_VIEW_BY_REMEDIATION[view]
        or report.example_count != example_count
        or report.view_metrics.sample_count != example_count
        or artifacts.source_commit != source_commit
        or artifacts.config_sha256 != config_sha256_value
        or artifacts.dataset_manifest_sha256 != dataset_manifest_sha256
        or artifacts.tokenizer_manifest_sha256 != tokenizer_manifest_sha256
        or artifacts.output_contract_sha256 != output_contract_sha256
        or artifacts.checkpoint_sha256 != checkpoint_manifest_sha256
        or report.predictions_sha256 != artifacts.prediction_artifact_sha256
        or report.baseline_report_sha256 != artifacts.comparator_artifact_sha256
    ):
        raise PipelineExecutionError(
            "semantic evaluation report differs from its exact frozen development scope"
        )


def _free_running_structural_failure_score(
    predictions: tuple[DualPathCompactPrediction, ...],
) -> float:
    """Mean of six truth-independent binary failures for v0.2 selection.

    Each example contributes constrained parse/schema/cap failures followed by the
    same three unconstrained failures.  Lower is better; the training core then uses
    validation NLL and earlier step only as deterministic tie-breakers.
    """

    if not predictions:
        raise ValueError("structural checkpoint selection requires predictions")
    failures = sum(
        int(not prediction.constrained.compact_parse_success)
        + int(not prediction.constrained.schema_valid)
        + int(prediction.constrained.generation_cap_exhausted)
        + int(not prediction.unconstrained.compact_parse_success)
        + int(not prediction.unconstrained.schema_valid)
        + int(prediction.unconstrained.generation_cap_exhausted)
        for prediction in predictions
    )
    return float(failures / (6 * len(predictions)))


def _exact_semantic_rate_and_mean_latency(
    examples: tuple[RemediationExample, ...],
    predictions: tuple[DualPathCompactPrediction, ...],
    *,
    path_name: Literal["constrained", "unconstrained"],
) -> tuple[float, float]:
    """Measure exact canonical output and raw decoder latency for one path."""

    examples_by_id = {item.example_id: item for item in examples}
    if (
        not predictions
        or len(predictions) != len(examples_by_id)
        or {item.example_id for item in predictions} != set(examples_by_id)
    ):
        raise PipelineExecutionError("v0.2 behavioral report inventory differs from examples")
    paths = tuple(getattr(item, path_name) for item in predictions)
    exact_count = sum(
        path.canonical_target_json == examples_by_id[prediction.example_id].canonical_target_json
        for prediction, path in zip(predictions, paths, strict=True)
    )
    return exact_count / len(paths), sum(path.elapsed_seconds for path in paths) / len(paths)


def _smoke_examples(
    dataset: SafeDevelopmentDataset,
) -> tuple[tuple[RemediationExample, ...], tuple[RemediationExample, ...]]:
    train = tuple(item for item in dataset.examples if item.view is RemediationView.IID_TRAIN)
    validation = tuple(
        item for item in dataset.examples if item.view is RemediationView.IID_VALIDATION
    )
    if not train or not validation:
        raise ValueError("smoke training requires IID train and validation support")
    return train[:SMOKE_EXAMPLES_PER_VIEW], validation[:SMOKE_EXAMPLES_PER_VIEW]


def _sequence_length_inventory_sha256(
    examples: tuple[CompactTokenizedExample, ...],
) -> str:
    """Bind the exact token length used to select a pilot row for every example."""

    if not examples or len({item.example_id for item in examples}) != len(examples):
        raise ValueError("pilot length inventory must be non-empty and unique")
    return canonical_sha256(
        tuple(
            sorted(
                (
                    item.example_id,
                    item.task_name.value,
                    item.group_id,
                    len(item.token_ids),
                )
                for item in examples
            )
        )
    )


def _longest_pilot_examples_per_task(
    examples: tuple[RemediationExample, ...],
    tokenized: tuple[CompactTokenizedExample, ...],
) -> tuple[tuple[RemediationExample, ...], tuple[CompactTokenizedExample, ...]]:
    """Choose one deterministic longest sequence per task for the MPS pilot.

    The complete view is profiled first.  Selecting the longest row for every task
    retains task coverage while guaranteeing that at least one globally longest row
    is part of the bounded pilot rather than relying on the first dataset rows.
    """

    if not examples or not tokenized or len(examples) != len(tokenized):
        raise ValueError("pilot examples and tokenized inventory must align")
    examples_by_id = {item.example_id: item for item in examples}
    tokenized_by_id = {item.example_id: item for item in tokenized}
    if (
        len(examples_by_id) != len(examples)
        or len(tokenized_by_id) != len(tokenized)
        or set(examples_by_id) != set(tokenized_by_id)
    ):
        raise ValueError("pilot examples and tokenized identifiers must match uniquely")
    for example_id, example in examples_by_id.items():
        encoded = tokenized_by_id[example_id]
        if encoded.task_name is not example.task_name or encoded.group_id != example.group_id:
            raise ValueError("pilot tokenized lineage differs from its source example")

    selected_examples: list[RemediationExample] = []
    selected_tokenized: list[CompactTokenizedExample] = []
    for task_name in TaskName:
        candidates = tuple(item for item in tokenized if item.task_name is task_name)
        if not candidates:
            raise ValueError("pilot inventory must contain every compact-output task")
        selected = min(candidates, key=lambda item: (-len(item.token_ids), item.example_id))
        selected_examples.append(examples_by_id[selected.example_id])
        selected_tokenized.append(selected)

    if max(len(item.token_ids) for item in selected_tokenized) != max(
        len(item.token_ids) for item in tokenized
    ):
        raise PipelineExecutionError("pilot selection omitted the globally longest sequence")
    return tuple(selected_examples), tuple(selected_tokenized)


def _pilot_exercises_global_maximum(
    examples: tuple[CompactTokenizedExample, ...],
    *,
    batch_size: int,
    seed: int,
    steps: int,
) -> bool:
    """Prove the configured task-balanced pilot actually draws a maximum row."""

    if not examples:
        raise ValueError("pilot sampling proof requires examples")
    maximum_length = max(len(item.token_ids) for item in examples)
    maximum_indices = {
        index for index, item in enumerate(examples) if len(item.token_ids) == maximum_length
    }
    return any(
        maximum_indices.intersection(
            task_balanced_batch_indices(
                examples,
                batch_size=batch_size,
                seed=seed,
                step=step,
            )
        )
        for step in range(steps)
    )


def _smoke_model_config(base: TransformerConfig) -> TransformerConfig:
    return TransformerConfig(
        model_version=base.model_version,
        layers=2,
        width=64,
        heads=4,
        context_length=base.context_length,
        feed_forward_multiplier=2,
        dropout=0.0,
        tie_embeddings=base.tie_embeddings,
        bias=base.bias,
    )


def _smoke_training_config(base: RemediationTraining) -> RemediationTraining:
    return RemediationTraining(
        seed=base.seed,
        device="cpu",
        allow_cpu_fallback=True,
        steps=SMOKE_STEPS,
        batch_size=min(2, base.batch_size),
        learning_rate=base.learning_rate,
        weight_decay=base.weight_decay,
        gradient_clip_norm=base.gradient_clip_norm,
        evaluation_interval=1,
        durable_checkpoint_interval=1,
    )


def _latest_resume_source(
    context: StageContext,
    candidate_id: str,
) -> Path | None:
    return _latest_resume_from_roots(_prior_training_state_roots(context, candidate_id))


@dataclass(frozen=True, slots=True)
class _PriorTrainingStateRoot:
    attempt_number: int
    root: Path
    latest_step: int | None
    latest_state: Path | None


def _latest_resume_from_roots(
    roots: tuple[_PriorTrainingStateRoot, ...],
) -> Path | None:
    committed = tuple(
        (item.attempt_number, item.latest_step, item.latest_state)
        for item in roots
        if item.latest_state is not None and item.latest_step is not None
    )
    if not committed:
        return None
    return max(committed, key=lambda item: (item[1], item[0]))[2]


def _prior_training_state_roots(
    context: StageContext,
    candidate_id: str,
) -> tuple[_PriorTrainingStateRoot, ...]:
    """Verify every earlier candidate root and inventory its newest durable state."""

    if (
        type(candidate_id) is not str
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,95}", candidate_id) is None
    ):
        raise ValueError("training candidate identifier is invalid")
    stage_root = context.attempt_directory.parent
    if (
        stage_root.is_symlink()
        or not stage_root.is_dir()
        or context.attempt_directory.is_symlink()
        or not context.attempt_directory.is_dir()
    ):
        raise ValueError("current training stage attempt is unsafe")
    current_match = re.fullmatch(r"attempt-([0-9]{4})", context.attempt_directory.name)
    if current_match is None:
        raise ValueError("current stage attempt name is invalid")
    current = int(current_match.group(1))
    roots: list[_PriorTrainingStateRoot] = []
    for attempt in sorted(stage_root.iterdir(), key=lambda path: path.name):
        match = re.fullmatch(r"attempt-([0-9]{4})", attempt.name)
        if match is None:
            if attempt.name != "completed.json" or attempt.is_symlink() or not attempt.is_file():
                raise ValueError("training stage contains an unsafe attempt entry")
            continue
        if attempt.is_symlink() or not attempt.is_dir():
            raise ValueError("training stage contains an unsafe attempt entry")
        attempt_number = int(match.group(1))
        if attempt_number >= current:
            continue
        state_root = attempt / "training-state" / candidate_id
        state_parent = state_root.parent
        if (
            state_parent.is_symlink()
            or (state_parent.exists() and not state_parent.is_dir())
            or state_root.is_symlink()
        ):
            raise ValueError("previous durable training root traverses a symlink")
        if not state_root.exists():
            continue
        if not state_root.is_dir():
            raise ValueError("previous durable training root is unsafe")
        resolved_attempt = attempt.resolve(strict=True)
        resolved_root = state_root.resolve(strict=True)
        if not resolved_root.is_relative_to(resolved_attempt):
            raise ValueError("previous durable training root escapes its attempt")
        latest = latest_committed_training_state(state_root, candidate_id=candidate_id)
        latest_step: int | None = None
        if latest is not None:
            step_match = re.fullmatch(r"state-step-([0-9]{8})", latest.name)
            if step_match is None or latest.parent != resolved_root:
                raise PipelineExecutionError("verified resume state identity is invalid")
            latest_step = int(step_match.group(1))
        roots.append(
            _PriorTrainingStateRoot(
                attempt_number=attempt_number,
                root=resolved_root,
                latest_step=latest_step,
                latest_state=latest,
            )
        )
    return tuple(roots)


def _copy_resume_state(source: Path, destination_root: Path) -> Path:
    if source.is_symlink() or not source.is_dir():
        raise ValueError("resume source must be a regular state directory")
    for path in source.rglob("*"):
        if path.is_symlink() or (not path.is_file() and not path.is_dir()):
            raise ValueError("resume state contains an unsafe entry")
    if destination_root.is_symlink() or not destination_root.is_dir():
        raise ValueError("resume destination root must be a regular directory")
    destination = destination_root / source.name
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("resume state destination already exists")
    temporary = destination_root / f".{source.name}.tmp-resume"
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError("resume temporary destination already exists")
    temporary.mkdir(mode=0o750)
    shutil.copytree(source, temporary, copy_function=shutil.copy2, dirs_exist_ok=True)
    for path in sorted(temporary.rglob("*"), reverse=True):
        if path.is_file():
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    directory_descriptor = os.open(temporary, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    os.rename(temporary, destination)
    parent_descriptor = os.open(destination_root, os.O_RDONLY)
    try:
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)
    return destination


def _prepare_resume_state(
    context: StageContext,
    *,
    guard: _ResourceGuard,
    candidate_id: str,
    destination_root: Path,
    state_upper_bound_bytes: int,
    targeted_binding: TargetedSamplingBinding | None = None,
) -> Path | None:
    """Consolidate, copy, checkpoint, then retire cross-attempt durable states."""

    if destination_root.is_symlink() or not destination_root.is_dir():
        raise ValueError("resume destination root is unsafe")
    try:
        destination_relative = destination_root.relative_to(context.attempt_directory)
    except ValueError as error:
        raise ValueError("resume destination escapes its current stage attempt") from error
    cursor = context.attempt_directory
    for part in destination_relative.parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError("resume destination traverses a symlink")
    resolved_attempt = context.attempt_directory.resolve(strict=True)
    resolved_destination = destination_root.resolve(strict=True)
    if not resolved_destination.is_relative_to(resolved_attempt):
        raise ValueError("resume destination escapes its current stage attempt")

    prior_roots = _prior_training_state_roots(context, candidate_id)
    for item in prior_roots:
        if item.latest_state is not None:
            ensure_targeted_sampling_binding(
                item.root,
                targeted_binding,
                create_if_missing=False,
            )
    source = _latest_resume_from_roots(prior_roots)
    if source is None:
        return None
    source_root = source.parent
    # Reduce older attempts first, but never remove the selected source before a
    # durable successor exists. Every root was fully verified during inventory.
    for item in prior_roots:
        if item.root != source_root and item.latest_state is not None:
            retire_superseded_training_states(
                item.root,
                candidate_id=candidate_id,
                successor_directory=source,
            )
    if latest_committed_training_state(source_root, candidate_id=candidate_id) != source:
        raise PipelineExecutionError("selected resume state changed during consolidation")

    guard.enforce_projected_write(
        context,
        reservation_bytes=state_upper_bound_bytes,
    )
    # Publish the tiny checksum-bound sidecar before the much larger state copy.
    # A crash can then leave only an ignorable binding-only attempt, never a
    # committed targeted state that subsequent resume validation cannot bind.
    ensure_targeted_sampling_binding(
        destination_root,
        targeted_binding,
        create_if_missing=targeted_binding is not None,
    )
    copied = _copy_resume_state(source, destination_root)
    verified_copy = latest_committed_training_state(
        destination_root,
        candidate_id=candidate_id,
    )
    if verified_copy != copied.resolve(strict=True):
        raise PipelineExecutionError("copied resume state failed durable verification")
    context.progress.checkpoint(
        checkpoint=copied.relative_to(context.run_directory).as_posix(),
        message=f"Candidate {candidate_id} cross-attempt resume state committed.",
    )
    retire_superseded_training_states(
        source_root,
        candidate_id=candidate_id,
        successor_directory=copied,
    )
    return copied


def _training_progress_callback(
    context: StageContext,
    candidate_id: str,
) -> Callable[[TrainingProgress], None]:
    def report(event: TrainingProgress) -> None:
        metric: ProgressMetric | None = None
        if event.validation_nll is not None:
            metric = ProgressMetric(name="validation_nll", value=event.validation_nll)
        elif event.event != "evaluation":
            metric = context.progress.snapshot().latest_metric
        context.progress.report(
            message=f"Candidate {candidate_id} {event.event}.",
            completed_units=event.step,
            total_units=event.total_steps,
            latest_metric=metric,
        )
        if event.checkpoint_name is None:
            return
        if event.event in {"durable_checkpoint", "stopped"}:
            checkpoint = (
                context.attempt_directory / "training-state" / candidate_id / event.checkpoint_name
            )
        elif event.event == "final_checkpoint":
            checkpoint = context.attempt_directory / event.checkpoint_name
        else:
            raise PipelineExecutionError("training progress checkpoint event is invalid")
        if checkpoint.is_symlink() or not checkpoint.is_dir():
            raise PipelineExecutionError("reported training checkpoint is missing or unsafe")
        checkpoint_relative = checkpoint.relative_to(context.run_directory).as_posix()
        context.progress.checkpoint(
            checkpoint=checkpoint_relative,
            message=f"Candidate {candidate_id} {event.event} committed.",
        )

    return report


def _run_training(
    context: StageContext,
    *,
    guard: _ResourceGuard,
    inputs: _ExecutionInputs,
    candidate_id: str,
    sampling_strategy: Literal[
        "uniform_control",
        "task_balanced",
        "task_class_balanced",
        "fault_continuation_focused",
        "hierarchical_task_label_balanced",
        "fault_boosted_hierarchical",
    ],
    model_config: TransformerConfig,
    training: RemediationTraining,
    train_examples: tuple[RemediationExample, ...],
    validation_examples: tuple[RemediationExample, ...],
    evaluation_callback: EvaluationCallback | None,
) -> tuple[CompactTrainingResult, tuple[ArtifactReference, ...]]:
    generation_caps = inputs.generation_caps
    tokenized_train = _tokenize_examples(
        train_examples,
        inputs.tokenizer,
        context_length=model_config.context_length,
        generation_caps=generation_caps,
    )
    tokenized_validation = _tokenize_examples(
        validation_examples,
        inputs.tokenizer,
        context_length=model_config.context_length,
        generation_caps=generation_caps,
    )
    train_inventory_sha256 = canonical_sha256(
        tuple((item.example_id, item.checksum_sha256) for item in train_examples)
    )
    validation_inventory_sha256 = canonical_sha256(
        tuple((item.example_id, item.checksum_sha256) for item in validation_examples)
    )
    sampling_metadata = (
        tuple(
            SamplingMetadataRecord(
                example_id=item.example_id,
                task_name=item.task_name,
                classification_label=item.classification_label,
                augmentation=item.augmentation,
            )
            for item in train_examples
        )
        if sampling_strategy
        in {
            "task_class_balanced",
            "fault_continuation_focused",
            "hierarchical_task_label_balanced",
            "fault_boosted_hierarchical",
        }
        else None
    )
    targeted_binding = (
        bind_targeted_sampling(
            candidate_id=candidate_id,
            training_config_sha256=canonical_sha256(
                training.model_dump(mode="json", round_trip=True)
            ),
            train_inventory_sha256=train_inventory_sha256,
            train_tokenized_sha256=tokenized_inventory_sha256(tokenized_train),
            sampling_metadata_inventory_sha256=sampling_metadata_inventory_sha256(
                sampling_metadata
            ),
            sampling_strategy=cast(
                Literal[
                    "task_class_balanced",
                    "fault_continuation_focused",
                    "hierarchical_task_label_balanced",
                    "fault_boosted_hierarchical",
                ],
                sampling_strategy,
            ),
        )
        if sampling_metadata is not None
        else None
    )
    state_root = context.attempt_directory / "training-state" / candidate_id
    state_root.mkdir(parents=True, mode=0o750)
    if state_root.is_symlink() or not state_root.is_dir():
        raise ValueError("durable training root is unsafe")
    state_upper_bound = durable_training_state_upper_bound_bytes(
        model_config,
        vocab_size=inputs.tokenizer.vocab_size,
    )
    checkpoint_upper_bound = selected_checkpoint_upper_bound_bytes(
        model_config,
        vocab_size=inputs.tokenizer.vocab_size,
    )
    resume_state = _prepare_resume_state(
        context,
        guard=guard,
        candidate_id=candidate_id,
        destination_root=state_root,
        state_upper_bound_bytes=state_upper_bound,
        targeted_binding=targeted_binding,
    )
    existing_state_count = 0 if resume_state is None else 1
    guard.enforce_projected_write(
        context,
        reservation_bytes=(
            (MAX_TRANSIENT_DURABLE_STATES - existing_state_count) * state_upper_bound
            + checkpoint_upper_bound
        ),
    )
    checkpoint_directory = context.attempt_directory / f"checkpoint-{candidate_id}"

    def stop_requested(_step: int) -> bool:
        return guard.stop_required(context)

    outcome: CompactTrainingOutcome = train_compact_model(
        candidate_id=candidate_id,
        sampling_strategy=sampling_strategy,
        model_config=model_config,
        training=training,
        vocab_size=inputs.tokenizer.vocab_size,
        tokenizer_manifest=inputs.tokenizer.manifest,
        train_examples=tokenized_train,
        validation_examples=tokenized_validation,
        train_inventory_sha256=train_inventory_sha256,
        validation_inventory_sha256=validation_inventory_sha256,
        output_directory=checkpoint_directory,
        durable_state_root=state_root,
        source_commit=context.source_commit,
        resume_state_directory=resume_state,
        evaluation_callback=evaluation_callback,
        progress_callback=_training_progress_callback(context, candidate_id),
        stop_requested=stop_requested,
        sampling_metadata=sampling_metadata,
    )
    result_path = context.attempt_directory / f"training-{candidate_id}.json"
    _write_contract(result_path, cast(ContractModel, outcome))
    if type(outcome) is CompactTrainingStopped:
        raise KeyboardInterrupt
    if type(outcome) is not CompactTrainingResult:
        raise TypeError("training returned an unsupported outcome contract")
    artifacts = (
        _artifact_reference(result_path, run_directory=context.run_directory),
        *(
            (
                _artifact_reference(
                    state_root / TARGETED_SAMPLING_BINDING_FILENAME,
                    run_directory=context.run_directory,
                ),
            )
            if targeted_binding is not None
            else ()
        ),
        *_directory_artifacts(
            checkpoint_directory,
            run_directory=context.run_directory,
        ),
    )
    return outcome, tuple(artifacts)


def _load_training_result(attempt: Path, candidate_id: str) -> CompactTrainingResult:
    result = _read_contract(
        attempt / f"training-{candidate_id}.json",
        CompactTrainingResult,
    )
    binding_path = attempt / "training-state" / candidate_id / TARGETED_SAMPLING_BINDING_FILENAME
    if result.sampling_strategy in {
        "task_class_balanced",
        "fault_continuation_focused",
        "hierarchical_task_label_balanced",
        "fault_boosted_hierarchical",
    }:
        binding = load_targeted_sampling_binding(binding_path.parent)
        if (
            binding.sampling_strategy != result.sampling_strategy
            or binding.candidate_id != result.candidate_id
            or binding.training_config_sha256 != result.training_config_sha256
            or binding.train_inventory_sha256 != result.train_inventory_sha256
            or binding.train_tokenized_sha256 != result.train_tokenized_sha256
        ):
            raise PipelineExecutionError(
                "targeted sampling binding differs from the completed training result"
            )
    elif binding_path.exists() or binding_path.is_symlink():
        raise PipelineExecutionError("historical training result has a targeted sampling binding")
    return result


def _load_candidate_checkpoint(
    attempt: Path,
    candidate_id: str,
    result: CompactTrainingResult,
    tokenizer: ProjectTokenizer,
) -> tuple[TransformerLM, CheckpointManifest, torch.device]:
    requested_device = torch.device(result.device.resolved)
    model, manifest = load_checkpoint(
        attempt / f"checkpoint-{candidate_id}",
        expected_manifest_sha256=result.checkpoint_manifest_sha256,
        expected_tokenizer_sha256=tokenizer.manifest.checksum_sha256,
        device=requested_device,
    )
    try:
        actual_device = next(model.parameters()).device
    except StopIteration:
        raise PipelineExecutionError("loaded checkpoint model has no parameters") from None
    if actual_device.type != requested_device.type or (
        requested_device.index is not None and actual_device.index != requested_device.index
    ):
        raise PipelineExecutionError("loaded checkpoint model differs from its resolved device")
    return model, manifest, actual_device


def _training_checkpoint_matches_result(
    result: CompactTrainingResult,
    checkpoint: CheckpointManifest,
    *,
    model_config: TransformerConfig,
    training: RemediationTraining,
    tokenizer_manifest_sha256: str,
    source_commit: str,
) -> bool:
    """Cross-bind one selected checkpoint to its exact training result and policy."""

    expected_model_sha256 = canonical_sha256(model_config.model_dump(mode="json", round_trip=True))
    expected_training_sha256 = canonical_sha256(training.model_dump(mode="json", round_trip=True))
    return (
        result.source_commit == source_commit
        and result.training_steps == training.steps
        and result.training_config_sha256 == expected_training_sha256
        and result.model_config_sha256 == expected_model_sha256
        and result.tokenizer_manifest_sha256 == tokenizer_manifest_sha256
        and checkpoint.checksum_sha256 == result.checkpoint_manifest_sha256
        and checkpoint.transformer_config == model_config
        and checkpoint.vocab_size == result.vocab_size
        and checkpoint.parameter_count == result.parameter_count
        and checkpoint.tokenizer_manifest_sha256 == tokenizer_manifest_sha256
        and checkpoint.source_commit == source_commit
        and checkpoint.seed == training.seed
        and checkpoint.training_steps == result.selected_step
        and checkpoint.initial_loss == result.initial_validation_nll
        and checkpoint.final_loss == result.selected_validation_nll
        and checkpoint.weights_sha256 == result.checkpoint_weights_sha256
        and checkpoint.weights_size_bytes == result.checkpoint_size_bytes
    )


def _load_stage_dataset(
    context: StageContext,
    config: PipelineConfig,
    stage: str,
    directory_name: str,
) -> SafeDevelopmentDataset:
    return load_safe_development_artifact(
        _upstream_attempt(context, config, stage) / directory_name
    )


def _with_training_seed(training: RemediationTraining, seed: int) -> RemediationTraining:
    payload = training.model_dump(mode="python", round_trip=True)
    payload["seed"] = seed
    return RemediationTraining.model_validate(payload)


def _v04_pilot_training_config(
    inputs: _ExecutionInputs,
    *,
    batch_size: int,
) -> RemediationTraining:
    return RemediationTraining(
        seed=inputs.v04.training.seed,
        device=inputs.v04.training.device,
        allow_cpu_fallback=inputs.v04.training.allow_cpu_fallback,
        steps=inputs.v04.pilot.steps,
        batch_size=batch_size,
        learning_rate=inputs.v04.training.learning_rate,
        weight_decay=inputs.v04.training.weight_decay,
        gradient_clip_norm=inputs.v04.training.gradient_clip_norm,
        evaluation_interval=inputs.v04.pilot.steps,
        durable_checkpoint_interval=inputs.v04.pilot.steps,
    )


def _verify_v04_pilot_training_evidence(
    pilot_attempt: Path,
    pilot: V04PilotReport,
    inputs: _ExecutionInputs,
    *,
    source_commit: str,
) -> None:
    """Reopen and bind every passing pilot result before main training."""

    if (
        pilot.candidate_id != inputs.v04.pilot.candidate_id
        or inputs.v04.longer_context_model.context_length != 1024
        or pilot.material_truncation_threshold
        != inputs.v04.variants.material_prompt_truncation_rate
    ):
        raise PipelineExecutionError("v0.4 pilot inventory differs from its frozen configuration")
    augmentation = inputs.v03.augmentation
    material = build_frozen_v03_iid_material(
        inputs.v03_dataset_config,
        source_commit=inputs.frozen_data_source_commit,
        train_template_families=tuple(augmentation.train_template_families),
        train_alias_families=tuple(augmentation.train_alias_families),
        renderer_variants_per_projection=augmentation.renderer_variants_per_projection,
        include_insufficient_evidence_views=augmentation.include_insufficient_evidence_views,
    )
    train_examples = tuple(
        item for item in material.dataset.examples if item.view is RemediationView.IID_TRAIN
    )
    validation_examples = tuple(
        item for item in material.dataset.examples if item.view is RemediationView.IID_VALIDATION
    )
    control_train = _tokenize_examples(
        train_examples,
        inputs.tokenizer,
        context_length=inputs.v02.model.context_length,
        generation_caps=inputs.generation_caps,
    )
    measured_truncation_rate = sum(item.prompt_truncated for item in control_train) / len(
        control_train
    )
    expected_activation = (
        pilot.prompt_truncation_rate >= pilot.material_truncation_threshold
        and measured_truncation_rate >= pilot.material_truncation_threshold
    )
    if (
        pilot.v03_train_prompt_truncation_rate != measured_truncation_rate
        or pilot.activated is not expected_activation
    ):
        raise PipelineExecutionError(
            "v0.4 pilot activation differs from recomputed frozen IID truncation evidence"
        )
    if not pilot.activated:
        return
    if tuple(item.batch_size for item in pilot.measurements) != tuple(inputs.v04.pilot.batch_sizes):
        raise PipelineExecutionError("v0.4 pilot batch inventory differs from configuration")
    longer_train = _tokenize_examples(
        train_examples,
        inputs.tokenizer,
        context_length=1024,
        generation_caps=inputs.generation_caps,
    )
    longer_validation = _tokenize_examples(
        validation_examples,
        inputs.tokenizer,
        context_length=1024,
        generation_caps=inputs.generation_caps,
    )
    pilot_train, pilot_train_tokenized = _longest_pilot_examples_per_task(
        train_examples,
        longer_train,
    )
    pilot_validation, pilot_validation_tokenized = _longest_pilot_examples_per_task(
        validation_examples,
        longer_validation,
    )
    expected_train_tokenized_sha256 = _tokenized_inventory_sha256(pilot_train_tokenized)
    expected_validation_tokenized_sha256 = _tokenized_inventory_sha256(pilot_validation_tokenized)
    expected_train_inventory_sha256 = canonical_sha256(
        tuple((item.example_id, item.checksum_sha256) for item in pilot_train)
    )
    expected_validation_inventory_sha256 = canonical_sha256(
        tuple((item.example_id, item.checksum_sha256) for item in pilot_validation)
    )
    train_lengths = tuple(len(item.token_ids) for item in longer_train)
    validation_lengths = tuple(len(item.token_ids) for item in longer_validation)
    expected_profile = {
        "train_example_count": len(train_examples),
        "validation_example_count": len(validation_examples),
        "pilot_train_example_count": len(pilot_train),
        "pilot_validation_example_count": len(pilot_validation),
        "train_length_inventory_sha256": _sequence_length_inventory_sha256(longer_train),
        "validation_length_inventory_sha256": _sequence_length_inventory_sha256(longer_validation),
        "maximum_train_sequence_tokens": max(train_lengths),
        "maximum_validation_sequence_tokens": max(validation_lengths),
        "mean_train_sequence_tokens": sum(train_lengths) / len(train_lengths),
        "mean_validation_sequence_tokens": sum(validation_lengths) / len(validation_lengths),
    }
    expected_model_sha256 = canonical_sha256(
        inputs.v04.longer_context_model.model_dump(mode="json", round_trip=True)
    )
    for measurement in pilot.measurements:
        candidate_id = f"{pilot.candidate_id}-b{measurement.batch_size}"
        result = _load_training_result(pilot_attempt, candidate_id)
        checkpoint = _read_contract(
            pilot_attempt / f"checkpoint-{candidate_id}" / "manifest.json",
            CheckpointManifest,
            maximum_bytes=1024 * 1024,
        )
        expected_training = _v04_pilot_training_config(
            inputs,
            batch_size=measurement.batch_size,
        )
        expected_training_sha256 = canonical_sha256(
            expected_training.model_dump(mode="json", round_trip=True)
        )
        if (
            result.candidate_id != candidate_id
            or result.sampling_strategy != "task_balanced"
            or result.training_steps != inputs.v04.pilot.steps
            or result.checksum_sha256 != measurement.training_result_sha256
            or result.training_config_sha256 != measurement.training_config_sha256
            or result.training_config_sha256 != expected_training_sha256
            or result.model_config_sha256 != measurement.model_config_sha256
            or result.model_config_sha256 != expected_model_sha256
            or result.source_commit != source_commit
            or result.train_inventory_sha256 != expected_train_inventory_sha256
            or result.validation_inventory_sha256 != expected_validation_inventory_sha256
            or result.train_tokenized_sha256 != measurement.train_tokenized_sha256
            or result.validation_tokenized_sha256 != measurement.validation_tokenized_sha256
            or result.train_tokenized_sha256 != expected_train_tokenized_sha256
            or result.validation_tokenized_sha256 != expected_validation_tokenized_sha256
            or result.tokenizer_manifest_sha256 != measurement.tokenizer_manifest_sha256
            or result.tokenizer_manifest_sha256 != inputs.tokenizer.manifest.checksum_sha256
            or result.checkpoint_manifest_sha256 != measurement.checkpoint_manifest_sha256
            or checkpoint.checksum_sha256 != result.checkpoint_manifest_sha256
            or checkpoint.transformer_config != inputs.v04.longer_context_model
            or checkpoint.tokenizer_manifest_sha256 != result.tokenizer_manifest_sha256
            or checkpoint.source_commit != result.source_commit
            or checkpoint.training_steps != result.selected_step
            or checkpoint.weights_sha256 != result.checkpoint_weights_sha256
            or checkpoint.weights_size_bytes != result.checkpoint_size_bytes
            or result.device != measurement.device
            or result.train_example_count != measurement.pilot_train_example_count
            or result.validation_example_count != measurement.pilot_validation_example_count
            or measurement.elapsed_seconds != result.elapsed_seconds
            or measurement.process_peak_rss_bytes != result.process_peak_rss_bytes
            or any(
                getattr(measurement, field) != expected
                for field, expected in expected_profile.items()
            )
            or not _pilot_exercises_global_maximum(
                pilot_train_tokenized,
                batch_size=measurement.batch_size,
                seed=expected_training.seed,
                steps=expected_training.steps,
            )
        ):
            raise PipelineExecutionError(
                "v0.4 pilot training evidence differs from its independently reopened result"
            )


def _evaluate_candidate_view(
    context: StageContext,
    *,
    guard: _ResourceGuard,
    inputs: _ExecutionInputs,
    config_sha256_value: str,
    dataset: SafeDevelopmentDataset,
    train_examples: tuple[RemediationExample, ...],
    evaluation_examples: tuple[RemediationExample, ...],
    view: RemediationView,
    model: TransformerLM,
    checkpoint_manifest: CheckpointManifest,
    device: torch.device,
    stem: str,
    confidence_transform: Callable[[float], float] | None = None,
) -> tuple[
    SemanticEvaluationReport,
    RemediationBaselineReport,
    tuple[DualPathCompactPrediction, ...],
    tuple[ArtifactReference, ...],
]:
    scoped = _subset_dataset(
        dataset,
        (*train_examples, *evaluation_examples),
        dataset_version=dataset.manifest.dataset_version,
    )
    tokenized_train = _tokenize_examples(
        train_examples,
        inputs.tokenizer,
        context_length=model.config.context_length,
        generation_caps=inputs.generation_caps,
    )
    tokenized_evaluation = _tokenize_examples(
        evaluation_examples,
        inputs.tokenizer,
        context_length=model.config.context_length,
        generation_caps=inputs.generation_caps,
    )
    _raise_if_stop(context, guard)
    context.progress.report(
        message="Baseline comparator evaluation started.",
        completed_units=0,
        total_units=1,
    )
    baseline = run_remediation_baselines(
        scoped,
        inputs.tokenizer,
        inputs.baseline_config,
        tokenized_train=tokenized_train,
        tokenized_validation=tokenized_evaluation,
        evaluation_view=view,
        stop_requested=lambda: guard.stop_required(context),
    )
    context.progress.report(
        message="Baseline comparator evaluation completed.",
        completed_units=1,
        total_units=1,
    )
    _raise_if_stop(context, guard)
    predictions = _guarded_decode_examples(
        context,
        guard=guard,
        model=model,
        tokenizer=inputs.tokenizer,
        examples=evaluation_examples,
        generation_caps=inputs.generation_caps,
        device=device,
        progress_message="Model evaluation decoding in progress.",
    )
    # No scientific view artifact is written until every baseline and model
    # prediction for the view has completed and the final stop boundary passes.
    _raise_if_stop(context, guard)
    baseline_artifact = _contract_artifact(context, f"{stem}-baselines.json", baseline)
    prediction_manifest, prediction_artifacts = _write_predictions(
        context,
        stem=f"{stem}-predictions",
        view=view,
        examples=evaluation_examples,
        predictions=predictions,
    )
    binding = DevelopmentArtifactBinding(
        source_commit=context.source_commit,
        config_sha256=config_sha256_value,
        dataset_manifest_sha256=scoped.manifest.checksum_sha256,
        tokenizer_manifest_sha256=inputs.tokenizer.manifest.checksum_sha256,
        output_contract_sha256=inputs.compact_contract_sha256,
        checkpoint_sha256=checkpoint_manifest.checksum_sha256,
        prediction_artifact_sha256=prediction_manifest.predictions_sha256,
        comparator_artifact_sha256=baseline.checksum_sha256,
    )
    evaluation = evaluate_semantic_predictions(
        view=view,
        examples=evaluation_examples,
        predictions=predictions,
        baseline_report=baseline,
        artifacts=binding,
        confidence_transform=confidence_transform,
    )
    evaluation_artifact = _contract_artifact(
        context,
        f"{stem}-semantic-evaluation.json",
        evaluation,
    )
    return (
        evaluation,
        baseline,
        predictions,
        (baseline_artifact, *prediction_artifacts, evaluation_artifact),
    )


def _stage_completion_outcome(
    run_directory: Path,
    record_path: str,
) -> ArtifactReference:
    attempt = run_directory / record_path
    marker_path = attempt.parent / "completed.json"
    if marker_path.is_symlink() or not marker_path.is_file():
        raise PipelineExecutionError("terminal review cannot resolve stage completion marker")
    raw = _strict_json(marker_path.read_bytes())
    if type(raw) is not dict or type(raw.get("outcome")) is not dict:
        raise PipelineExecutionError("stage completion marker is malformed")
    return ArtifactReference.model_validate(raw["outcome"], strict=True)


class _PipelineRuntime:
    def __init__(
        self,
        *,
        project_root: Path,
        config: PipelineConfig,
        source_commit: str,
        inputs: _ExecutionInputs,
    ) -> None:
        self.project_root = project_root
        self.config = config
        self.source_commit = source_commit
        self.inputs = inputs
        self.guard = _ResourceGuard(config)

    def _start(self, context: StageContext) -> str:
        if context.project_root.resolve(strict=True) != self.project_root:
            raise PipelineExecutionError("stage project root differs from its frozen runtime")
        if context.source_commit != self.source_commit:
            raise PipelineExecutionError("stage source commit differs from its frozen runtime")
        runner_commit = _verify_runner_source(
            self.project_root,
            source_commit=self.source_commit,
            run_root=self.config.run_root,
        )
        self.guard.enforce_start(context)
        return runner_commit

    def _finish(self, context: StageContext, outcome: StageOutcome) -> StageOutcome:
        self.guard.enforce_end(context)
        return outcome

    def preflight(self, context: StageContext) -> StageOutcome:
        runner_commit = self._start(context)
        verify_v02_prefix_reuse(self.project_root, self.config)
        draft = ExecutionPreflightReport.model_construct(
            runner_source_commit=runner_commit,
            runner_worktree_clean=True,
            pipeline_config_sha256=config_sha256(self.config),
            v02_config_sha256=config_sha256(self.inputs.v02),
            v03_config_sha256=config_sha256(self.inputs.v03),
            v04_config_sha256=config_sha256(self.inputs.v04),
            frozen_data_source_commit=self.inputs.frozen_data_source_commit,
            tokenizer_manifest_sha256=self.inputs.tokenizer.manifest.checksum_sha256,
            compact_contract_sha256=self.inputs.compact_contract_sha256,
            v02_inventory_report_sha256=self.inputs.frozen_v02_inventory.checksum_sha256,
            v03_counterfactual_cap_report_sha256=(
                self.inputs.frozen_v03_counterfactual_cap.checksum_sha256
            ),
            checksum_sha256="0" * 64,
        )
        report = _bound_model(draft, ExecutionPreflightReport)
        artifact = _contract_artifact(context, "preflight.json", report)
        return self._finish(
            context,
            _stage_outcome(
                "Development-only provenance and frozen-input preflight passed.",
                artifacts=(artifact,),
            ),
        )

    def v02_inventory_and_caps(self, context: StageContext) -> StageOutcome:
        self._start(context)
        if self.config.reuse_v02_prefix is not None:
            return self._finish(
                context, _v02_reuse_outcome(context, self.config, "v02_inventory_and_caps")
            )
        dataset = build_safe_development_dataset(
            self.inputs.v02_dataset_config,
            source_commit=self.inputs.frozen_data_source_commit,
            views=(RemediationView.IID_TRAIN, RemediationView.IID_VALIDATION),
        )
        dataset_directory = context.attempt_directory / "dataset-v02"
        write_safe_development_artifact(dataset, dataset_directory)
        measured = measure_compact_inventory(
            dataset,
            self.inputs.tokenizer,
            self.inputs.v02.inventory,
        )
        if measured != self.inputs.frozen_v02_inventory:
            raise PipelineExecutionError("v0.2 inventory did not reproduce its frozen report")
        report_artifact = _contract_artifact(context, "v02-inventory.json", measured)
        artifacts = (
            report_artifact,
            *_directory_artifacts(dataset_directory, run_directory=context.run_directory),
        )
        return self._finish(
            context,
            _stage_outcome(
                "Frozen v0.2 IID inventory and compact generation caps reproduced exactly.",
                artifacts=artifacts,
                metrics=(
                    StageMetric(
                        name="prompt_truncation_rate",
                        value=float(measured.prompt_truncation_rate),
                        unit="ratio",
                    ),
                ),
            ),
        )

    def v02_smoke(self, context: StageContext) -> StageOutcome:
        self._start(context)
        if self.config.reuse_v02_prefix is not None:
            return self._finish(context, _v02_reuse_outcome(context, self.config, "v02_smoke"))
        dataset = _load_stage_dataset(context, self.config, "v02_inventory_and_caps", "dataset-v02")
        train, validation = _smoke_examples(dataset)
        result, artifacts = _run_training(
            context,
            guard=self.guard,
            inputs=self.inputs,
            candidate_id="v02-smoke",
            sampling_strategy="uniform_control",
            model_config=_smoke_model_config(self.inputs.v02.model),
            training=_smoke_training_config(self.inputs.v02.training),
            train_examples=train,
            validation_examples=validation,
            evaluation_callback=None,
        )
        _load_candidate_checkpoint(
            context.attempt_directory,
            "v02-smoke",
            result,
            self.inputs.tokenizer,
        )
        return self._finish(
            context,
            _stage_outcome(
                "Two-step CPU v0.2 smoke training and safe checkpoint reload completed.",
                artifacts=artifacts,
                metrics=(
                    StageMetric(
                        name="final_training_nll",
                        value=float(result.final_training_nll),
                        unit="nll",
                    ),
                ),
            ),
        )

    def v02_development_training(self, context: StageContext) -> StageOutcome:
        self._start(context)
        if self.config.reuse_v02_prefix is not None:
            return self._finish(
                context, _v02_reuse_outcome(context, self.config, "v02_development_training")
            )
        dataset = _load_stage_dataset(context, self.config, "v02_inventory_and_caps", "dataset-v02")
        train = tuple(item for item in dataset.examples if item.view is RemediationView.IID_TRAIN)
        validation = tuple(
            item for item in dataset.examples if item.view is RemediationView.IID_VALIDATION
        )

        def evaluation_callback(model: TransformerLM, _step: int, _nll: float) -> float:
            predictions = _guarded_decode_examples(
                context,
                guard=self.guard,
                model=model,
                tokenizer=self.inputs.tokenizer,
                examples=validation,
                generation_caps=self.inputs.generation_caps,
                device=next(model.parameters()).device,
                progress_message="v0.2 checkpoint-selection decoding in progress.",
            )
            return _free_running_structural_failure_score(predictions)

        result, artifacts = _run_training(
            context,
            guard=self.guard,
            inputs=self.inputs,
            candidate_id="v02-development",
            sampling_strategy="uniform_control",
            model_config=self.inputs.v02.model,
            training=self.inputs.v02.training,
            train_examples=train,
            validation_examples=validation,
            evaluation_callback=evaluation_callback,
        )
        return self._finish(
            context,
            _stage_outcome(
                "v0.2 compact-target development control training completed.",
                artifacts=artifacts,
                metrics=(
                    StageMetric(
                        name="selected_validation_nll",
                        value=float(result.selected_validation_nll),
                        unit="nll",
                    ),
                ),
            ),
        )

    def v02_development_gate(self, context: StageContext) -> StageOutcome:
        self._start(context)
        if self.config.reuse_v02_prefix is not None:
            return self._finish(
                context, _v02_reuse_outcome(context, self.config, "v02_development_gate")
            )
        dataset = _load_stage_dataset(context, self.config, "v02_inventory_and_caps", "dataset-v02")
        training_attempt = _upstream_attempt(context, self.config, "v02_development_training")
        result = _load_training_result(training_attempt, "v02-development")
        model, checkpoint_manifest, device = _load_candidate_checkpoint(
            training_attempt,
            "v02-development",
            result,
            self.inputs.tokenizer,
        )
        examples = tuple(
            item for item in dataset.examples if item.view is RemediationView.IID_VALIDATION
        )
        predictions = _guarded_decode_examples(
            context,
            guard=self.guard,
            model=model,
            tokenizer=self.inputs.tokenizer,
            examples=examples,
            generation_caps=self.inputs.generation_caps,
            device=device,
            progress_message="v0.2 behavioral evaluation decoding in progress.",
        )
        prediction_manifest, prediction_artifacts = _write_predictions(
            context,
            stem="v02-validation-predictions",
            view=RemediationView.IID_VALIDATION,
            examples=examples,
            predictions=predictions,
        )
        constrained = tuple(item.constrained for item in predictions)
        unconstrained = tuple(item.unconstrained for item in predictions)
        count = len(predictions)
        constrained_exact, constrained_latency = _exact_semantic_rate_and_mean_latency(
            examples,
            predictions,
            path_name="constrained",
        )
        unconstrained_exact, unconstrained_latency = _exact_semantic_rate_and_mean_latency(
            examples,
            predictions,
            path_name="unconstrained",
        )
        inventory = self.inputs.frozen_v02_inventory
        draft = V02DevelopmentGateReport.model_construct(
            inventory_report_sha256=inventory.checksum_sha256,
            prediction_manifest_sha256=prediction_manifest.checksum_sha256,
            training_result_sha256=result.checksum_sha256,
            checkpoint_manifest_sha256=checkpoint_manifest.checksum_sha256,
            checkpoint_weights_sha256=result.checkpoint_weights_sha256,
            example_count=count,
            constrained_parse_rate=sum(item.compact_parse_success for item in constrained) / count,
            constrained_schema_validity_rate=sum(item.schema_valid for item in constrained) / count,
            constrained_exact_semantic_match_rate=constrained_exact,
            constrained_mean_latency_seconds=constrained_latency,
            unconstrained_parse_rate=sum(item.compact_parse_success for item in unconstrained)
            / count,
            unconstrained_schema_validity_rate=sum(item.schema_valid for item in unconstrained)
            / count,
            unconstrained_exact_semantic_match_rate=unconstrained_exact,
            unconstrained_mean_latency_seconds=unconstrained_latency,
            generation_cap_exhaustion_rate=(
                sum(item.generation_cap_exhausted for item in constrained) / count
            ),
            process_peak_rss_bytes=result.process_peak_rss_bytes,
            mps_peak_current_allocated_bytes=result.mps_peak_current_allocated_bytes,
            mps_peak_driver_allocated_bytes=result.mps_peak_driver_allocated_bytes,
            checkpoint_size_bytes=result.checkpoint_size_bytes,
            inventory_example_count=inventory.example_count,
            prompt_truncation_count=inventory.prompt_truncation_count,
            prompt_truncation_rate=inventory.prompt_truncation_rate,
            target_fit_rate=inventory.target_fit_rate,
            round_trip_rate=inventory.round_trip_rate,
            reachability_rate=inventory.reachability_rate,
            task_footer_retained_rate=inventory.task_footer_retained_rate,
            cap_exhaustion_target_rate=inventory.cap_exhaustion_target_rate,
            advancement_allowed=(
                all(item.compact_parse_success and item.schema_valid for item in constrained)
                and sum(item.generation_cap_exhausted for item in constrained) / count
                <= V02_MAXIMUM_CAP_EXHAUSTION_RATE
                and inventory.prompt_truncation_count == V02_FROZEN_PROMPT_TRUNCATION_COUNT
                and inventory.prompt_truncation_rate
                == V02_FROZEN_PROMPT_TRUNCATION_COUNT / V01_PROMPT_TRUNCATION_EXAMPLE_COUNT
                and inventory.target_fit_rate == 1.0
                and inventory.round_trip_rate == 1.0
                and inventory.reachability_rate == 1.0
                and inventory.task_footer_retained_rate == 1.0
                and inventory.cap_exhaustion_target_rate == 0.0
            ),
            checksum_sha256="0" * 64,
        )
        report = _bound_model(draft, V02DevelopmentGateReport)
        artifact = _contract_artifact(context, "v02-development-gate.json", report)
        warning = (
            "D-073 records the 689/882 to 668/882 truncation change as modest; "
            "the v0.4 context pilot remains eligible pending the v0.3 materiality check."
            if report.advancement_allowed
            else "The v0.2 structural or exact D-073 reproduction gate did not pass."
        )
        return self._finish(
            context,
            _stage_outcome(
                "v0.2 development gate evaluated against the frozen structural contract.",
                advancement_allowed=bool(report.advancement_allowed),
                artifacts=(*prediction_artifacts, artifact),
                warnings=(warning,),
            ),
        )

    def v03_data_audit(self, context: StageContext) -> StageOutcome:
        self._start(context)
        augmentation = self.inputs.v03.augmentation
        material = build_frozen_v03_iid_material(
            self.inputs.v03_dataset_config,
            source_commit=self.inputs.frozen_data_source_commit,
            train_template_families=tuple(augmentation.train_template_families),
            train_alias_families=tuple(augmentation.train_alias_families),
            renderer_variants_per_projection=augmentation.renderer_variants_per_projection,
            include_insufficient_evidence_views=augmentation.include_insufficient_evidence_views,
        )
        dataset = material.dataset
        structured_fingerprints = material.structured_fingerprints
        iid_structured_separation = _task_scoped_structured_separation(
            structured_fingerprints,
            views=(RemediationView.IID_TRAIN, RemediationView.IID_VALIDATION),
        )
        dataset_directory = context.attempt_directory / "dataset-v03"
        write_safe_development_artifact(dataset, dataset_directory)
        audit = audit_safe_development_dataset(dataset)
        raw_cap = measure_counterfactual_cap_extension(
            material.raw_dataset,
            self.inputs.tokenizer,
            self.inputs.v02.inventory,
            self.inputs.frozen_v02_inventory,
        )
        deduplicated_cap = measure_counterfactual_cap_extension(
            dataset,
            self.inputs.tokenizer,
            self.inputs.v02.inventory,
            self.inputs.frozen_v02_inventory,
        )
        cap_compatibility = _v03_cap_compatibility_report(
            material,
            frozen_cap=self.inputs.frozen_v03_counterfactual_cap,
            raw_cap=raw_cap,
            deduplicated_cap=deduplicated_cap,
        )
        selection = build_semantic_selection_manifest(dataset, self.inputs.v03)
        calibration_selection = (
            build_calibration_selection_manifest(dataset, self.inputs.v03, selection)
            if isinstance(self.inputs.v03, V03Config)
            and self.inputs.v03.targeted_policy is not None
            else None
        )
        artifacts = (
            *_directory_artifacts(dataset_directory, run_directory=context.run_directory),
            _contract_artifact(context, "v03-development-audit.json", audit),
            _contract_artifact(
                context,
                "v03-iid-structured-separation.json",
                iid_structured_separation,
            ),
            _contract_artifact(context, "v03-counterfactual-cap.json", raw_cap),
            _contract_artifact(
                context,
                "v03-deduplicated-counterfactual-cap.json",
                deduplicated_cap,
            ),
            _contract_artifact(
                context,
                "v03-counterfactual-cap-compatibility.json",
                cap_compatibility,
            ),
            _contract_artifact(context, "v03-semantic-selection.json", selection),
            *(
                (
                    _contract_artifact(
                        context,
                        "v03-calibration-selection.json",
                        calibration_selection,
                    ),
                )
                if calibration_selection is not None
                else ()
            ),
        )
        return self._finish(
            context,
            _stage_outcome(
                "v0.3 IID data, augmentation audit, cap extension, and fixed "
                "48-row selection froze.",
                advancement_allowed=bool(
                    audit.passed and iid_structured_separation.passed and cap_compatibility.passed
                ),
                artifacts=artifacts,
                metrics=(
                    StageMetric(
                        name="semantic_selection_examples",
                        value=float(selection.selected_example_count),
                        unit="examples",
                    ),
                    *(
                        (
                            StageMetric(
                                name="calibration_selection_examples",
                                value=float(calibration_selection.selected_example_count),
                                unit="examples",
                            ),
                        )
                        if calibration_selection is not None
                        else ()
                    ),
                    StageMetric(
                        name="deduplicated_exact_prompt_rows",
                        value=float(cap_compatibility.removed_example_count),
                        unit="examples",
                    ),
                ),
            ),
        )

    def v03_smoke(self, context: StageContext) -> StageOutcome:
        self._start(context)
        dataset = _load_stage_dataset(context, self.config, "v03_data_audit", "dataset-v03")
        train, validation = _smoke_examples(dataset)
        result, artifacts = _run_training(
            context,
            guard=self.guard,
            inputs=self.inputs,
            candidate_id="v03-smoke",
            sampling_strategy="task_balanced",
            model_config=_smoke_model_config(self.inputs.v02.model),
            training=_smoke_training_config(self.inputs.v03.training),
            train_examples=train,
            validation_examples=validation,
            evaluation_callback=None,
        )
        _load_candidate_checkpoint(
            context.attempt_directory,
            "v03-smoke",
            result,
            self.inputs.tokenizer,
        )
        return self._finish(
            context,
            _stage_outcome(
                "Two-step CPU v0.3 task-balanced smoke and checkpoint reload passed.",
                artifacts=artifacts,
            ),
        )

    def v03_candidate_training(self, context: StageContext) -> StageOutcome:
        self._start(context)
        audit_attempt = _upstream_attempt(context, self.config, "v03_data_audit")
        dataset = load_safe_development_artifact(audit_attempt / "dataset-v03")
        selection = _read_contract(
            audit_attempt / "v03-semantic-selection.json",
            SemanticSelectionManifest,
        )
        selected_examples = resolve_semantic_selection_examples(
            dataset,
            selection,
            self.inputs.v03,
        )
        train_examples = tuple(
            item for item in dataset.examples if item.view is RemediationView.IID_TRAIN
        )
        artifacts: list[ArtifactReference] = []
        for candidate in self.inputs.v03.candidates:
            training = _with_training_seed(self.inputs.v03.training, candidate.seed)

            def evaluation_callback(
                model: TransformerLM,
                _step: int,
                _nll: float,
                *,
                selected: tuple[RemediationExample, ...] = selected_examples,
            ) -> float:
                device = next(model.parameters()).device
                predictions = _guarded_decode_examples(
                    context,
                    guard=self.guard,
                    model=model,
                    tokenizer=self.inputs.tokenizer,
                    examples=selected,
                    generation_caps=self.inputs.generation_caps,
                    device=device,
                    progress_message="v0.3 checkpoint-selection decoding in progress.",
                )
                composite = semantic_composite_score(selected, predictions)
                return _semantic_checkpoint_selection_score(
                    self.inputs.v03.selection,
                    composite=composite,
                )

            _result, candidate_artifacts = _run_training(
                context,
                guard=self.guard,
                inputs=self.inputs,
                candidate_id=candidate.candidate_id,
                sampling_strategy=candidate.sampling,
                model_config=self.inputs.v02.model,
                training=training,
                train_examples=train_examples,
                validation_examples=selected_examples,
                evaluation_callback=evaluation_callback,
            )
            artifacts.extend(candidate_artifacts)
        return self._finish(
            context,
            _stage_outcome(
                "Both preregistered v0.3 candidates completed development-only training.",
                artifacts=tuple(artifacts),
                metrics=(
                    StageMetric(
                        name="candidate_count",
                        value=float(len(self.inputs.v03.candidates)),
                        unit="candidates",
                    ),
                ),
            ),
        )

    def v03_development_evaluation(self, context: StageContext) -> StageOutcome:
        self._start(context)
        audit_attempt = _upstream_attempt(context, self.config, "v03_data_audit")
        training_attempt = _upstream_attempt(context, self.config, "v03_candidate_training")
        dataset = load_safe_development_artifact(audit_attempt / "dataset-v03")
        selection = _read_contract(
            audit_attempt / "v03-semantic-selection.json",
            SemanticSelectionManifest,
        )
        selected_examples = resolve_semantic_selection_examples(
            dataset,
            selection,
            self.inputs.v03,
        )
        train_examples = tuple(
            item for item in dataset.examples if item.view is RemediationView.IID_TRAIN
        )
        scores: list[CandidateScore] = []
        artifacts: list[ArtifactReference] = []
        for candidate in self.inputs.v03.candidates:
            result = _load_training_result(training_attempt, candidate.candidate_id)
            model, manifest, device = _load_candidate_checkpoint(
                training_attempt,
                candidate.candidate_id,
                result,
                self.inputs.tokenizer,
            )
            evaluation, _baseline, _predictions, evaluation_artifacts = _evaluate_candidate_view(
                context,
                guard=self.guard,
                inputs=self.inputs,
                config_sha256_value=config_sha256(self.inputs.v03),
                dataset=dataset,
                train_examples=train_examples,
                evaluation_examples=selected_examples,
                view=RemediationView.IID_VALIDATION,
                model=model,
                checkpoint_manifest=manifest,
                device=device,
                stem=candidate.candidate_id,
            )
            artifacts.extend(evaluation_artifacts)
            scores.append(
                CandidateScore(
                    candidate_id=candidate.candidate_id,
                    checkpoint_manifest_sha256=manifest.checksum_sha256,
                    semantic_composite=_semantic_report_composite(evaluation),
                    selected_validation_nll=float(result.selected_validation_nll),
                    selected_step=result.selected_step,
                    evaluation_report_sha256=evaluation.checksum_sha256,
                )
            )
        ordered_scores = tuple(sorted(scores, key=lambda item: item.candidate_id))
        selected = _select_v03_candidate(self.inputs.v03.selection, ordered_scores)
        draft = CandidateSelectionReport.model_construct(
            selection_manifest_sha256=selection.checksum_sha256,
            candidates=ordered_scores,
            selected_candidate_id=selected.candidate_id,
            selected_checkpoint_manifest_sha256=selected.checkpoint_manifest_sha256,
            checksum_sha256="0" * 64,
        )
        report = _bound_model(draft, CandidateSelectionReport)
        artifacts.append(_contract_artifact(context, "v03-candidate-selection.json", report))
        full_iid_validation = tuple(
            item for item in dataset.examples if item.view is RemediationView.IID_VALIDATION
        )
        if not full_iid_validation:
            raise PipelineExecutionError("v0.3 full IID validation view is empty")
        selected_result = _load_training_result(
            training_attempt,
            report.selected_candidate_id,
        )
        selected_model, selected_manifest, selected_device = _load_candidate_checkpoint(
            training_attempt,
            report.selected_candidate_id,
            selected_result,
            self.inputs.tokenizer,
        )
        gate_examples = full_iid_validation
        calibration_report = None
        if isinstance(self.inputs.v03, V03Config) and self.inputs.v03.targeted_policy is not None:
            calibration_manifest = _read_contract(
                audit_attempt / "v03-calibration-selection.json", CalibrationSelectionManifest
            )
            calibration_examples = resolve_calibration_selection_examples(
                dataset, self.inputs.v03, selection, calibration_manifest
            )
            calibration_predictions = _guarded_decode_examples(
                context,
                guard=self.guard,
                model=selected_model,
                tokenizer=self.inputs.tokenizer,
                examples=calibration_examples,
                generation_caps=self.inputs.generation_caps,
                device=selected_device,
                progress_message="v0.3 calibration decoding in progress.",
            )
            calibration_prediction_manifest, calibration_prediction_artifacts = _write_predictions(
                context,
                stem="v03-calibration-predictions",
                view=RemediationView.IID_VALIDATION,
                examples=calibration_examples,
                predictions=calibration_predictions,
            )
            artifacts.extend(calibration_prediction_artifacts)
            observations = _calibration_observations_by_identity(
                calibration_examples,
                calibration_predictions,
            )
            calibration_report = fit_temperature(
                observations,
                calibration_selection_manifest_sha256=calibration_manifest.checksum_sha256,
                calibration_prediction_manifest_sha256=(
                    calibration_prediction_manifest.checksum_sha256
                ),
                calibration_predictions_sha256=calibration_prediction_manifest.predictions_sha256,
                selected_checkpoint_manifest_sha256=selected_manifest.checksum_sha256,
            )
            artifacts.append(
                _contract_artifact(context, "v03-temperature-calibration.json", calibration_report)
            )
            excluded = {item.example_id for item in (*selected_examples, *calibration_examples)}
            gate_examples = tuple(
                item for item in full_iid_validation if item.example_id not in excluded
            )
            if len(gate_examples) != 427:
                raise PipelineExecutionError("targeted v0.3 gate partition is not 48+56+427")
        full_artifacts: tuple[ArtifactReference, ...]
        if calibration_report is not None:
            raw_evaluation, raw_baseline, raw_predictions, raw_artifacts = _evaluate_candidate_view(
                context,
                guard=self.guard,
                inputs=self.inputs,
                config_sha256_value=config_sha256(self.inputs.v03),
                dataset=dataset,
                train_examples=train_examples,
                evaluation_examples=gate_examples,
                view=RemediationView.IID_VALIDATION,
                model=selected_model,
                checkpoint_manifest=selected_manifest,
                device=selected_device,
                stem="v03-selected-gate-iid-raw",
            )
            artifacts.extend(raw_artifacts)
            full_evaluation = evaluate_semantic_predictions(
                view=RemediationView.IID_VALIDATION,
                examples=gate_examples,
                predictions=raw_predictions,
                baseline_report=raw_baseline,
                artifacts=raw_evaluation.view_metrics.artifacts,
                confidence_transform=lambda value: apply_temperature(
                    value, calibration_report.selected_temperature
                ),
            )
            full_artifacts = (
                _contract_artifact(
                    context,
                    "v03-selected-gate-iid-semantic-evaluation.json",
                    full_evaluation,
                ),
            )
        else:
            full_evaluation, _full_baseline, _full_predictions, full_artifacts = (
                _evaluate_candidate_view(
                    context,
                    guard=self.guard,
                    inputs=self.inputs,
                    config_sha256_value=config_sha256(self.inputs.v03),
                    dataset=dataset,
                    train_examples=train_examples,
                    evaluation_examples=gate_examples,
                    view=RemediationView.IID_VALIDATION,
                    model=selected_model,
                    checkpoint_manifest=selected_manifest,
                    device=selected_device,
                    stem="v03-selected-full-iid",
                )
            )
        artifacts.extend(full_artifacts)
        return self._finish(
            context,
            _stage_outcome(
                "v0.3 selected raw, calibrated on a disjoint subset, then gated on IID remainder.",
                artifacts=tuple(artifacts),
                metrics=(
                    StageMetric(
                        name="full_iid_validation_examples",
                        value=float(len(gate_examples)),
                        unit="examples",
                    ),
                    StageMetric(
                        name="selected_semantic_composite",
                        value=float(selected.semantic_composite),
                        unit="ratio",
                    ),
                ),
            ),
        )

    def _reconstruct_v03_gate_evidence(
        self,
        context: StageContext,
    ) -> tuple[V03AcceptanceResult, TargetedV03GateBinding | None]:
        """Reopen and reconstruct v0.3 evidence without publishing stage state."""

        iid_dataset = _load_stage_dataset(
            context,
            self.config,
            "v03_data_audit",
            "dataset-v03",
        )
        full_iid_validation_count = sum(
            item.view is RemediationView.IID_VALIDATION for item in iid_dataset.examples
        )
        if full_iid_validation_count < 1:
            raise PipelineExecutionError("v0.3 full-IID validation inventory is empty")
        audit_attempt = _upstream_attempt(context, self.config, "v03_data_audit")
        frozen_selection = _read_contract(
            audit_attempt / "v03-semantic-selection.json",
            SemanticSelectionManifest,
        )
        selected_examples = resolve_semantic_selection_examples(
            iid_dataset,
            frozen_selection,
            self.inputs.v03,
        )
        train_examples = tuple(
            item for item in iid_dataset.examples if item.view is RemediationView.IID_TRAIN
        )
        selection_dataset = _subset_dataset(
            iid_dataset,
            (*train_examples, *selected_examples),
            dataset_version=iid_dataset.manifest.dataset_version,
        )
        tokenized_train = _tokenize_examples(
            train_examples,
            self.inputs.tokenizer,
            context_length=self.inputs.v02.model.context_length,
            generation_caps=self.inputs.generation_caps,
        )
        tokenized_selection = _tokenize_examples(
            selected_examples,
            self.inputs.tokenizer,
            context_length=self.inputs.v02.model.context_length,
            generation_caps=self.inputs.generation_caps,
        )
        expected_train_inventory_sha256 = canonical_sha256(
            tuple((item.example_id, item.checksum_sha256) for item in train_examples)
        )
        expected_selection_inventory_sha256 = canonical_sha256(
            tuple((item.example_id, item.checksum_sha256) for item in selected_examples)
        )
        expected_train_tokenized_sha256 = _tokenized_inventory_sha256(tokenized_train)
        expected_selection_tokenized_sha256 = _tokenized_inventory_sha256(tokenized_selection)
        expected_model_sha256 = canonical_sha256(
            self.inputs.v02.model.model_dump(mode="json", round_trip=True)
        )
        expected_config_sha256 = config_sha256(self.inputs.v03)
        evaluation_attempt = _upstream_attempt(context, self.config, "v03_development_evaluation")
        training_attempt = _upstream_attempt(context, self.config, "v03_candidate_training")
        selection = _read_contract(
            evaluation_attempt / "v03-candidate-selection.json",
            CandidateSelectionReport,
        )
        candidate_policies = {item.candidate_id: item for item in self.inputs.v03.candidates}
        if selection.selection_manifest_sha256 != frozen_selection.checksum_sha256 or tuple(
            item.candidate_id for item in selection.candidates
        ) != tuple(sorted(candidate_policies)):
            raise PipelineExecutionError(
                "v0.3 candidate selection differs from its frozen manifest/configuration"
            )
        for score in selection.candidates:
            policy = candidate_policies[score.candidate_id]
            result = _load_training_result(training_attempt, score.candidate_id)
            checkpoint = _read_contract(
                training_attempt / f"checkpoint-{score.candidate_id}" / "manifest.json",
                CheckpointManifest,
                maximum_bytes=1024 * 1024,
            )
            selection_evaluation = _read_contract(
                evaluation_attempt / f"{score.candidate_id}-semantic-evaluation.json",
                SemanticEvaluationReport,
            )
            expected_training = _with_training_seed(self.inputs.v03.training, policy.seed)
            expected_training_sha256 = canonical_sha256(
                expected_training.model_dump(mode="json", round_trip=True)
            )
            selection_artifacts = selection_evaluation.view_metrics.artifacts
            selection_composite = _semantic_report_composite(selection_evaluation)
            if (
                result.candidate_id != score.candidate_id
                or result.sampling_strategy != policy.sampling
                or result.source_commit != context.source_commit
                or result.checkpoint_manifest_sha256 != score.checkpoint_manifest_sha256
                or checkpoint.checksum_sha256 != score.checkpoint_manifest_sha256
                or result.training_config_sha256 != expected_training_sha256
                or result.model_config_sha256 != expected_model_sha256
                or result.tokenizer_manifest_sha256
                != self.inputs.tokenizer.manifest.checksum_sha256
                or result.train_example_count != len(train_examples)
                or result.validation_example_count != len(selected_examples)
                or result.train_inventory_sha256 != expected_train_inventory_sha256
                or result.validation_inventory_sha256 != expected_selection_inventory_sha256
                or result.train_tokenized_sha256 != expected_train_tokenized_sha256
                or result.validation_tokenized_sha256 != expected_selection_tokenized_sha256
                or result.selected_validation_nll != score.selected_validation_nll
                or result.selected_step != score.selected_step
                or result.selected_score
                != _semantic_checkpoint_selection_score(
                    self.inputs.v03.selection,
                    composite=selection_composite,
                )
                or not _training_checkpoint_matches_result(
                    result,
                    checkpoint,
                    model_config=self.inputs.v02.model,
                    training=expected_training,
                    tokenizer_manifest_sha256=self.inputs.tokenizer.manifest.checksum_sha256,
                    source_commit=context.source_commit,
                )
                or selection_evaluation.evaluation_view is not RemediationView.IID_VALIDATION
                or selection_evaluation.view_metrics.view is not DevelopmentView.IID_VALIDATION
                or selection_evaluation.example_count != len(selected_examples)
                or selection_evaluation.view_metrics.sample_count != len(selected_examples)
                or selection_evaluation.checksum_sha256 != score.evaluation_report_sha256
                or selection_artifacts.source_commit != context.source_commit
                or selection_artifacts.config_sha256 != expected_config_sha256
                or selection_artifacts.dataset_manifest_sha256
                != selection_dataset.manifest.checksum_sha256
                or selection_artifacts.tokenizer_manifest_sha256
                != self.inputs.tokenizer.manifest.checksum_sha256
                or selection_artifacts.output_contract_sha256 != self.inputs.compact_contract_sha256
                or selection_artifacts.checkpoint_sha256 != score.checkpoint_manifest_sha256
                or selection_evaluation.predictions_sha256
                != selection_artifacts.prediction_artifact_sha256
                or selection_evaluation.baseline_report_sha256
                != selection_artifacts.comparator_artifact_sha256
                or selection_composite != score.semantic_composite
            ):
                raise PipelineExecutionError(
                    "v0.3 candidate ranking differs from immutable training/evaluation evidence"
                )
        verified_selected = _select_v03_candidate(
            self.inputs.v03.selection,
            selection.candidates,
        )
        if (
            verified_selected.candidate_id != selection.selected_candidate_id
            or verified_selected.checkpoint_manifest_sha256
            != selection.selected_checkpoint_manifest_sha256
        ):
            raise PipelineExecutionError("v0.3 candidate ranking changed during gate verification")
        targeted = (
            isinstance(self.inputs.v03, V03Config) and self.inputs.v03.targeted_policy is not None
        )
        calibration: TemperatureCalibrationReport | None = None
        calibration_selection: CalibrationSelectionManifest | None = None
        calibration_prediction_manifest: PredictionArtifactManifest | None = None
        calibration_predictions: tuple[DualPathCompactPrediction, ...] | None = None
        calibration_examples: tuple[RemediationExample, ...] = ()
        gate_examples = tuple(
            item for item in iid_dataset.examples if item.view is RemediationView.IID_VALIDATION
        )
        if targeted:
            calibration = _read_contract(
                evaluation_attempt / "v03-temperature-calibration.json",
                TemperatureCalibrationReport,
            )
            calibration_selection = _read_contract(
                audit_attempt / "v03-calibration-selection.json",
                CalibrationSelectionManifest,
            )
            calibration_examples = resolve_calibration_selection_examples(
                iid_dataset,
                self.inputs.v03,
                frozen_selection,
                calibration_selection,
            )
            calibration_prediction_manifest, calibration_predictions = _read_predictions(
                manifest_path=evaluation_attempt / "v03-calibration-predictions-manifest.json",
                predictions_path=evaluation_attempt / "v03-calibration-predictions.jsonl",
                view=RemediationView.IID_VALIDATION,
                examples=calibration_examples,
            )
            if (
                calibration.observation_count != 56
                or calibration.calibration_selection_manifest_sha256
                != calibration_selection.checksum_sha256
                or calibration.calibration_prediction_manifest_sha256
                != calibration_prediction_manifest.checksum_sha256
                or calibration.calibration_predictions_sha256
                != calibration_prediction_manifest.predictions_sha256
                or calibration.selected_checkpoint_manifest_sha256
                != selection.selected_checkpoint_manifest_sha256
            ):
                raise PipelineExecutionError(
                    "temperature calibration binding differs from its immutable evidence"
                )
            excluded = {item.example_id for item in (*selected_examples, *calibration_examples)}
            gate_examples = tuple(
                item
                for item in iid_dataset.examples
                if item.view is RemediationView.IID_VALIDATION and item.example_id not in excluded
            )
            if len(gate_examples) != 427:
                raise PipelineExecutionError("targeted v0.3 gate partition is not 48+56+427")
        evaluation = _read_contract(
            evaluation_attempt
            / (
                "v03-selected-gate-iid-semantic-evaluation.json"
                if targeted
                else "v03-selected-full-iid-semantic-evaluation.json"
            ),
            SemanticEvaluationReport,
        )
        full_artifacts = evaluation.view_metrics.artifacts
        if (
            evaluation.evaluation_view is not RemediationView.IID_VALIDATION
            or evaluation.view_metrics.view is not DevelopmentView.IID_VALIDATION
            or evaluation.example_count != (427 if targeted else full_iid_validation_count)
            or evaluation.view_metrics.sample_count
            != (427 if targeted else full_iid_validation_count)
            or evaluation.view_metrics.artifacts.dataset_manifest_sha256
            != (
                _subset_dataset(
                    iid_dataset,
                    (*train_examples, *gate_examples),
                    dataset_version=iid_dataset.manifest.dataset_version,
                ).manifest.checksum_sha256
                if targeted
                else iid_dataset.manifest.checksum_sha256
            )
            or full_artifacts.source_commit != context.source_commit
            or full_artifacts.config_sha256 != expected_config_sha256
            or full_artifacts.tokenizer_manifest_sha256
            != self.inputs.tokenizer.manifest.checksum_sha256
            or full_artifacts.output_contract_sha256 != self.inputs.compact_contract_sha256
            or full_artifacts.checkpoint_sha256 != selection.selected_checkpoint_manifest_sha256
            or evaluation.predictions_sha256 != full_artifacts.prediction_artifact_sha256
            or evaluation.baseline_report_sha256 != full_artifacts.comparator_artifact_sha256
        ):
            raise PipelineExecutionError(
                "v0.3 full-IID evaluation differs from the selected checkpoint"
            )
        raw_evaluation: SemanticEvaluationReport | None = None
        targeted_acceptance: V03AcceptanceResult | None = None
        targeted_binding: TargetedV03GateBinding | None = None
        if targeted:
            if (
                calibration is None
                or calibration_selection is None
                or calibration_prediction_manifest is None
                or calibration_predictions is None
            ):
                raise PipelineExecutionError("targeted calibration evidence disappeared")
            raw_evaluation = _read_contract(
                evaluation_attempt / "v03-selected-gate-iid-raw-semantic-evaluation.json",
                SemanticEvaluationReport,
            )
            raw_baseline = _read_contract(
                evaluation_attempt / "v03-selected-gate-iid-raw-baselines.json",
                RemediationBaselineReport,
            )
            raw_prediction_manifest, raw_predictions = _read_predictions(
                manifest_path=(
                    evaluation_attempt / "v03-selected-gate-iid-raw-predictions-manifest.json"
                ),
                predictions_path=(
                    evaluation_attempt / "v03-selected-gate-iid-raw-predictions.jsonl"
                ),
                view=RemediationView.IID_VALIDATION,
                examples=gate_examples,
            )
            gate_dataset = _subset_dataset(
                iid_dataset,
                (*train_examples, *gate_examples),
                dataset_version=iid_dataset.manifest.dataset_version,
            )
            reconstructed_binding = DevelopmentArtifactBinding(
                source_commit=context.source_commit,
                config_sha256=expected_config_sha256,
                dataset_manifest_sha256=gate_dataset.manifest.checksum_sha256,
                tokenizer_manifest_sha256=self.inputs.tokenizer.manifest.checksum_sha256,
                output_contract_sha256=self.inputs.compact_contract_sha256,
                checkpoint_sha256=selection.selected_checkpoint_manifest_sha256,
                prediction_artifact_sha256=raw_prediction_manifest.predictions_sha256,
                comparator_artifact_sha256=raw_baseline.checksum_sha256,
            )
            targeted_acceptance, targeted_binding = _reconstruct_targeted_v03_gate(
                selection=selection,
                calibration_selection=calibration_selection,
                calibration=calibration,
                calibration_examples=calibration_examples,
                calibration_prediction_manifest=calibration_prediction_manifest,
                calibration_predictions=calibration_predictions,
                gate_examples=gate_examples,
                raw_prediction_manifest=raw_prediction_manifest,
                raw_predictions=raw_predictions,
                raw_baseline=raw_baseline,
                expected_artifacts=reconstructed_binding,
                saved_raw_evaluation=raw_evaluation,
                saved_calibrated_evaluation=evaluation,
            )
        acceptance = (
            targeted_acceptance
            if targeted_acceptance is not None
            else evaluate_v03_acceptance(evaluation.view_metrics)
        )
        return acceptance, targeted_binding

    def v03_gate(self, context: StageContext) -> StageOutcome:
        self._start(context)
        acceptance, targeted_binding = self._reconstruct_v03_gate_evidence(context)
        artifact = _contract_artifact(context, "v03-acceptance.json", acceptance)
        gate_artifacts: tuple[ArtifactReference, ...] = (artifact,)
        if targeted_binding is not None:
            gate_artifacts = (
                artifact,
                _contract_artifact(context, "v03-targeted-gate-binding.json", targeted_binding),
            )
        return self._finish(
            context,
            _stage_outcome(
                "v0.3 IID semantic acceptance gate evaluated with named preregistered checks.",
                advancement_allowed=bool(acceptance.advancement_allowed),
                artifacts=gate_artifacts,
            ),
        )

    def v04_shadow_freeze(self, context: StageContext) -> StageOutcome:
        self._start(context)
        augmentation = self.inputs.v03.augmentation
        regenerated_material = build_frozen_v03_iid_material(
            self.inputs.v03_dataset_config,
            source_commit=self.inputs.frozen_data_source_commit,
            train_template_families=tuple(augmentation.train_template_families),
            train_alias_families=tuple(augmentation.train_alias_families),
            renderer_variants_per_projection=augmentation.renderer_variants_per_projection,
            include_insufficient_evidence_views=augmentation.include_insufficient_evidence_views,
        )
        regenerated_iid = regenerated_material.dataset
        iid_structured_fingerprints = regenerated_material.structured_fingerprints
        shadow, shadow_structured_fingerprints = (
            build_safe_development_dataset_with_structured_fingerprints(
                self.inputs.v03_dataset_config,
                source_commit=self.inputs.frozen_data_source_commit,
                views=SHADOW_VIEWS,
            )
        )
        shadow_directory = context.attempt_directory / "dataset-v04-shadow"
        write_safe_development_artifact(shadow, shadow_directory)
        shadow_audit = audit_safe_development_dataset(shadow)
        iid = _load_stage_dataset(context, self.config, "v03_data_audit", "dataset-v03")
        _require_exact_regenerated_iid(regenerated_iid, iid)
        structured_separation = _task_scoped_structured_separation(
            (*iid_structured_fingerprints, *shadow_structured_fingerprints),
            views=tuple(RemediationView),
        )
        separation = _development_separation_report(iid, shadow, structured_separation)
        artifacts = (
            *_directory_artifacts(shadow_directory, run_directory=context.run_directory),
            _contract_artifact(context, "v04-shadow-audit.json", shadow_audit),
            _contract_artifact(context, "v04-development-separation.json", separation),
        )
        passed = bool(shadow_audit.passed and separation.passed)
        return self._finish(
            context,
            _stage_outcome(
                "Six preregistered shadow views froze with IID separation evidence.",
                advancement_allowed=passed,
                artifacts=artifacts,
                metrics=(
                    StageMetric(
                        name="shadow_examples",
                        value=float(len(shadow.examples)),
                        unit="examples",
                    ),
                ),
            ),
        )

    def v04_pilot(self, context: StageContext) -> StageOutcome:
        self._start(context)
        iid = _load_stage_dataset(context, self.config, "v03_data_audit", "dataset-v03")
        train_examples = tuple(
            item for item in iid.examples if item.view is RemediationView.IID_TRAIN
        )
        validation_examples = tuple(
            item for item in iid.examples if item.view is RemediationView.IID_VALIDATION
        )
        tokenized = _tokenize_examples(
            train_examples,
            self.inputs.tokenizer,
            context_length=self.inputs.v02.model.context_length,
            generation_caps=self.inputs.generation_caps,
        )
        v03_train_truncation_rate = sum(item.prompt_truncated for item in tokenized) / len(
            tokenized
        )
        frozen_inventory = self.inputs.frozen_v02_inventory
        frozen_truncation_rate = (
            V02_FROZEN_PROMPT_TRUNCATION_COUNT / V01_PROMPT_TRUNCATION_EXAMPLE_COUNT
        )
        if (
            frozen_inventory.example_count != V01_PROMPT_TRUNCATION_EXAMPLE_COUNT
            or frozen_inventory.prompt_truncation_count != V02_FROZEN_PROMPT_TRUNCATION_COUNT
            or frozen_inventory.prompt_truncation_rate != frozen_truncation_rate
        ):
            raise PipelineExecutionError("v0.4 pilot cannot reproduce the D-073 activation")
        threshold = self.inputs.v04.variants.material_prompt_truncation_rate
        activated = frozen_truncation_rate >= threshold and v03_train_truncation_rate >= threshold
        if self.inputs.v04.training.device != "mps":
            raise PipelineExecutionError("v0.4 mandatory context pilot must request MPS")
        measurements: list[V04PilotMeasurement] = []
        artifacts: list[ArtifactReference] = []
        if activated:
            longer_train = _tokenize_examples(
                train_examples,
                self.inputs.tokenizer,
                context_length=self.inputs.v04.longer_context_model.context_length,
                generation_caps=self.inputs.generation_caps,
            )
            longer_validation = _tokenize_examples(
                validation_examples,
                self.inputs.tokenizer,
                context_length=self.inputs.v04.longer_context_model.context_length,
                generation_caps=self.inputs.generation_caps,
            )
            pilot_train, pilot_train_tokenized = _longest_pilot_examples_per_task(
                train_examples,
                longer_train,
            )
            pilot_validation, pilot_validation_tokenized = _longest_pilot_examples_per_task(
                validation_examples,
                longer_validation,
            )
            train_length_inventory_sha256 = _sequence_length_inventory_sha256(longer_train)
            validation_length_inventory_sha256 = _sequence_length_inventory_sha256(
                longer_validation
            )
            maximum_train_sequence_tokens = max(len(item.token_ids) for item in longer_train)
            maximum_validation_sequence_tokens = max(
                len(item.token_ids) for item in longer_validation
            )
            mean_train_sequence_tokens = sum(len(item.token_ids) for item in longer_train) / len(
                longer_train
            )
            mean_validation_sequence_tokens = sum(
                len(item.token_ids) for item in longer_validation
            ) / len(longer_validation)
            expected_pilot_train_sha256 = _tokenized_inventory_sha256(pilot_train_tokenized)
            expected_pilot_validation_sha256 = _tokenized_inventory_sha256(
                pilot_validation_tokenized
            )
            for batch_size in self.inputs.v04.pilot.batch_sizes:
                maximum_train_sequence_exercised = _pilot_exercises_global_maximum(
                    pilot_train_tokenized,
                    batch_size=batch_size,
                    seed=self.inputs.v04.training.seed,
                    steps=self.inputs.v04.pilot.steps,
                )
                if not maximum_train_sequence_exercised:
                    raise PipelineExecutionError(
                        "v0.4 pilot schedule does not exercise its maximum training sequence"
                    )
                training = _v04_pilot_training_config(
                    self.inputs,
                    batch_size=batch_size,
                )
                candidate_id = f"{self.inputs.v04.pilot.candidate_id}-b{batch_size}"
                result, result_artifacts = _run_training(
                    context,
                    guard=self.guard,
                    inputs=self.inputs,
                    candidate_id=candidate_id,
                    sampling_strategy="task_balanced",
                    model_config=self.inputs.v04.longer_context_model,
                    training=training,
                    train_examples=pilot_train,
                    validation_examples=pilot_validation,
                    evaluation_callback=None,
                )
                if (
                    result.train_example_count != len(pilot_train)
                    or result.validation_example_count != len(pilot_validation)
                    or result.train_tokenized_sha256 != expected_pilot_train_sha256
                    or result.validation_tokenized_sha256 != expected_pilot_validation_sha256
                ):
                    raise PipelineExecutionError(
                        "v0.4 pilot result differs from its longest-sequence inventory"
                    )
                _load_candidate_checkpoint(
                    context.attempt_directory,
                    candidate_id,
                    result,
                    self.inputs.tokenizer,
                )
                artifacts.extend(result_artifacts)
                measurements.append(
                    V04PilotMeasurement(
                        batch_size=batch_size,
                        training_result_sha256=result.checksum_sha256,
                        training_config_sha256=result.training_config_sha256,
                        model_config_sha256=result.model_config_sha256,
                        train_tokenized_sha256=result.train_tokenized_sha256,
                        validation_tokenized_sha256=result.validation_tokenized_sha256,
                        tokenizer_manifest_sha256=result.tokenizer_manifest_sha256,
                        checkpoint_manifest_sha256=result.checkpoint_manifest_sha256,
                        device=result.device,
                        train_example_count=len(longer_train),
                        validation_example_count=len(longer_validation),
                        pilot_train_example_count=result.train_example_count,
                        pilot_validation_example_count=result.validation_example_count,
                        train_length_inventory_sha256=train_length_inventory_sha256,
                        validation_length_inventory_sha256=(validation_length_inventory_sha256),
                        maximum_train_sequence_tokens=maximum_train_sequence_tokens,
                        maximum_validation_sequence_tokens=maximum_validation_sequence_tokens,
                        mean_train_sequence_tokens=float(mean_train_sequence_tokens),
                        mean_validation_sequence_tokens=float(mean_validation_sequence_tokens),
                        maximum_train_sequence_exercised=True,
                        finite_loss=True,
                        checkpoint_reloaded=True,
                        elapsed_seconds=float(result.elapsed_seconds),
                        process_peak_rss_bytes=result.process_peak_rss_bytes,
                    )
                )
        draft = V04PilotReport.model_construct(
            candidate_id=self.inputs.v04.pilot.candidate_id,
            requested_device="mps",
            required_resolved_device="mps",
            mandatory_batch_resolved_device=(
                measurements[-1].device.resolved if measurements else None
            ),
            prompt_truncation_rate=frozen_truncation_rate,
            v03_train_prompt_truncation_rate=v03_train_truncation_rate,
            material_truncation_threshold=threshold,
            activated=activated,
            measurements=tuple(measurements),
            passed=bool(
                not activated
                or (
                    measurements
                    and measurements[-1].batch_size == 4
                    and all(
                        item.device.requested == "mps"
                        and item.device.resolved == "mps"
                        and not item.device.fallback_used
                        for item in measurements
                    )
                )
            ),
            checksum_sha256="0" * 64,
        )
        report = _bound_model(draft, V04PilotReport)
        artifacts.append(_contract_artifact(context, "v04-pilot.json", report))
        return self._finish(
            context,
            _stage_outcome(
                (
                    "Eligible v0.4 context pilot completed with native-MPS evidence."
                    if report.activated
                    else "Longer-context pilot was not activated because v0.3 prompt "
                    "truncation was below the frozen materiality threshold."
                ),
                advancement_allowed=bool(report.passed),
                artifacts=tuple(artifacts),
                metrics=(
                    StageMetric(
                        name="prompt_truncation_rate",
                        value=float(frozen_truncation_rate),
                        unit="ratio",
                    ),
                    StageMetric(
                        name="v03_train_prompt_truncation_rate",
                        value=float(v03_train_truncation_rate),
                        unit="ratio",
                    ),
                ),
            ),
        )

    def v04_candidate_training(self, context: StageContext) -> StageOutcome:
        self._start(context)
        pilot_attempt = _upstream_attempt(context, self.config, "v04_pilot")
        pilot = _read_contract(pilot_attempt / "v04-pilot.json", V04PilotReport)
        _verify_v04_pilot_training_evidence(
            pilot_attempt,
            pilot,
            self.inputs,
            source_commit=context.source_commit,
        )
        selection_attempt = _upstream_attempt(context, self.config, "v03_development_evaluation")
        selection = _read_contract(
            selection_attempt / "v03-candidate-selection.json",
            CandidateSelectionReport,
        )
        if not pilot.activated:
            training_attempt = _upstream_attempt(context, self.config, "v03_candidate_training")
            result = _load_training_result(
                training_attempt,
                selection.selected_candidate_id,
            )
            draft = V04CandidateTrainingReport.model_construct(
                activated=False,
                candidate_id=selection.selected_candidate_id,
                reused_v03_candidate=True,
                source_stage="v03_candidate_training",
                training_result_sha256=result.checksum_sha256,
                checkpoint_manifest_sha256=result.checkpoint_manifest_sha256,
                checksum_sha256="0" * 64,
            )
            report = _bound_model(draft, V04CandidateTrainingReport)
            artifact = _contract_artifact(context, "v04-candidate-training.json", report)
            return self._finish(
                context,
                _stage_outcome(
                    "Longer context was not activated; the frozen v0.3 candidate is reused.",
                    artifacts=(artifact,),
                ),
            )

        passed_pilot_batches = {
            item.batch_size
            for item in pilot.measurements
            if (
                item.finite_loss
                and item.checkpoint_reloaded
                and item.device.requested == "mps"
                and item.device.resolved == "mps"
                and not item.device.fallback_used
            )
        }
        if (
            not pilot.passed
            or pilot.requested_device != "mps"
            or pilot.required_resolved_device != "mps"
            or pilot.mandatory_batch_resolved_device != "mps"
            or self.inputs.v04.training.batch_size not in passed_pilot_batches
        ):
            raise PipelineExecutionError("v0.4 main training batch size lacks a passing MPS pilot")
        iid = _load_stage_dataset(context, self.config, "v03_data_audit", "dataset-v03")
        audit_attempt = _upstream_attempt(context, self.config, "v03_data_audit")
        selection_manifest = _read_contract(
            audit_attempt / "v03-semantic-selection.json",
            SemanticSelectionManifest,
        )
        iid_selection = resolve_semantic_selection_examples(
            iid,
            selection_manifest,
            self.inputs.v03,
        )
        train_examples = tuple(
            item for item in iid.examples if item.view is RemediationView.IID_TRAIN
        )

        def evaluation_callback(model: TransformerLM, _step: int, _nll: float) -> float:
            device = next(model.parameters()).device
            predictions = _guarded_decode_examples(
                context,
                guard=self.guard,
                model=model,
                tokenizer=self.inputs.tokenizer,
                examples=iid_selection,
                generation_caps=self.inputs.generation_caps,
                device=device,
                progress_message="v0.4 checkpoint-selection decoding in progress.",
            )
            return 1.0 - semantic_composite_score(iid_selection, predictions)

        result, artifacts = _run_training(
            context,
            guard=self.guard,
            inputs=self.inputs,
            candidate_id=self.inputs.v04.pilot.candidate_id,
            sampling_strategy="task_balanced",
            model_config=self.inputs.v04.longer_context_model,
            training=self.inputs.v04.training,
            train_examples=train_examples,
            validation_examples=iid_selection,
            evaluation_callback=evaluation_callback,
        )
        draft = V04CandidateTrainingReport.model_construct(
            activated=True,
            candidate_id=self.inputs.v04.pilot.candidate_id,
            reused_v03_candidate=False,
            source_stage="v04_candidate_training",
            training_result_sha256=result.checksum_sha256,
            checkpoint_manifest_sha256=result.checkpoint_manifest_sha256,
            checksum_sha256="0" * 64,
        )
        report = _bound_model(draft, V04CandidateTrainingReport)
        report_artifact = _contract_artifact(context, "v04-candidate-training.json", report)
        return self._finish(
            context,
            _stage_outcome(
                "Activated v0.4 longer-context candidate completed IID-only selection training.",
                artifacts=(*artifacts, report_artifact),
            ),
        )

    def _v04_checkpoint(
        self,
        context: StageContext,
    ) -> tuple[
        V04CandidateTrainingReport,
        CompactTrainingResult,
        TransformerLM,
        CheckpointManifest,
        torch.device,
    ]:
        v04_attempt = _upstream_attempt(context, self.config, "v04_candidate_training")
        report = _read_contract(
            v04_attempt / "v04-candidate-training.json",
            V04CandidateTrainingReport,
        )
        training_attempt = (
            _upstream_attempt(context, self.config, "v03_candidate_training")
            if report.reused_v03_candidate
            else v04_attempt
        )
        result = _load_training_result(training_attempt, report.candidate_id)
        if (
            result.checksum_sha256 != report.training_result_sha256
            or result.checkpoint_manifest_sha256 != report.checkpoint_manifest_sha256
        ):
            raise PipelineExecutionError("v0.4 candidate binding differs from training evidence")
        model, checkpoint, device = _load_candidate_checkpoint(
            training_attempt,
            report.candidate_id,
            result,
            self.inputs.tokenizer,
        )
        return report, result, model, checkpoint, device

    def _require_v04_variant_training_provenance(
        self,
        context: StageContext,
        report: V04CandidateTrainingReport,
        result: CompactTrainingResult,
        checkpoint: CheckpointManifest,
    ) -> None:
        """Rebuild and bind the active 1024-context candidate's training inputs."""

        iid = _load_stage_dataset(context, self.config, "v03_data_audit", "dataset-v03")
        augmentation = self.inputs.v03.augmentation
        regenerated = build_frozen_v03_iid_material(
            self.inputs.v03_dataset_config,
            source_commit=self.inputs.frozen_data_source_commit,
            train_template_families=tuple(augmentation.train_template_families),
            train_alias_families=tuple(augmentation.train_alias_families),
            renderer_variants_per_projection=augmentation.renderer_variants_per_projection,
            include_insufficient_evidence_views=augmentation.include_insufficient_evidence_views,
        )
        _require_exact_regenerated_iid(regenerated.dataset, iid)
        audit_attempt = _upstream_attempt(context, self.config, "v03_data_audit")
        frozen_selection = _read_contract(
            audit_attempt / "v03-semantic-selection.json",
            SemanticSelectionManifest,
        )
        validation_examples = resolve_semantic_selection_examples(
            iid,
            frozen_selection,
            self.inputs.v03,
        )
        train_examples = tuple(
            item for item in iid.examples if item.view is RemediationView.IID_TRAIN
        )
        if (
            not train_examples
            or len(validation_examples) != self.inputs.v03.semantic_selection_example_limit
        ):
            raise PipelineExecutionError(
                "v0.4 variant training inputs differ from the exact frozen inventories"
            )
        context_length = self.inputs.v04.longer_context_model.context_length
        tokenized_train = _tokenize_examples(
            train_examples,
            self.inputs.tokenizer,
            context_length=context_length,
            generation_caps=self.inputs.generation_caps,
        )
        tokenized_validation = _tokenize_examples(
            validation_examples,
            self.inputs.tokenizer,
            context_length=context_length,
            generation_caps=self.inputs.generation_caps,
        )
        expected_train_inventory_sha256 = canonical_sha256(
            tuple((item.example_id, item.checksum_sha256) for item in train_examples)
        )
        expected_validation_inventory_sha256 = canonical_sha256(
            tuple((item.example_id, item.checksum_sha256) for item in validation_examples)
        )
        if (
            not report.activated
            or report.reused_v03_candidate
            or report.source_stage != "v04_candidate_training"
            or report.candidate_id != self.inputs.v04.pilot.candidate_id
            or result.candidate_id != report.candidate_id
            or result.checksum_sha256 != report.training_result_sha256
            or result.sampling_strategy != "task_balanced"
            or result.train_example_count != len(train_examples)
            or result.validation_example_count != len(validation_examples)
            or result.train_inventory_sha256 != expected_train_inventory_sha256
            or result.validation_inventory_sha256 != expected_validation_inventory_sha256
            or result.train_tokenized_sha256 != _tokenized_inventory_sha256(tokenized_train)
            or result.validation_tokenized_sha256
            != _tokenized_inventory_sha256(tokenized_validation)
            or not _training_checkpoint_matches_result(
                result,
                checkpoint,
                model_config=self.inputs.v04.longer_context_model,
                training=self.inputs.v04.training,
                tokenizer_manifest_sha256=self.inputs.tokenizer.manifest.checksum_sha256,
                source_commit=context.source_commit,
            )
        ):
            raise PipelineExecutionError(
                "v0.4 variant training evidence differs from exact frozen inputs"
            )

    def _verified_v04_candidate_inventory(
        self,
        context: StageContext,
    ) -> tuple[tuple[str, int, str], ...]:
        """Reopen the exact control/variant identities consumed by v0.4 ranking."""

        selection_attempt = _upstream_attempt(
            context,
            self.config,
            "v03_development_evaluation",
        )
        selection = _read_contract(
            selection_attempt / "v03-candidate-selection.json",
            CandidateSelectionReport,
        )
        v03_training_attempt = _upstream_attempt(
            context,
            self.config,
            "v03_candidate_training",
        )
        control_result = _load_training_result(
            v03_training_attempt,
            selection.selected_candidate_id,
        )
        control_model, control_checkpoint, _control_device = _load_candidate_checkpoint(
            v03_training_attempt,
            selection.selected_candidate_id,
            control_result,
            self.inputs.tokenizer,
        )
        if (
            control_result.candidate_id != selection.selected_candidate_id
            or control_result.checkpoint_manifest_sha256
            != selection.selected_checkpoint_manifest_sha256
            or control_checkpoint.checksum_sha256 != selection.selected_checkpoint_manifest_sha256
            or control_model.config != self.inputs.v02.model
        ):
            raise PipelineExecutionError(
                "v0.4 control identity differs from verified v0.3 selection evidence"
            )
        expected = [
            (
                selection.selected_candidate_id,
                control_model.config.context_length,
                control_checkpoint.checksum_sha256,
            )
        ]
        candidate_report, candidate_result, model, checkpoint, _device = self._v04_checkpoint(
            context
        )
        if candidate_report.activated:
            if (
                candidate_report.candidate_id == selection.selected_candidate_id
                or candidate_result.candidate_id != candidate_report.candidate_id
                or model.config != self.inputs.v04.longer_context_model
                or checkpoint.checksum_sha256 != candidate_report.checkpoint_manifest_sha256
            ):
                raise PipelineExecutionError(
                    "v0.4 variant identity differs from candidate-training evidence"
                )
            self._require_v04_variant_training_provenance(
                context,
                candidate_report,
                candidate_result,
                checkpoint,
            )
            expected.append(
                (
                    candidate_report.candidate_id,
                    model.config.context_length,
                    checkpoint.checksum_sha256,
                )
            )
        elif (
            candidate_report.candidate_id != selection.selected_candidate_id
            or candidate_result.candidate_id != control_result.candidate_id
            or candidate_result.checksum_sha256 != control_result.checksum_sha256
            or checkpoint.checksum_sha256 != control_checkpoint.checksum_sha256
            or model.config != control_model.config
        ):
            raise PipelineExecutionError(
                "inactive v0.4 candidate does not exactly reuse the verified control"
            )
        return tuple(sorted(expected))

    def v04_shadow_evaluation(self, context: StageContext) -> StageOutcome:
        self._start(context)
        if (
            self.inputs.v04.variants.context_candidate_selection_rule
            != "all_gates_then_highest_min_view_composite_then_iid_composite_then_shorter_context"
        ):
            raise PipelineExecutionError("v0.4 context-candidate selection rule is not frozen")
        iid = _load_stage_dataset(context, self.config, "v03_data_audit", "dataset-v03")
        shadow = _load_stage_dataset(
            context, self.config, "v04_shadow_freeze", "dataset-v04-shadow"
        )
        train_examples = tuple(
            item for item in iid.examples if item.view is RemediationView.IID_TRAIN
        )
        full_iid_validation = tuple(
            item for item in iid.examples if item.view is RemediationView.IID_VALIDATION
        )
        if not train_examples or not full_iid_validation:
            raise PipelineExecutionError("v0.4 requires non-empty train and full IID views")

        selection_attempt = _upstream_attempt(
            context,
            self.config,
            "v03_development_evaluation",
        )
        selection = _read_contract(
            selection_attempt / "v03-candidate-selection.json",
            CandidateSelectionReport,
        )
        v03_training_attempt = _upstream_attempt(
            context,
            self.config,
            "v03_candidate_training",
        )
        control_result = _load_training_result(
            v03_training_attempt,
            selection.selected_candidate_id,
        )
        control_model, control_checkpoint, control_device = _load_candidate_checkpoint(
            v03_training_attempt,
            selection.selected_candidate_id,
            control_result,
            self.inputs.tokenizer,
        )
        if control_model.config.context_length != self.inputs.v02.model.context_length:
            raise PipelineExecutionError("v0.4 control no longer uses the frozen 512 context")
        candidates = [
            _EvaluationCandidate(
                candidate_id=selection.selected_candidate_id,
                result=control_result,
                model=control_model,
                checkpoint=control_checkpoint,
                device=control_device,
            )
        ]
        candidate_report, candidate_result, model, checkpoint, device = self._v04_checkpoint(
            context
        )
        if candidate_report.activated:
            if model.config.context_length != self.inputs.v04.longer_context_model.context_length:
                raise PipelineExecutionError("activated v0.4 candidate is not the frozen variant")
            candidates.append(
                _EvaluationCandidate(
                    candidate_id=candidate_report.candidate_id,
                    result=candidate_result,
                    model=model,
                    checkpoint=checkpoint,
                    device=device,
                )
            )
        elif (
            candidate_report.candidate_id != selection.selected_candidate_id
            or checkpoint.checksum_sha256 != control_checkpoint.checksum_sha256
        ):
            raise PipelineExecutionError("inactive v0.4 candidate did not reuse the control")

        ordered_candidates = tuple(sorted(candidates, key=lambda item: item.candidate_id))
        if len({item.candidate_id for item in ordered_candidates}) != len(ordered_candidates):
            raise PipelineExecutionError("v0.4 comparison candidate identities collide")
        artifacts: list[ArtifactReference] = []
        candidate_evidence: list[V04CandidateEvaluation] = []
        candidate_acceptance: dict[str, V04AcceptanceResult] = {}
        for candidate_index, candidate in enumerate(ordered_candidates):
            stem = f"v04-candidate-{candidate_index:02d}"
            iid_evaluation, _iid_baseline, _iid_predictions, iid_artifacts = (
                _evaluate_candidate_view(
                    context,
                    guard=self.guard,
                    inputs=self.inputs,
                    config_sha256_value=config_sha256(self.inputs.v04),
                    dataset=iid,
                    train_examples=train_examples,
                    evaluation_examples=full_iid_validation,
                    view=RemediationView.IID_VALIDATION,
                    model=candidate.model,
                    checkpoint_manifest=candidate.checkpoint,
                    device=candidate.device,
                    stem=f"{stem}-iid",
                )
            )
            artifacts.extend(iid_artifacts)
            iid_composite = _semantic_report_composite(iid_evaluation)
            iid_acceptance = evaluate_v03_acceptance(iid_evaluation.view_metrics)
            artifacts.append(
                _contract_artifact(
                    context,
                    f"{stem}-iid-acceptance.json",
                    iid_acceptance,
                )
            )

            shadow_evaluations: list[SemanticEvaluationReport] = []
            shadow_index: list[tuple[RemediationView, str]] = []
            view_composites = [iid_composite]
            for view in SHADOW_VIEWS:
                examples = tuple(item for item in shadow.examples if item.view is view)
                evaluation, _baseline, _predictions, view_artifacts = _evaluate_candidate_view(
                    context,
                    guard=self.guard,
                    inputs=self.inputs,
                    config_sha256_value=config_sha256(self.inputs.v04),
                    dataset=shadow,
                    train_examples=train_examples,
                    evaluation_examples=examples,
                    view=view,
                    model=candidate.model,
                    checkpoint_manifest=candidate.checkpoint,
                    device=candidate.device,
                    stem=f"{stem}-{view.value}",
                )
                shadow_evaluations.append(evaluation)
                shadow_index.append((view, evaluation.checksum_sha256))
                artifacts.extend(view_artifacts)
                view_composites.append(_semantic_report_composite(evaluation))
            acceptance = evaluate_v04_acceptance(
                iid_acceptance,
                tuple(item.view_metrics for item in shadow_evaluations),
            )
            candidate_acceptance[candidate.candidate_id] = acceptance
            artifacts.append(
                _contract_artifact(
                    context,
                    f"{stem}-acceptance.json",
                    acceptance,
                )
            )
            candidate_evidence.append(
                V04CandidateEvaluation(
                    candidate_id=candidate.candidate_id,
                    context_length=candidate.model.config.context_length,
                    checkpoint_manifest_sha256=candidate.checkpoint.checksum_sha256,
                    iid_report_sha256=iid_evaluation.checksum_sha256,
                    iid_acceptance_sha256=iid_acceptance.checksum_sha256,
                    shadow_reports=tuple(shadow_index),
                    v04_acceptance_sha256=acceptance.checksum_sha256,
                    all_required_gates_passed=bool(acceptance.advancement_allowed),
                    worst_view_semantic_composite=min(view_composites),
                    iid_semantic_composite=iid_composite,
                )
            )

        ordered_evidence = tuple(sorted(candidate_evidence, key=lambda item: item.candidate_id))
        selected = _select_v04_candidate(ordered_evidence)
        selected_acceptance = candidate_acceptance[selected.candidate_id]
        artifacts.append(
            _contract_artifact(
                context,
                "v04-acceptance.json",
                selected_acceptance,
            )
        )
        draft = V04EvaluationIndex.model_construct(
            candidates=ordered_evidence,
            selected_candidate_id=selected.candidate_id,
            checkpoint_manifest_sha256=selected.checkpoint_manifest_sha256,
            iid_report_sha256=selected.iid_report_sha256,
            shadow_reports=selected.shadow_reports,
            v04_acceptance_sha256=selected.v04_acceptance_sha256,
            checksum_sha256="0" * 64,
        )
        index = _bound_model(draft, V04EvaluationIndex)
        index_artifact = _contract_artifact(context, "v04-evaluation-index.json", index)
        artifacts.append(index_artifact)
        return self._finish(
            context,
            _stage_outcome(
                "Control and activated context variant were compared on full IID and shadows.",
                artifacts=tuple(artifacts),
                metrics=(
                    StageMetric(
                        name="candidate_count",
                        value=float(len(ordered_evidence)),
                        unit="candidates",
                    ),
                    StageMetric(
                        name="required_shadow_views",
                        value=float(len(SHADOW_VIEWS)),
                        unit="views",
                    ),
                    StageMetric(
                        name="selected_worst_view_semantic_composite",
                        value=float(selected.worst_view_semantic_composite),
                        unit="ratio",
                    ),
                ),
            ),
        )

    def v04_gate_and_final_policy_freeze(self, context: StageContext) -> StageOutcome:
        self._start(context)
        iid_dataset = _load_stage_dataset(context, self.config, "v03_data_audit", "dataset-v03")
        shadow_dataset = _load_stage_dataset(
            context,
            self.config,
            "v04_shadow_freeze",
            "dataset-v04-shadow",
        )
        train_examples = tuple(
            item for item in iid_dataset.examples if item.view is RemediationView.IID_TRAIN
        )
        expected_scopes: dict[RemediationView, tuple[int, str]] = {}
        for view in (RemediationView.IID_VALIDATION, *SHADOW_VIEWS):
            source_dataset = (
                iid_dataset if view is RemediationView.IID_VALIDATION else shadow_dataset
            )
            examples = tuple(item for item in source_dataset.examples if item.view is view)
            if not train_examples or not examples:
                raise PipelineExecutionError("v0.4 exact evaluation view inventory is empty")
            scoped = _subset_dataset(
                source_dataset,
                (*train_examples, *examples),
                dataset_version=source_dataset.manifest.dataset_version,
            )
            expected_scopes[view] = (len(examples), scoped.manifest.checksum_sha256)
        expected_config_sha256 = config_sha256(self.inputs.v04)
        evaluation_attempt = _upstream_attempt(context, self.config, "v04_shadow_evaluation")
        index = _read_contract(
            evaluation_attempt / "v04-evaluation-index.json",
            V04EvaluationIndex,
        )
        expected_candidates = self._verified_v04_candidate_inventory(context)
        indexed_candidates = tuple(
            (item.candidate_id, item.context_length, item.checkpoint_manifest_sha256)
            for item in index.candidates
        )
        if indexed_candidates != expected_candidates:
            raise PipelineExecutionError(
                "v0.4 evaluation candidate inventory differs from verified training evidence"
            )
        verified_acceptance: dict[str, V04AcceptanceResult] = {}
        for candidate_index, candidate in enumerate(index.candidates):
            stem = f"v04-candidate-{candidate_index:02d}"
            iid_report = _read_contract(
                evaluation_attempt / f"{stem}-iid-semantic-evaluation.json",
                SemanticEvaluationReport,
            )
            iid_acceptance = _read_contract(
                evaluation_attempt / f"{stem}-iid-acceptance.json",
                V03AcceptanceResult,
            )
            shadow_reports = tuple(
                _read_contract(
                    evaluation_attempt / f"{stem}-{view.value}-semantic-evaluation.json",
                    SemanticEvaluationReport,
                )
                for view in SHADOW_VIEWS
            )
            candidate_acceptance = _read_contract(
                evaluation_attempt / f"{stem}-acceptance.json",
                V04AcceptanceResult,
            )
            iid_count, iid_dataset_sha256 = expected_scopes[RemediationView.IID_VALIDATION]
            _require_semantic_report_scope(
                iid_report,
                view=RemediationView.IID_VALIDATION,
                example_count=iid_count,
                dataset_manifest_sha256=iid_dataset_sha256,
                source_commit=context.source_commit,
                config_sha256_value=expected_config_sha256,
                tokenizer_manifest_sha256=self.inputs.tokenizer.manifest.checksum_sha256,
                output_contract_sha256=self.inputs.compact_contract_sha256,
                checkpoint_manifest_sha256=candidate.checkpoint_manifest_sha256,
            )
            for view, report in zip(SHADOW_VIEWS, shadow_reports, strict=True):
                view_count, view_dataset_sha256 = expected_scopes[view]
                _require_semantic_report_scope(
                    report,
                    view=view,
                    example_count=view_count,
                    dataset_manifest_sha256=view_dataset_sha256,
                    source_commit=context.source_commit,
                    config_sha256_value=expected_config_sha256,
                    tokenizer_manifest_sha256=self.inputs.tokenizer.manifest.checksum_sha256,
                    output_contract_sha256=self.inputs.compact_contract_sha256,
                    checkpoint_manifest_sha256=candidate.checkpoint_manifest_sha256,
                )
            expected_iid_acceptance = evaluate_v03_acceptance(iid_report.view_metrics)
            expected_candidate_acceptance = evaluate_v04_acceptance(
                expected_iid_acceptance,
                tuple(item.view_metrics for item in shadow_reports),
            )
            iid_composite = _semantic_report_composite(iid_report)
            view_composites = (
                iid_composite,
                *tuple(_semantic_report_composite(item) for item in shadow_reports),
            )
            if (
                iid_report.evaluation_view is not RemediationView.IID_VALIDATION
                or iid_report.checksum_sha256 != candidate.iid_report_sha256
                or iid_acceptance.checksum_sha256 != candidate.iid_acceptance_sha256
                or iid_acceptance != expected_iid_acceptance
                or iid_acceptance.view_metrics != iid_report.view_metrics
                or iid_report.view_metrics.artifacts.checkpoint_sha256
                != candidate.checkpoint_manifest_sha256
                or tuple(item.evaluation_view for item in shadow_reports) != SHADOW_VIEWS
                or tuple(item.checksum_sha256 for item in shadow_reports)
                != tuple(checksum for _view, checksum in candidate.shadow_reports)
                or candidate_acceptance.v03_result != iid_acceptance
                or candidate_acceptance.shadow_view_metrics
                != tuple(item.view_metrics for item in shadow_reports)
                or candidate_acceptance.checksum_sha256 != candidate.v04_acceptance_sha256
                or candidate_acceptance != expected_candidate_acceptance
                or bool(candidate_acceptance.advancement_allowed)
                is not candidate.all_required_gates_passed
                or candidate.iid_semantic_composite != iid_composite
                or candidate.worst_view_semantic_composite != min(view_composites)
                or any(
                    item.view_metrics.artifacts.checkpoint_sha256
                    != candidate.checkpoint_manifest_sha256
                    for item in shadow_reports
                )
            ):
                raise PipelineExecutionError(
                    "v0.4 candidate index differs from its immutable evaluation evidence"
                )
            verified_acceptance[candidate.candidate_id] = candidate_acceptance
        if _select_v04_candidate(index.candidates).candidate_id != index.selected_candidate_id:
            raise PipelineExecutionError("v0.4 candidate ranking changed during gate verification")
        acceptance = _read_contract(
            evaluation_attempt / "v04-acceptance.json",
            V04AcceptanceResult,
        )
        selected_acceptance = verified_acceptance[index.selected_candidate_id]
        if (
            acceptance != selected_acceptance
            or acceptance.checksum_sha256 != index.v04_acceptance_sha256
        ):
            raise PipelineExecutionError("v0.4 gate acceptance differs from selected evidence")
        draft = FinalEvaluationPolicyFreeze.model_construct(
            v04_acceptance_sha256=acceptance.checksum_sha256,
            development_gate_passed=bool(acceptance.advancement_allowed),
            status=(
                "locked_pending_owner_reviewed_fresh_extension"
                if acceptance.advancement_allowed
                else "locked_development_gate_failed"
            ),
            checksum_sha256="0" * 64,
        )
        policy = _bound_model(draft, FinalEvaluationPolicyFreeze)
        policy_artifact = _contract_artifact(
            context,
            "final-evaluation-policy.json",
            policy,
        )
        return self._finish(
            context,
            _stage_outcome(
                "Worst-split v0.4 development gate evaluated and final-access policy froze.",
                advancement_allowed=bool(acceptance.advancement_allowed),
                artifacts=(policy_artifact,),
            ),
        )

    def review_bundle(self, context: StageContext) -> StageOutcome:
        self._start(context)
        state = PipelineStore(
            context.run_directory,
            maximum_state_bytes=self.config.maximum_status_bytes,
        ).load_state()
        bindings: list[ReviewStageBinding] = []
        for record in state.stages[:-1]:
            if record.status is not StageStatus.COMPLETED or record.latest_attempt_path is None:
                raise PipelineExecutionError("complete review bundle lacks a completed prefix")
            bindings.append(
                ReviewStageBinding(
                    stage=record.name,
                    outcome=_stage_completion_outcome(
                        context.run_directory,
                        record.latest_attempt_path,
                    ),
                )
            )
        policy_attempt = _upstream_attempt(context, self.config, "v04_gate_and_final_policy_freeze")
        policy = _read_contract(
            policy_attempt / "final-evaluation-policy.json",
            FinalEvaluationPolicyFreeze,
        )
        if not policy.development_gate_passed:
            raise PipelineExecutionError("complete review bundle requires a passing policy")
        for filename in (
            FINAL_EVALUATION_READY_FILENAME,
            OWNER_REVIEW_APPROVED_FILENAME,
            FRESH_EXTENSION_MANIFEST_FILENAME,
            FINAL_ACCESS_LEDGER_FILENAME,
        ):
            path = context.run_directory / filename
            if path.exists() or path.is_symlink():
                raise PipelineExecutionError("development review bundle found a final-access file")
        summary_path = context.attempt_directory / "REVIEW_BUNDLE.md"
        summary = _complete_review_markdown(
            run_name=self.config.run_name,
            source_commit=context.source_commit,
            bindings=tuple(bindings),
            policy=policy,
        )
        _write_bytes(summary_path, summary.encode("utf-8"))
        summary_relative = summary_path.relative_to(context.run_directory).as_posix()
        draft = ReviewBundleManifest.model_construct(
            run_name=self.config.run_name,
            source_commit=context.source_commit,
            pipeline_config_sha256=config_sha256(self.config),
            stages=tuple(bindings),
            final_policy_sha256=policy.checksum_sha256,
            summary_relative_path=summary_relative,
            summary_sha256=_sha256(summary_path),
            checksum_sha256="0" * 64,
        )
        manifest = _bound_model(draft, ReviewBundleManifest)
        manifest_artifact = _contract_artifact(context, "review-bundle.json", manifest)
        summary_artifact = _artifact_reference(
            summary_path,
            run_directory=context.run_directory,
        )
        return self._finish(
            context,
            _stage_outcome(
                "Machine-readable and human-readable development review bundles completed.",
                artifacts=(manifest_artifact, summary_artifact),
            ),
        )


def _complete_review_markdown(
    *,
    run_name: str,
    source_commit: str,
    bindings: tuple[ReviewStageBinding, ...],
    policy: FinalEvaluationPolicyFreeze,
) -> str:
    if (
        type(run_name) is not str
        or not run_name
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,95}", run_name)
        or type(source_commit) is not str
        or not re.fullmatch(r"[0-9a-f]{7,64}", source_commit)
        or type(bindings) is not tuple
        or any(type(binding) is not ReviewStageBinding for binding in bindings)
        or type(policy) is not FinalEvaluationPolicyFreeze
    ):
        raise TypeError("complete review summary requires exact frozen contracts")
    if tuple(binding.stage for binding in bindings) != PIPELINE_STAGES[:-1]:
        raise ValueError("complete review summary differs from the frozen stage prefix")
    lines = [
        "# ReactorBench-LM development review bundle",
        "",
        f"- Run: `{run_name}`",
        f"- Source commit: `{source_commit}`",
        f"- Final-access policy: `{policy.status}`",
        f"- Final-access policy checksum: `{policy.checksum_sha256}`",
        "- Final or historical held-out payload included: `no`",
        "",
        "## Immutable development stages",
        "",
        "| Stage | Outcome | SHA-256 |",
        "|---|---|---|",
    ]
    lines.extend(
        f"| `{binding.stage}` | `{binding.outcome.relative_path}` | `{binding.outcome.sha256}` |"
        for binding in bindings
    )
    lines.extend(
        (
            "",
            "This bundle records development evidence only. It does not authorize, run, "
            "or report final evaluation.",
            "",
        )
    )
    return "\n".join(lines)


def _terminal_review_markdown(
    *,
    state: PipelineState,
    stages: tuple[TerminalReviewStage, ...],
    completed_prefix_length: int,
) -> str:
    lines = [
        "# ReactorBench-LM terminal development review",
        "",
        f"- Run: `{state.run_name}`",
        f"- Source commit: `{state.source_commit}`",
        f"- Pipeline status: `{state.status}`",
        f"- Pipeline-state checksum: `{state.checksum_sha256}`",
        f"- Completed stage prefix: `{completed_prefix_length}/{len(PIPELINE_STAGES)}`",
        "- Final evaluation accessed: `no`",
        "",
        "## Recorded stage prefix",
        "",
    ]
    if stages:
        lines.extend(("| Stage | Status | Attempt | Outcome SHA-256 |", "|---|---|---|---|"))
        for stage in stages:
            outcome_checksum = "-" if stage.outcome is None else stage.outcome.sha256
            lines.append(
                f"| `{stage.stage}` | `{stage.status}` | `{stage.latest_attempt_path}` | "
                f"`{outcome_checksum}` |"
            )
    else:
        lines.append("No stage began before the stop request was honored.")
    lines.extend(
        (
            "",
            "This checksum-bound bundle contains only development pipeline metadata and "
            "immutable outcome references. It contains no final, fresh, golden, or "
            "historical held-out payload.",
            "",
        )
    )
    return "\n".join(lines)


def _terminal_review_stages(
    *,
    run_directory: Path,
    state: PipelineState,
) -> tuple[tuple[TerminalReviewStage, ...], int]:
    stages: list[TerminalReviewStage] = []
    completed_prefix_length = 0
    for record in state.stages:
        if record.status is StageStatus.PENDING:
            break
        if record.status is StageStatus.RUNNING or record.latest_attempt_path is None:
            raise PipelineExecutionError("terminal review found a nonterminal stage")
        expected_attempt = (
            f"stages/{record.ordinal:02d}-{record.name}/attempt-{record.attempt_count:04d}"
        )
        if record.latest_attempt_path != expected_attempt:
            raise PipelineExecutionError("terminal review stage attempt is not canonical")
        attempt = run_directory / expected_attempt
        if attempt.is_symlink() or not attempt.is_dir():
            raise PipelineExecutionError("terminal review stage attempt is unsafe")
        outcome: ArtifactReference | None = None
        if record.status in {StageStatus.COMPLETED, StageStatus.BLOCKED}:
            outcome = _stage_completion_outcome(run_directory, expected_attempt)
            expected_outcome_path = f"{expected_attempt}/outcome.json"
            if outcome.relative_path != expected_outcome_path or outcome != _artifact_reference(
                run_directory / expected_outcome_path,
                run_directory=run_directory,
            ):
                raise PipelineExecutionError("terminal review outcome binding is invalid")
        else:
            marker = attempt.parent / "completed.json"
            if marker.exists() or marker.is_symlink():
                raise PipelineExecutionError("unsuccessful stage has a committed outcome")
        stage_status = cast(
            Literal["completed", "blocked", "failed", "stopped"],
            record.status.value,
        )
        stages.append(
            TerminalReviewStage(
                stage=record.name,
                status=stage_status,
                latest_attempt_path=expected_attempt,
                outcome=outcome,
            )
        )
        if record.status is StageStatus.COMPLETED:
            completed_prefix_length += 1
            continue
        break
    return tuple(stages), completed_prefix_length


def _write_or_verify_bytes(path: Path, expected: bytes) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != expected:
            raise PipelineExecutionError("existing terminal review output differs")
        return
    try:
        _write_bytes(path, expected)
    except FileExistsError:
        if path.is_symlink() or not path.is_file() or path.read_bytes() != expected:
            raise PipelineExecutionError("concurrent terminal review output differs") from None


def write_terminal_review_bundle(
    *,
    project_root: Path,
    config: PipelineConfig,
    source_commit: str,
    state: PipelineState,
) -> ReviewBundleOutput:
    """Publish one idempotent, development-only terminal-state evidence bundle."""

    if (
        not isinstance(project_root, Path)
        or type(config) is not PipelineConfig
        or type(source_commit) is not str
        or not re.fullmatch(r"[0-9a-f]{7,64}", source_commit)
        or type(state) is not PipelineState
    ):
        raise TypeError("terminal review requires exact project/config/state bindings")
    if project_root.is_symlink() or not project_root.is_dir():
        raise ValueError("terminal review project root is unsafe")
    root = project_root.resolve(strict=True)
    validated_state = PipelineState.model_validate_json(
        canonical_json_bytes(state.model_dump(mode="json", round_trip=True)),
        strict=True,
    )
    if validated_state != state or state.status not in {
        "completed",
        "blocked",
        "failed",
        "stopped",
    }:
        raise ValueError("terminal review requires a valid terminal pipeline state")
    run_directory = root / config.run_root / config.run_name
    if run_directory.is_symlink() or not run_directory.is_dir():
        raise ValueError("terminal review run directory is unsafe")
    if not run_directory.resolve(strict=True).is_relative_to(root):
        raise ValueError("terminal review run directory escapes the project")
    store = PipelineStore(
        run_directory,
        maximum_state_bytes=config.maximum_status_bytes,
    )
    manifest = store.load_manifest()
    durable_state = store.load_state()
    expected_config_sha256 = config_sha256(config)
    if (
        durable_state != state
        or state.run_name != config.run_name
        or state.pipeline_config_sha256 != expected_config_sha256
        or state.source_commit != source_commit
        or manifest.run_name != config.run_name
        or manifest.pipeline_config_sha256 != expected_config_sha256
        or manifest.source_commit != source_commit
        or manifest.v02_config_sha256 != config.v02_config_sha256
        or manifest.v03_config_sha256 != config.v03_config_sha256
        or manifest.v04_config_sha256 != config.v04_config_sha256
    ):
        raise PipelineExecutionError("terminal review state/source/config binding differs")

    def unreachable(_context: StageContext) -> StageOutcome:
        raise AssertionError("terminal review dry-run invoked a scientific stage")

    actions: Mapping[str, StageAction] = MappingProxyType(
        dict.fromkeys(PIPELINE_STAGES, unreachable)
    )
    audited = PipelineEngine(
        project_root=root,
        config=config,
        store=store,
        actions=actions,
        stop_requested=lambda: False,
    ).run(dry_run=True)
    if audited != state:
        raise PipelineExecutionError("terminal review changed during boundary audit")
    for filename in (
        FINAL_EVALUATION_READY_FILENAME,
        OWNER_REVIEW_APPROVED_FILENAME,
        FRESH_EXTENSION_MANIFEST_FILENAME,
        FINAL_ACCESS_LEDGER_FILENAME,
        FINAL_RESULT_FILENAME,
        FINAL_REVIEW_FILENAME,
    ):
        forbidden = run_directory / filename
        if forbidden.exists() or forbidden.is_symlink():
            raise PipelineExecutionError("terminal review found a final-access artifact")

    stages, completed_prefix_length = _terminal_review_stages(
        run_directory=run_directory,
        state=state,
    )
    review_root = run_directory / TERMINAL_REVIEW_DIRECTORY
    if not review_root.exists() and not review_root.is_symlink():
        try:
            review_root.mkdir(mode=0o750)
        except FileExistsError:
            pass
    if review_root.is_symlink() or not review_root.is_dir():
        raise PipelineExecutionError("terminal review root is unsafe")
    bundle_directory = review_root / f"state-{state.checksum_sha256}"
    if not bundle_directory.exists() and not bundle_directory.is_symlink():
        try:
            bundle_directory.mkdir(mode=0o750)
        except FileExistsError:
            pass
    if bundle_directory.is_symlink() or not bundle_directory.is_dir():
        raise PipelineExecutionError("terminal review bundle directory is unsafe")
    allowed_names = {
        TERMINAL_REVIEW_MANIFEST_FILENAME,
        TERMINAL_REVIEW_SUMMARY_FILENAME,
    }
    if any(child.name not in allowed_names for child in bundle_directory.iterdir()):
        raise PipelineExecutionError("terminal review bundle contains an unexpected entry")

    summary_path = bundle_directory / TERMINAL_REVIEW_SUMMARY_FILENAME
    summary_bytes = _terminal_review_markdown(
        state=state,
        stages=stages,
        completed_prefix_length=completed_prefix_length,
    ).encode("utf-8")
    summary_relative = summary_path.relative_to(run_directory).as_posix()
    draft = TerminalReviewBundleManifest.model_construct(
        run_name=config.run_name,
        source_commit=source_commit,
        pipeline_config_sha256=expected_config_sha256,
        pipeline_state_sha256=state.checksum_sha256,
        pipeline_status=cast(
            Literal["completed", "blocked", "failed", "stopped"],
            state.status,
        ),
        stages=stages,
        completed_prefix_length=completed_prefix_length,
        summary_relative_path=summary_relative,
        summary_sha256=hashlib.sha256(summary_bytes).hexdigest(),
        checksum_sha256="0" * 64,
    )
    terminal_manifest = _bound_model(draft, TerminalReviewBundleManifest)
    manifest_bytes = (
        canonical_json_bytes(terminal_manifest.model_dump(mode="json", round_trip=True)) + b"\n"
    )
    manifest_path = bundle_directory / TERMINAL_REVIEW_MANIFEST_FILENAME
    _write_or_verify_bytes(summary_path, summary_bytes)
    _write_or_verify_bytes(manifest_path, manifest_bytes)
    verified_manifest = _read_contract(
        manifest_path,
        TerminalReviewBundleManifest,
        maximum_bytes=config.maximum_status_bytes,
    )
    if verified_manifest != terminal_manifest or _sha256(summary_path) != (
        terminal_manifest.summary_sha256
    ):
        raise PipelineExecutionError("terminal review publication failed verification")
    if store.load_state() != state:
        raise PipelineExecutionError("pipeline state changed during terminal review publication")
    return ReviewBundleOutput(
        manifest_path=manifest_path,
        summary_path=summary_path,
        manifest=verified_manifest,
    )


@dataclass(frozen=True, slots=True)
class _GateEvidenceContext:
    run_directory: Path
    source_commit: str


def _verified_replay_reference(root: Path, reference: ArtifactReference) -> Path:
    path = root / reference.relative_path
    cursor = root
    for part in Path(reference.relative_path).parts:
        cursor /= part
        if cursor.is_symlink():
            raise PipelineExecutionError("gate replay evidence traverses a symbolic link")
    if (
        path.is_symlink()
        or not path.is_file()
        or not path.resolve(strict=True).is_relative_to(root.resolve(strict=True))
        or path.stat().st_size != reference.size_bytes
        or _sha256(path) != reference.sha256
    ):
        raise PipelineExecutionError("gate replay artifact checksum or size differs")
    return path


def replay_targeted_v03_gate(
    *,
    project_root: Path,
    config: PipelineConfig,
    replay_source_commit: str,
) -> tuple[Path, V03GateReplayCertification, V03AcceptanceResult]:
    """Certify targeted-03 from immutable completed evidence without retraining."""

    source_name = "phase6-remediation-v0.4.0-targeted-03"
    replay_name = "phase6-remediation-v0.4.0-targeted-03-gate-replay-01"
    if (
        not isinstance(project_root, Path)
        or type(config) is not PipelineConfig
        or not re.fullmatch(r"[0-9a-f]{40,64}", replay_source_commit)
        or config.run_name != source_name
    ):
        raise TypeError("targeted v0.3 gate replay requires its exact frozen bindings")
    root = project_root.resolve(strict=True)
    if (
        _verify_runner_source(
            root,
            source_commit=replay_source_commit,
            run_root=config.run_root,
        )
        != replay_source_commit
    ):
        raise PipelineExecutionError("gate replay source binding differs from the clean checkout")
    source_root = root / config.run_root / source_name
    if source_root.is_symlink() or not source_root.is_dir():
        raise PipelineExecutionError("preserved targeted-03 run is unavailable or unsafe")
    store = PipelineStore(source_root, maximum_state_bytes=config.maximum_status_bytes)
    manifest = store.load_manifest()
    state = store.load_state()
    pipeline_checksum = config_sha256(config)
    manifest_path = source_root / "run-manifest.json"
    state_path = source_root / "pipeline-state.json"
    original_manifest_file_sha256 = _sha256(manifest_path)
    original_state_file_sha256 = _sha256(state_path)
    if (
        manifest.run_name != source_name
        or state.run_name != source_name
        or manifest.pipeline_config_sha256 != pipeline_checksum
        or state.pipeline_config_sha256 != pipeline_checksum
        or state.source_commit != manifest.source_commit
        or manifest.v02_config_sha256 != config.v02_config_sha256
        or manifest.v03_config_sha256 != config.v03_config_sha256
        or manifest.v04_config_sha256 != config.v04_config_sha256
        or state.status != "failed"
        or any(record.status is not StageStatus.COMPLETED for record in state.stages[:9])
        or state.stages[9].status is not StageStatus.FAILED
        or any(record.status is not StageStatus.PENDING for record in state.stages[10:])
    ):
        raise PipelineExecutionError("preserved targeted-03 state differs from the replay policy")

    completion_hashes: dict[int, str] = {}
    for ordinal, record in enumerate(state.stages[:9]):
        marker_path = source_root / "stages" / f"{ordinal:02d}-{record.name}" / "completed.json"
        marker = _read_contract(
            marker_path,
            StageCompletionMarker,
            maximum_bytes=config.maximum_status_bytes,
        )
        if (
            marker.run_name != source_name
            or marker.pipeline_config_sha256 != pipeline_checksum
            or marker.source_commit != manifest.source_commit
            or marker.stage != record.name
            or marker.ordinal != ordinal
            or marker.attempt != record.attempt_count
            or marker.attempt_relative_path != record.latest_attempt_path
        ):
            raise PipelineExecutionError("gate replay completion marker binding differs")
        outcome_path = _verified_replay_reference(source_root, marker.outcome)
        outcome = _read_contract(
            outcome_path,
            StageOutcome,
            maximum_bytes=config.maximum_status_bytes,
        )
        if (
            len(outcome.artifacts) != record.artifact_count
            or len(outcome.metrics) != record.metric_count
            or outcome.summary != record.summary
            or outcome.advancement_allowed is not record.advancement_allowed
        ):
            raise PipelineExecutionError("gate replay stage outcome differs from durable state")
        for reference in outcome.artifacts:
            _verified_replay_reference(source_root, reference)
        completion_hashes[ordinal] = _sha256(marker_path)

    inputs = _load_execution_inputs(project_root=root, config=config)
    runtime = _PipelineRuntime(
        project_root=root,
        config=config,
        source_commit=manifest.source_commit,
        inputs=inputs,
    )
    evidence_context = _GateEvidenceContext(
        run_directory=source_root,
        source_commit=manifest.source_commit,
    )
    acceptance, binding = runtime._reconstruct_v03_gate_evidence(
        cast(StageContext, evidence_context)
    )
    if binding is None or len(acceptance.checks) != 10:
        raise PipelineExecutionError("targeted gate replay did not reconstruct complete evidence")
    if (
        _sha256(manifest_path) != original_manifest_file_sha256
        or _sha256(state_path) != original_state_file_sha256
        or store.load_manifest() != manifest
        or store.load_state() != state
    ):
        raise PipelineExecutionError("source run changed during gate replay")

    replay_root = root / config.run_root / replay_name
    if replay_root.exists() or replay_root.is_symlink():
        raise FileExistsError("gate replay identity already exists")
    replay_root.mkdir(mode=0o750)
    _write_contract(replay_root / "v03-acceptance.json", acceptance)
    _write_contract(replay_root / "v03-targeted-gate-binding.json", binding)
    passed_count = sum(check.passed for check in acceptance.checks)
    draft = V03GateReplayCertification.model_construct(
        source_run_name=source_name,
        replay_name=replay_name,
        source_run_manifest_sha256=manifest.checksum_sha256,
        source_pipeline_state_file_sha256=original_state_file_sha256,
        source_pipeline_state_contract_sha256=state.checksum_sha256,
        source_commit=manifest.source_commit,
        replay_source_commit=replay_source_commit,
        pipeline_config_sha256=pipeline_checksum,
        training_completion_marker_sha256=completion_hashes[7],
        evaluation_completion_marker_sha256=completion_hashes[8],
        acceptance_sha256=acceptance.checksum_sha256,
        targeted_gate_binding_sha256=binding.checksum_sha256,
        passed_check_count=passed_count,
        total_check_count=10,
        advancement_allowed=acceptance.advancement_allowed,
        thresholds_unchanged=True,
        retraining_performed=False,
        final_evaluation_accessed=False,
        checksum_sha256="0" * 64,
    )
    certification = _bound_model(draft, V03GateReplayCertification)
    certification_path = replay_root / "gate-replay-certification.json"
    _write_contract(certification_path, certification)
    if (
        _read_contract(certification_path, V03GateReplayCertification) != certification
        or _read_contract(replay_root / "v03-acceptance.json", V03AcceptanceResult) != acceptance
        or _read_contract(replay_root / "v03-targeted-gate-binding.json", TargetedV03GateBinding)
        != binding
    ):
        raise PipelineExecutionError("gate replay publication failed strict verification")
    return replay_root, certification, acceptance


def build_stage_actions(
    project_root: Path,
    config: PipelineConfig,
    source_commit: str,
) -> Mapping[str, StageAction]:
    """Construct the frozen development graph after a read-only full preflight."""

    if (
        not isinstance(project_root, Path)
        or type(config) is not PipelineConfig
        or type(source_commit) is not str
        or not re.fullmatch(r"[0-9a-f]{7,64}", source_commit)
    ):
        raise TypeError("stage actions require exact project/config/source bindings")
    if project_root.is_symlink() or not project_root.is_dir():
        raise ValueError("pipeline project root is unsafe")
    root = project_root.resolve(strict=True)
    if tuple(config.stage_order) != PIPELINE_STAGES:
        raise ValueError("pipeline config differs from the frozen stage graph")
    verified_commit = _verify_runner_source(
        root,
        source_commit=source_commit,
        run_root=config.run_root,
    )
    if verified_commit != source_commit:
        raise PipelineExecutionError("pipeline source binding must use the full Git commit")
    inputs = _load_execution_inputs(project_root=root, config=config)
    runtime = _PipelineRuntime(
        project_root=root,
        config=config,
        source_commit=source_commit,
        inputs=inputs,
    )
    actions: dict[str, StageAction] = {
        "preflight": runtime.preflight,
        "v02_inventory_and_caps": runtime.v02_inventory_and_caps,
        "v02_smoke": runtime.v02_smoke,
        "v02_development_training": runtime.v02_development_training,
        "v02_development_gate": runtime.v02_development_gate,
        "v03_data_audit": runtime.v03_data_audit,
        "v03_smoke": runtime.v03_smoke,
        "v03_candidate_training": runtime.v03_candidate_training,
        "v03_development_evaluation": runtime.v03_development_evaluation,
        "v03_gate": runtime.v03_gate,
        "v04_shadow_freeze": runtime.v04_shadow_freeze,
        "v04_pilot": runtime.v04_pilot,
        "v04_candidate_training": runtime.v04_candidate_training,
        "v04_shadow_evaluation": runtime.v04_shadow_evaluation,
        "v04_gate_and_final_policy_freeze": runtime.v04_gate_and_final_policy_freeze,
        "review_bundle": runtime.review_bundle,
    }
    if tuple(actions) != PIPELINE_STAGES or any(
        not callable(action) for action in actions.values()
    ):
        raise PipelineExecutionError("constructed actions differ from the frozen graph")
    return MappingProxyType(actions)


def run_final_evaluation(
    *,
    project_root: Path,
    config: PipelineConfig,
    source_commit: str,
    explicit_confirmation: bool,
) -> Never:
    """Fail closed until a separately reviewed final-evidence executor exists."""

    if not isinstance(project_root, Path) or not project_root.is_absolute():
        raise TypeError("final evaluation project root must be an absolute Path")
    if type(config) is not PipelineConfig:
        raise TypeError("final evaluation config must use the exact pipeline contract")
    if type(source_commit) is not str or not re.fullmatch(r"[0-9a-f]{7,64}", source_commit):
        raise TypeError("final evaluation source commit is invalid")
    if type(explicit_confirmation) is not bool:
        raise TypeError("final evaluation confirmation must be an exact boolean")
    raise FinalEvaluationBlockedError(
        "Final evaluation remains locked because the distinct final-evidence executor "
        "has not been implemented or independently reviewed."
    )


__all__ = [
    "ExecutionPreflightReport",
    "FinalEvaluationBlockedError",
    "FinalEvaluationPolicyFreeze",
    "PipelineExecutionError",
    "PipelineResourceLimitError",
    "PipelineStopRequest",
    "ReviewBundleManifest",
    "ReviewBundleOutput",
    "TerminalReviewBundleManifest",
    "TerminalReviewStage",
    "V02DevelopmentGateReport",
    "V04CandidateEvaluation",
    "V04EvaluationIndex",
    "archive_pipeline_stop",
    "build_stage_actions",
    "build_stop_requested",
    "pipeline_stop_file",
    "replay_targeted_v03_gate",
    "request_pipeline_stop",
    "run_final_evaluation",
    "verify_final_evaluation_prerequisites",
    "write_terminal_review_bundle",
]
