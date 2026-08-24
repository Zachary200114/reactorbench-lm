"""Checksum-bound acceptance contracts for Phase 6 remediation.

The contracts in this module are intentionally limited to development evidence.
They do not load artifacts, select checkpoints, or grant access to a final holdout.
Callers must first calculate metrics for checksum-bound development views, then pass
those immutable metric packets to the deterministic gate functions below.
"""

from __future__ import annotations

import math
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, StrictBool, StrictFloat, StrictInt, StrictStr, model_validator

from reactorbench.schemas.base import ContractModel, canonical_sha256

Sha256 = Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{64}$")]
SourceCommit = Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{7,64}$")]
Probability = Annotated[StrictFloat, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
FiniteFloat = Annotated[StrictFloat, Field(allow_inf_nan=False)]
NonNegativeSupport = Annotated[StrictInt, Field(ge=0)]
PositiveSupport = Annotated[StrictInt, Field(ge=1)]
Confidence95 = Annotated[StrictFloat, Field(ge=0.95, le=0.95, allow_inf_nan=False)]


class RemediationVersion(StrEnum):
    """Versions whose development evidence is accepted by this contract."""

    V03 = "0.3.0"
    V04 = "0.4.0"


class DevelopmentView(StrEnum):
    """Closed development-only view inventory.

    These names deliberately exclude final-test and golden-suite concepts so an
    acceptance packet cannot silently relabel either as development evidence.
    """

    IID_VALIDATION = "iid_validation"
    RENDERER_SHADOW = "renderer_shadow"
    COMPONENT_ROLE_SHADOW = "component_role_shadow"
    SEVERITY_SHADOW = "severity_shadow"
    COMPOSITION_SHADOW = "composition_shadow"
    COUNTERFACTUAL_SHADOW = "counterfactual_shadow"
    NOISE_SHADOW = "noise_shadow"


REQUIRED_V04_SHADOW_VIEWS: tuple[DevelopmentView, ...] = (
    DevelopmentView.RENDERER_SHADOW,
    DevelopmentView.COMPONENT_ROLE_SHADOW,
    DevelopmentView.SEVERITY_SHADOW,
    DevelopmentView.COMPOSITION_SHADOW,
    DevelopmentView.COUNTERFACTUAL_SHADOW,
    DevelopmentView.NOISE_SHADOW,
)


class MetricEstimate(ContractModel):
    """A supported point estimate and its preregistered bootstrap interval."""

    support: NonNegativeSupport
    estimate: Probability
    interval_lower: Probability
    interval_upper: Probability
    confidence_level: Confidence95 = 0.95
    bootstrap_resamples: Literal[2000] = 2000
    bootstrap_seed: Literal[6602] = 6602

    @model_validator(mode="after")
    def interval_contains_estimate(self) -> MetricEstimate:
        if not self.interval_lower <= self.estimate <= self.interval_upper:
            raise ValueError("metric interval must contain its estimate")
        if self.support == 0 and (
            self.estimate != 0.0 or self.interval_lower != 0.0 or self.interval_upper != 0.0
        ):
            raise ValueError("unsupported metrics must use an explicit zero-valued N/A marker")
        return self


class DevelopmentArtifactBinding(ContractModel):
    """Exact evidence lineage for one development-view metric packet."""

    source_commit: SourceCommit
    config_sha256: Sha256
    dataset_manifest_sha256: Sha256
    tokenizer_manifest_sha256: Sha256
    output_contract_sha256: Sha256
    checkpoint_sha256: Sha256
    prediction_artifact_sha256: Sha256
    comparator_artifact_sha256: Sha256


class SemanticMetricSet(ContractModel):
    """All metrics required on each semantic development view."""

    constrained_parse_rate: MetricEstimate
    constrained_schema_validity_rate: MetricEstimate
    unconstrained_parse_rate: MetricEstimate
    unconstrained_schema_validity_rate: MetricEstimate
    fault_family_macro_f1: MetricEstimate
    strongest_fault_comparator_macro_f1: MetricEstimate
    next_action_macro_f1: MetricEstimate
    strongest_action_comparator_macro_f1: MetricEstimate
    continuation_macro_f1: MetricEstimate
    evidence_f1: MetricEstimate
    required_abstention_accuracy: MetricEstimate
    no_fault_false_positive_rate: MetricEstimate
    expected_calibration_error: MetricEstimate
    selective_risk_at_80_percent_coverage: MetricEstimate

    @model_validator(mode="after")
    def comparator_supports_match(self) -> SemanticMetricSet:
        if self.fault_family_macro_f1.support != self.strongest_fault_comparator_macro_f1.support:
            raise ValueError("fault model and comparator supports must match")
        if self.next_action_macro_f1.support != self.strongest_action_comparator_macro_f1.support:
            raise ValueError("action model and comparator supports must match")
        return self


class DevelopmentViewMetrics(ContractModel):
    """Strict, checksum-bound semantic evidence for one development view."""

    contract_version: RemediationVersion
    view: DevelopmentView
    sample_count: PositiveSupport
    artifacts: DevelopmentArtifactBinding
    metrics: SemanticMetricSet
    composition_score_interval: MetricEstimate | None
    checksum_sha256: Sha256

    @model_validator(mode="after")
    def version_support_and_checksum_are_valid(self) -> DevelopmentViewMetrics:
        if self.contract_version is RemediationVersion.V03:
            if self.view is not DevelopmentView.IID_VALIDATION:
                raise ValueError("v0.3 metric packets are limited to IID validation")
        elif self.view is DevelopmentView.IID_VALIDATION:
            raise ValueError("v0.4 metric packets must be development shadow views")

        full_support_metrics = (
            self.metrics.constrained_parse_rate,
            self.metrics.constrained_schema_validity_rate,
            self.metrics.unconstrained_parse_rate,
            self.metrics.unconstrained_schema_validity_rate,
            self.metrics.expected_calibration_error,
            self.metrics.selective_risk_at_80_percent_coverage,
        )
        if any(metric.support != self.sample_count for metric in full_support_metrics):
            raise ValueError("view-wide metric supports must equal sample_count")

        task_metrics = (
            self.metrics.fault_family_macro_f1,
            self.metrics.strongest_fault_comparator_macro_f1,
            self.metrics.next_action_macro_f1,
            self.metrics.strongest_action_comparator_macro_f1,
            self.metrics.continuation_macro_f1,
            self.metrics.evidence_f1,
            self.metrics.required_abstention_accuracy,
            self.metrics.no_fault_false_positive_rate,
        )
        if any(metric.support > self.sample_count for metric in task_metrics):
            raise ValueError("task metric support cannot exceed sample_count")

        is_composition = self.view is DevelopmentView.COMPOSITION_SHADOW
        if is_composition != (self.composition_score_interval is not None):
            raise ValueError("only the composition shadow view carries a composition interval")
        if (
            self.composition_score_interval is not None
            and self.composition_score_interval.support > self.sample_count
        ):
            raise ValueError("composition support cannot exceed sample_count")

        expected = canonical_sha256(
            self.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        if self.checksum_sha256 != expected:
            raise ValueError("development-view metric checksum mismatch")
        return self


def bind_development_view_metrics(
    *,
    contract_version: RemediationVersion,
    view: DevelopmentView,
    sample_count: int,
    artifacts: DevelopmentArtifactBinding,
    metrics: SemanticMetricSet,
    composition_score_interval: MetricEstimate | None = None,
) -> DevelopmentViewMetrics:
    """Validate and checksum one development-only metric packet."""

    draft = DevelopmentViewMetrics.model_construct(
        contract_version=contract_version,
        view=view,
        sample_count=sample_count,
        artifacts=artifacts,
        metrics=metrics,
        composition_score_interval=composition_score_interval,
        checksum_sha256="0" * 64,
    )
    checksum = canonical_sha256(
        draft.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
    )
    return DevelopmentViewMetrics(
        contract_version=contract_version,
        view=view,
        sample_count=sample_count,
        artifacts=artifacts,
        metrics=metrics,
        composition_score_interval=composition_score_interval,
        checksum_sha256=checksum,
    )


class AcceptanceThresholds(ContractModel):
    """Frozen v0.3/v0.4 development thresholds from the remediation plan."""

    constrained_parse_rate: Probability = 1.0
    constrained_schema_validity_rate: Probability = 1.0
    minimum_fault_comparator_margin: Probability = 0.02
    minimum_action_comparator_margin: Probability = 0.02
    minimum_continuation_macro_f1: Probability = 0.90
    minimum_evidence_f1: Probability = 0.70
    minimum_required_abstention_accuracy: Probability = 0.80
    maximum_no_fault_false_positive_rate: Probability = 0.10
    maximum_expected_calibration_error: Probability = 0.15
    maximum_selective_risk_at_80_percent_coverage: Probability = 0.20

    @model_validator(mode="after")
    def thresholds_remain_preregistered(self) -> AcceptanceThresholds:
        observed = (
            self.constrained_parse_rate,
            self.constrained_schema_validity_rate,
            self.minimum_fault_comparator_margin,
            self.minimum_action_comparator_margin,
            self.minimum_continuation_macro_f1,
            self.minimum_evidence_f1,
            self.minimum_required_abstention_accuracy,
            self.maximum_no_fault_false_positive_rate,
            self.maximum_expected_calibration_error,
            self.maximum_selective_risk_at_80_percent_coverage,
        )
        expected = (1.0, 1.0, 0.02, 0.02, 0.90, 0.70, 0.80, 0.10, 0.15, 0.20)
        if observed != expected:
            raise ValueError("acceptance thresholds must remain preregistered")
        return self


DEFAULT_ACCEPTANCE_THRESHOLDS = AcceptanceThresholds()


class CheckName(StrEnum):
    V03_ADVANCEMENT_PREREQUISITE = "v03_advancement_prerequisite"
    REQUIRED_SHADOW_VIEWS_PRESENT = "required_shadow_views_present"
    COMPOSITION_INTERVAL_REPORTED = "composition_interval_reported"
    CONSTRAINED_PARSE_RATE = "constrained_parse_rate"
    CONSTRAINED_SCHEMA_VALIDITY_RATE = "constrained_schema_validity_rate"
    FAULT_COMPARATOR_MARGIN = "fault_comparator_margin"
    ACTION_COMPARATOR_MARGIN = "action_comparator_margin"
    CONTINUATION_MACRO_F1 = "continuation_macro_f1"
    EVIDENCE_F1 = "evidence_f1"
    REQUIRED_ABSTENTION_ACCURACY = "required_abstention_accuracy"
    NO_FAULT_FALSE_POSITIVE_RATE = "no_fault_false_positive_rate"
    EXPECTED_CALIBRATION_ERROR = "expected_calibration_error"
    SELECTIVE_RISK_AT_80_PERCENT_COVERAGE = "selective_risk_at_80_percent_coverage"


class CheckRelation(StrEnum):
    AT_LEAST = "at_least"
    AT_MOST = "at_most"
    EXACTLY = "exactly"


class CheckAggregation(StrEnum):
    PROGRAM = "program"
    VIEW = "view"
    WORST_SHADOW = "worst_shadow"


class AcceptanceCheck(ContractModel):
    """One named, mechanically verifiable gate check."""

    name: CheckName
    aggregation: CheckAggregation
    contributing_view: DevelopmentView | None
    relation: CheckRelation
    observed: FiniteFloat
    required: FiniteFloat
    passed: StrictBool

    @model_validator(mode="after")
    def result_matches_values(self) -> AcceptanceCheck:
        if self.aggregation is CheckAggregation.PROGRAM:
            if self.contributing_view is not None:
                raise ValueError("program checks cannot name a contributing view")
        elif self.contributing_view is None:
            raise ValueError("view and worst-shadow checks require a contributing view")
        expected = _comparison_passes(self.observed, self.required, self.relation)
        if self.passed is not expected:
            raise ValueError("acceptance-check result does not match its values")
        return self


SEMANTIC_CHECK_ORDER: tuple[CheckName, ...] = (
    CheckName.CONSTRAINED_PARSE_RATE,
    CheckName.CONSTRAINED_SCHEMA_VALIDITY_RATE,
    CheckName.FAULT_COMPARATOR_MARGIN,
    CheckName.ACTION_COMPARATOR_MARGIN,
    CheckName.CONTINUATION_MACRO_F1,
    CheckName.EVIDENCE_F1,
    CheckName.REQUIRED_ABSTENTION_ACCURACY,
    CheckName.NO_FAULT_FALSE_POSITIVE_RATE,
    CheckName.EXPECTED_CALIBRATION_ERROR,
    CheckName.SELECTIVE_RISK_AT_80_PERCENT_COVERAGE,
)


def _comparison_passes(observed: float, required: float, relation: CheckRelation) -> bool:
    if not math.isfinite(observed) or not math.isfinite(required):
        return False
    observed_decimal = Decimal(str(observed))
    required_decimal = Decimal(str(required))
    if relation is CheckRelation.AT_LEAST:
        return observed_decimal >= required_decimal
    if relation is CheckRelation.AT_MOST:
        return observed_decimal <= required_decimal
    return observed_decimal == required_decimal


def _check(
    *,
    name: CheckName,
    aggregation: CheckAggregation,
    contributing_view: DevelopmentView | None,
    relation: CheckRelation,
    observed: float,
    required: float,
) -> AcceptanceCheck:
    observed_float = float(observed)
    required_float = float(required)
    return AcceptanceCheck(
        name=name,
        aggregation=aggregation,
        contributing_view=contributing_view,
        relation=relation,
        observed=observed_float,
        required=required_float,
        passed=_comparison_passes(observed_float, required_float, relation),
    )


def _margin(model_score: float, comparator_score: float) -> float:
    return float(Decimal(str(model_score)) - Decimal(str(comparator_score)))


def _semantic_checks(
    view_metrics: DevelopmentViewMetrics,
    thresholds: AcceptanceThresholds,
    *,
    aggregation: CheckAggregation,
) -> tuple[AcceptanceCheck, ...]:
    metrics = view_metrics.metrics
    view = view_metrics.view
    candidates = (
        _check(
            name=CheckName.CONSTRAINED_PARSE_RATE,
            aggregation=aggregation,
            contributing_view=view,
            relation=CheckRelation.EXACTLY,
            observed=metrics.constrained_parse_rate.estimate,
            required=thresholds.constrained_parse_rate,
        ),
        _check(
            name=CheckName.CONSTRAINED_SCHEMA_VALIDITY_RATE,
            aggregation=aggregation,
            contributing_view=view,
            relation=CheckRelation.EXACTLY,
            observed=metrics.constrained_schema_validity_rate.estimate,
            required=thresholds.constrained_schema_validity_rate,
        ),
        _check(
            name=CheckName.FAULT_COMPARATOR_MARGIN,
            aggregation=aggregation,
            contributing_view=view,
            relation=CheckRelation.AT_LEAST,
            observed=_margin(
                metrics.fault_family_macro_f1.estimate,
                metrics.strongest_fault_comparator_macro_f1.estimate,
            ),
            required=thresholds.minimum_fault_comparator_margin,
        ),
        _check(
            name=CheckName.ACTION_COMPARATOR_MARGIN,
            aggregation=aggregation,
            contributing_view=view,
            relation=CheckRelation.AT_LEAST,
            observed=_margin(
                metrics.next_action_macro_f1.estimate,
                metrics.strongest_action_comparator_macro_f1.estimate,
            ),
            required=thresholds.minimum_action_comparator_margin,
        ),
        _check(
            name=CheckName.CONTINUATION_MACRO_F1,
            aggregation=aggregation,
            contributing_view=view,
            relation=CheckRelation.AT_LEAST,
            observed=metrics.continuation_macro_f1.estimate,
            required=thresholds.minimum_continuation_macro_f1,
        ),
        _check(
            name=CheckName.EVIDENCE_F1,
            aggregation=aggregation,
            contributing_view=view,
            relation=CheckRelation.AT_LEAST,
            observed=metrics.evidence_f1.estimate,
            required=thresholds.minimum_evidence_f1,
        ),
        _check(
            name=CheckName.REQUIRED_ABSTENTION_ACCURACY,
            aggregation=aggregation,
            contributing_view=view,
            relation=CheckRelation.AT_LEAST,
            observed=metrics.required_abstention_accuracy.estimate,
            required=thresholds.minimum_required_abstention_accuracy,
        ),
        _check(
            name=CheckName.NO_FAULT_FALSE_POSITIVE_RATE,
            aggregation=aggregation,
            contributing_view=view,
            relation=CheckRelation.AT_MOST,
            observed=metrics.no_fault_false_positive_rate.estimate,
            required=thresholds.maximum_no_fault_false_positive_rate,
        ),
        _check(
            name=CheckName.EXPECTED_CALIBRATION_ERROR,
            aggregation=aggregation,
            contributing_view=view,
            relation=CheckRelation.AT_MOST,
            observed=metrics.expected_calibration_error.estimate,
            required=thresholds.maximum_expected_calibration_error,
        ),
        _check(
            name=CheckName.SELECTIVE_RISK_AT_80_PERCENT_COVERAGE,
            aggregation=aggregation,
            contributing_view=view,
            relation=CheckRelation.AT_MOST,
            observed=metrics.selective_risk_at_80_percent_coverage.estimate,
            required=thresholds.maximum_selective_risk_at_80_percent_coverage,
        ),
    )
    supports = (
        metrics.constrained_parse_rate.support,
        metrics.constrained_schema_validity_rate.support,
        metrics.fault_family_macro_f1.support,
        metrics.next_action_macro_f1.support,
        metrics.continuation_macro_f1.support,
        metrics.evidence_f1.support,
        metrics.required_abstention_accuracy.support,
        metrics.no_fault_false_positive_rate.support,
        metrics.expected_calibration_error.support,
        metrics.selective_risk_at_80_percent_coverage.support,
    )
    return tuple(check for check, support in zip(candidates, supports, strict=True) if support > 0)


class V03AcceptanceResult(ContractModel):
    """Checksum-bound v0.3 development-gate result."""

    result_version: Literal["0.3.0"]
    thresholds: AcceptanceThresholds
    view_metrics: DevelopmentViewMetrics
    checks: tuple[AcceptanceCheck, ...]
    advancement_allowed: StrictBool
    checksum_sha256: Sha256

    @model_validator(mode="after")
    def gate_and_checksum_are_valid(self) -> V03AcceptanceResult:
        if self.view_metrics.contract_version is not RemediationVersion.V03:
            raise ValueError("v0.3 result requires v0.3 metrics")
        if any(
            metric.support == 0
            for metric in (
                self.view_metrics.metrics.fault_family_macro_f1,
                self.view_metrics.metrics.next_action_macro_f1,
                self.view_metrics.metrics.continuation_macro_f1,
                self.view_metrics.metrics.evidence_f1,
                self.view_metrics.metrics.required_abstention_accuracy,
                self.view_metrics.metrics.no_fault_false_positive_rate,
            )
        ):
            raise ValueError("v0.3 IID validation must support every behavioral gate")
        if tuple(check.name for check in self.checks) != SEMANTIC_CHECK_ORDER:
            raise ValueError("v0.3 checks must use the canonical order")
        if any(
            check.aggregation is not CheckAggregation.VIEW
            or check.contributing_view is not DevelopmentView.IID_VALIDATION
            for check in self.checks
        ):
            raise ValueError("v0.3 checks must be IID-validation view checks")
        if self.advancement_allowed is not all(check.passed for check in self.checks):
            raise ValueError("v0.3 advancement does not match its checks")
        expected = canonical_sha256(
            self.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        if self.checksum_sha256 != expected:
            raise ValueError("v0.3 acceptance-result checksum mismatch")
        return self


def evaluate_v03_acceptance(
    view_metrics: DevelopmentViewMetrics,
    *,
    thresholds: AcceptanceThresholds = DEFAULT_ACCEPTANCE_THRESHOLDS,
) -> V03AcceptanceResult:
    """Evaluate the frozen v0.3 IID-development gate deterministically."""

    if type(view_metrics) is not DevelopmentViewMetrics:
        raise TypeError("v0.3 evaluation requires exact DevelopmentViewMetrics")
    if type(thresholds) is not AcceptanceThresholds:
        raise TypeError("v0.3 evaluation requires exact AcceptanceThresholds")
    if view_metrics.contract_version is not RemediationVersion.V03:
        raise ValueError("v0.3 evaluation requires a v0.3 metric packet")
    checks = _semantic_checks(view_metrics, thresholds, aggregation=CheckAggregation.VIEW)
    advancement_allowed = all(check.passed for check in checks)
    draft = V03AcceptanceResult.model_construct(
        result_version="0.3.0",
        thresholds=thresholds,
        view_metrics=view_metrics,
        checks=checks,
        advancement_allowed=advancement_allowed,
        checksum_sha256="0" * 64,
    )
    checksum = canonical_sha256(
        draft.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
    )
    return V03AcceptanceResult(
        result_version="0.3.0",
        thresholds=thresholds,
        view_metrics=view_metrics,
        checks=checks,
        advancement_allowed=advancement_allowed,
        checksum_sha256=checksum,
    )


class ShadowViewEvaluation(ContractModel):
    """Named checks for one v0.4 shadow view."""

    view: DevelopmentView
    metrics_checksum_sha256: Sha256
    performance_thresholded: StrictBool
    supported_check_names: tuple[CheckName, ...]
    checks: tuple[AcceptanceCheck, ...]
    passed: StrictBool

    @model_validator(mode="after")
    def check_set_is_valid(self) -> ShadowViewEvaluation:
        if self.view is DevelopmentView.IID_VALIDATION:
            raise ValueError("IID validation is not a v0.4 shadow view")
        names = tuple(check.name for check in self.checks)
        if self.view is DevelopmentView.COMPOSITION_SHADOW:
            if names != SEMANTIC_CHECK_ORDER[:2]:
                raise ValueError("composition shadow must gate only parse and schema validity")
        elif names[:2] != SEMANTIC_CHECK_ORDER[:2] or names != tuple(
            name for name in SEMANTIC_CHECK_ORDER if name in set(names)
        ):
            raise ValueError("shadow-view checks must be a canonical supported-metric subset")
        if self.supported_check_names != names:
            raise ValueError("shadow-view checks differ from the bound supported-metric set")
        if self.performance_thresholded is (self.view is DevelopmentView.COMPOSITION_SHADOW):
            raise ValueError("composition performance must be reported without a threshold")
        if any(
            check.aggregation is not CheckAggregation.VIEW
            or check.contributing_view is not self.view
            for check in self.checks
        ):
            raise ValueError("shadow-view checks must identify their own view")
        if self.passed is not all(check.passed for check in self.checks):
            raise ValueError("shadow-view result does not match its checks")
        return self


class V04AcceptanceResult(ContractModel):
    """Worst-split-aware v0.4 development-gate result."""

    result_version: Literal["0.4.0"]
    thresholds: AcceptanceThresholds
    v03_result: V03AcceptanceResult
    required_shadow_views: tuple[DevelopmentView, ...]
    shadow_view_metrics: tuple[DevelopmentViewMetrics, ...]
    program_checks: tuple[AcceptanceCheck, ...]
    view_evaluations: tuple[ShadowViewEvaluation, ...]
    worst_split_checks: tuple[AcceptanceCheck, ...]
    composition_score_interval: MetricEstimate | None
    advancement_allowed: StrictBool
    checksum_sha256: Sha256

    @model_validator(mode="after")
    def gate_and_checksum_are_valid(self) -> V04AcceptanceResult:
        if self.required_shadow_views != REQUIRED_V04_SHADOW_VIEWS:
            raise ValueError("v0.4 required shadow views must remain preregistered")
        observed_views = tuple(packet.view for packet in self.shadow_view_metrics)
        order = {view: index for index, view in enumerate(REQUIRED_V04_SHADOW_VIEWS)}
        if len(observed_views) != len(set(observed_views)):
            raise ValueError("v0.4 shadow views must be unique")
        if any(view not in order for view in observed_views):
            raise ValueError("v0.4 received an unsupported development view")
        if observed_views != tuple(sorted(observed_views, key=order.__getitem__)):
            raise ValueError("v0.4 shadow views must use preregistered order")
        if any(
            packet.contract_version is not RemediationVersion.V04
            for packet in self.shadow_view_metrics
        ):
            raise ValueError("v0.4 result requires v0.4 metric packets")

        if tuple(check.name for check in self.program_checks) != (
            CheckName.V03_ADVANCEMENT_PREREQUISITE,
            CheckName.REQUIRED_SHADOW_VIEWS_PRESENT,
            CheckName.COMPOSITION_INTERVAL_REPORTED,
        ):
            raise ValueError("v0.4 program checks must use the canonical order")
        if any(check.aggregation is not CheckAggregation.PROGRAM for check in self.program_checks):
            raise ValueError("v0.4 program checks must use program aggregation")

        if tuple(evaluation.view for evaluation in self.view_evaluations) != observed_views:
            raise ValueError("v0.4 view evaluations must align with metric packets")
        for packet, evaluation in zip(self.shadow_view_metrics, self.view_evaluations, strict=True):
            if evaluation.metrics_checksum_sha256 != packet.checksum_sha256:
                raise ValueError("v0.4 view evaluation metric checksum mismatch")
            expected_checks = _semantic_checks(
                packet,
                self.thresholds,
                aggregation=CheckAggregation.VIEW,
            )
            if packet.view is DevelopmentView.COMPOSITION_SHADOW:
                expected_checks = expected_checks[:2]
            if evaluation.supported_check_names != tuple(check.name for check in expected_checks):
                raise ValueError("v0.4 view evaluation support differs from its metrics")

        composition_packets = tuple(
            packet
            for packet in self.shadow_view_metrics
            if packet.view is DevelopmentView.COMPOSITION_SHADOW
        )
        expected_composition = (
            None if not composition_packets else composition_packets[0].composition_score_interval
        )
        if self.composition_score_interval != expected_composition:
            raise ValueError("v0.4 composition report does not match its metric packet")

        if any(
            check.aggregation is not CheckAggregation.WORST_SHADOW
            or check.contributing_view is None
            for check in self.worst_split_checks
        ):
            raise ValueError("v0.4 worst-split checks must identify a contributing view")

        common_fields = (
            "source_commit",
            "config_sha256",
            "tokenizer_manifest_sha256",
            "output_contract_sha256",
            "checkpoint_sha256",
        )
        if self.shadow_view_metrics:
            reference = self.shadow_view_metrics[0].artifacts
            if any(
                getattr(packet.artifacts, field) != getattr(reference, field)
                for packet in self.shadow_view_metrics[1:]
                for field in common_fields
            ):
                raise ValueError("v0.4 shadow views must evaluate one frozen candidate")

        all_checks_pass = (
            all(check.passed for check in self.program_checks)
            and all(evaluation.passed for evaluation in self.view_evaluations)
            and all(check.passed for check in self.worst_split_checks)
        )
        if self.advancement_allowed is not all_checks_pass:
            raise ValueError("v0.4 advancement does not match its checks")
        expected = canonical_sha256(
            self.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        if self.checksum_sha256 != expected:
            raise ValueError("v0.4 acceptance-result checksum mismatch")
        return self


def _shadow_evaluation(packet: DevelopmentViewMetrics) -> ShadowViewEvaluation:
    semantic_checks = _semantic_checks(
        packet,
        DEFAULT_ACCEPTANCE_THRESHOLDS,
        aggregation=CheckAggregation.VIEW,
    )
    checks = (
        semantic_checks[:2]
        if packet.view is DevelopmentView.COMPOSITION_SHADOW
        else semantic_checks
    )
    return ShadowViewEvaluation(
        view=packet.view,
        metrics_checksum_sha256=packet.checksum_sha256,
        performance_thresholded=packet.view is not DevelopmentView.COMPOSITION_SHADOW,
        supported_check_names=tuple(check.name for check in checks),
        checks=checks,
        passed=all(check.passed for check in checks),
    )


def _worst_split_checks(
    packets: tuple[DevelopmentViewMetrics, ...],
    thresholds: AcceptanceThresholds,
) -> tuple[AcceptanceCheck, ...]:
    if not packets:
        return ()
    per_view = {
        packet.view: {
            check.name: check
            for check in _semantic_checks(
                packet,
                thresholds,
                aggregation=CheckAggregation.WORST_SHADOW,
            )
        }
        for packet in packets
    }
    ordered_checks: list[AcceptanceCheck] = []
    for name in SEMANTIC_CHECK_ORDER:
        candidates = tuple(
            checks[name]
            for view, checks in per_view.items()
            if name in checks
            and (
                name
                in {
                    CheckName.CONSTRAINED_PARSE_RATE,
                    CheckName.CONSTRAINED_SCHEMA_VALIDITY_RATE,
                }
                or view is not DevelopmentView.COMPOSITION_SHADOW
            )
        )
        if not candidates:
            continue
        relation = candidates[0].relation
        if relation is CheckRelation.AT_MOST:
            worst = max(candidates, key=lambda check: check.observed)
        else:
            worst = min(candidates, key=lambda check: check.observed)
        ordered_checks.append(worst)
    return tuple(ordered_checks)


def evaluate_v04_acceptance(
    v03_result: V03AcceptanceResult,
    shadow_view_metrics: tuple[DevelopmentViewMetrics, ...],
    *,
    thresholds: AcceptanceThresholds = DEFAULT_ACCEPTANCE_THRESHOLDS,
) -> V04AcceptanceResult:
    """Evaluate v0.4 using every view and each metric's worst shadow split.

    Missing required views produce a failed coverage check instead of being silently
    ignored. Composition performance is carried with its interval but receives no
    score threshold; constrained parse and schema validity still apply to that view.
    """

    if type(v03_result) is not V03AcceptanceResult:
        raise TypeError("v0.4 evaluation requires an exact V03AcceptanceResult")
    if type(shadow_view_metrics) is not tuple or any(
        type(packet) is not DevelopmentViewMetrics for packet in shadow_view_metrics
    ):
        raise TypeError("v0.4 shadow metrics must be an exact metric-packet tuple")
    if type(thresholds) is not AcceptanceThresholds:
        raise TypeError("v0.4 evaluation requires exact AcceptanceThresholds")

    observed_views = tuple(packet.view for packet in shadow_view_metrics)
    if len(observed_views) != len(set(observed_views)):
        raise ValueError("v0.4 shadow metric packets must be unique by view")
    required_order = {view: index for index, view in enumerate(REQUIRED_V04_SHADOW_VIEWS)}
    if any(view not in required_order for view in observed_views):
        raise ValueError("v0.4 evaluation accepts only preregistered shadow views")
    ordered_packets = tuple(
        sorted(shadow_view_metrics, key=lambda packet: required_order[packet.view])
    )
    if any(packet.contract_version is not RemediationVersion.V04 for packet in ordered_packets):
        raise ValueError("v0.4 evaluation requires v0.4 metric packets")

    composition_packet = next(
        (packet for packet in ordered_packets if packet.view is DevelopmentView.COMPOSITION_SHADOW),
        None,
    )
    program_checks = (
        _check(
            name=CheckName.V03_ADVANCEMENT_PREREQUISITE,
            aggregation=CheckAggregation.PROGRAM,
            contributing_view=None,
            relation=CheckRelation.EXACTLY,
            observed=float(v03_result.advancement_allowed),
            required=1.0,
        ),
        _check(
            name=CheckName.REQUIRED_SHADOW_VIEWS_PRESENT,
            aggregation=CheckAggregation.PROGRAM,
            contributing_view=None,
            relation=CheckRelation.EXACTLY,
            observed=float(len(observed_views)),
            required=float(len(REQUIRED_V04_SHADOW_VIEWS)),
        ),
        _check(
            name=CheckName.COMPOSITION_INTERVAL_REPORTED,
            aggregation=CheckAggregation.PROGRAM,
            contributing_view=None,
            relation=CheckRelation.EXACTLY,
            observed=float(
                composition_packet is not None
                and composition_packet.composition_score_interval is not None
            ),
            required=1.0,
        ),
    )
    evaluations = tuple(_shadow_evaluation(packet) for packet in ordered_packets)
    worst_checks = _worst_split_checks(ordered_packets, thresholds)
    advancement_allowed = (
        all(check.passed for check in program_checks)
        and all(evaluation.passed for evaluation in evaluations)
        and all(check.passed for check in worst_checks)
    )
    composition_interval = (
        None if composition_packet is None else composition_packet.composition_score_interval
    )
    draft = V04AcceptanceResult.model_construct(
        result_version="0.4.0",
        thresholds=thresholds,
        v03_result=v03_result,
        required_shadow_views=REQUIRED_V04_SHADOW_VIEWS,
        shadow_view_metrics=ordered_packets,
        program_checks=program_checks,
        view_evaluations=evaluations,
        worst_split_checks=worst_checks,
        composition_score_interval=composition_interval,
        advancement_allowed=advancement_allowed,
        checksum_sha256="0" * 64,
    )
    checksum = canonical_sha256(
        draft.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
    )
    return V04AcceptanceResult(
        result_version="0.4.0",
        thresholds=thresholds,
        v03_result=v03_result,
        required_shadow_views=REQUIRED_V04_SHADOW_VIEWS,
        shadow_view_metrics=ordered_packets,
        program_checks=program_checks,
        view_evaluations=evaluations,
        worst_split_checks=worst_checks,
        composition_score_interval=composition_interval,
        advancement_allowed=advancement_allowed,
        checksum_sha256=checksum,
    )


__all__ = [
    "DEFAULT_ACCEPTANCE_THRESHOLDS",
    "REQUIRED_V04_SHADOW_VIEWS",
    "AcceptanceCheck",
    "AcceptanceThresholds",
    "CheckAggregation",
    "CheckName",
    "CheckRelation",
    "DevelopmentArtifactBinding",
    "DevelopmentView",
    "DevelopmentViewMetrics",
    "MetricEstimate",
    "RemediationVersion",
    "SemanticMetricSet",
    "ShadowViewEvaluation",
    "V03AcceptanceResult",
    "V04AcceptanceResult",
    "bind_development_view_metrics",
    "evaluate_v03_acceptance",
    "evaluate_v04_acceptance",
]
