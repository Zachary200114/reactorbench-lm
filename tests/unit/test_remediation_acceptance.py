from __future__ import annotations

from typing import Any, cast

import pytest
from pydantic import ValidationError

from reactorbench.remediation.acceptance import (
    DEFAULT_ACCEPTANCE_THRESHOLDS,
    REQUIRED_V04_SHADOW_VIEWS,
    AcceptanceCheck,
    AcceptanceThresholds,
    CheckAggregation,
    CheckName,
    CheckRelation,
    DevelopmentArtifactBinding,
    DevelopmentView,
    DevelopmentViewMetrics,
    MetricEstimate,
    RemediationVersion,
    SemanticMetricSet,
    ShadowViewEvaluation,
    V03AcceptanceResult,
    V04AcceptanceResult,
    bind_development_view_metrics,
    evaluate_v03_acceptance,
    evaluate_v04_acceptance,
)


def _estimate(value: float, *, support: int = 100) -> MetricEstimate:
    return MetricEstimate(
        support=support,
        estimate=value,
        interval_lower=max(0.0, value - 0.01),
        interval_upper=min(1.0, value + 0.01),
        confidence_level=0.95,
        bootstrap_resamples=2000,
        bootstrap_seed=6602,
    )


def _metric_set(**overrides: float) -> SemanticMetricSet:
    values = {
        "constrained_parse_rate": 1.0,
        "constrained_schema_validity_rate": 1.0,
        "unconstrained_parse_rate": 0.0,
        "unconstrained_schema_validity_rate": 0.0,
        "fault_family_macro_f1": 0.72,
        "strongest_fault_comparator_macro_f1": 0.70,
        "next_action_macro_f1": 0.67,
        "strongest_action_comparator_macro_f1": 0.65,
        "continuation_macro_f1": 0.90,
        "evidence_f1": 0.70,
        "required_abstention_accuracy": 0.80,
        "no_fault_false_positive_rate": 0.10,
        "expected_calibration_error": 0.15,
        "selective_risk_at_80_percent_coverage": 0.20,
    }
    values.update(overrides)
    return SemanticMetricSet(**{name: _estimate(value) for name, value in values.items()})


def _artifacts(
    *, checkpoint: str = "e", tokenizer: str = "2", config: str = "b"
) -> DevelopmentArtifactBinding:
    return DevelopmentArtifactBinding(
        source_commit="a" * 40,
        config_sha256=config * 64,
        dataset_manifest_sha256="c" * 64,
        tokenizer_manifest_sha256=tokenizer * 64,
        output_contract_sha256="d" * 64,
        checkpoint_sha256=checkpoint * 64,
        prediction_artifact_sha256="f" * 64,
        comparator_artifact_sha256="1" * 64,
    )


def _packet(
    view: DevelopmentView,
    *,
    version: RemediationVersion,
    checkpoint: str = "e",
    tokenizer: str = "2",
    config: str = "b",
    composition_score: float = 0.0,
    **metric_overrides: float,
) -> DevelopmentViewMetrics:
    composition = (
        _estimate(composition_score) if view is DevelopmentView.COMPOSITION_SHADOW else None
    )
    return bind_development_view_metrics(
        contract_version=version,
        view=view,
        sample_count=100,
        artifacts=_artifacts(checkpoint=checkpoint, tokenizer=tokenizer, config=config),
        metrics=_metric_set(**metric_overrides),
        composition_score_interval=composition,
    )


def _passing_v03(
    *, checkpoint: str = "e", tokenizer: str = "2", config: str = "b"
) -> V03AcceptanceResult:
    return evaluate_v03_acceptance(
        _packet(
            DevelopmentView.IID_VALIDATION,
            version=RemediationVersion.V03,
            checkpoint=checkpoint,
            tokenizer=tokenizer,
            config=config,
        )
    )


def _shadow_packets() -> tuple[DevelopmentViewMetrics, ...]:
    return tuple(
        _packet(view, version=RemediationVersion.V04) for view in REQUIRED_V04_SHADOW_VIEWS
    )


def test_v03_exact_boundaries_pass_and_unconstrained_rates_are_report_only() -> None:
    packet = _packet(
        DevelopmentView.IID_VALIDATION,
        version=RemediationVersion.V03,
        unconstrained_parse_rate=0.0,
        unconstrained_schema_validity_rate=0.0,
    )
    first = evaluate_v03_acceptance(packet)
    second = evaluate_v03_acceptance(packet)

    assert first == second
    assert first.advancement_allowed
    assert tuple(check.name for check in first.checks) == (
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
    assert first.checks[2].observed == 0.02


def test_v03_comparator_margin_failure_blocks_advancement() -> None:
    packet = _packet(
        DevelopmentView.IID_VALIDATION,
        version=RemediationVersion.V03,
        fault_family_macro_f1=0.719,
    )
    result = evaluate_v03_acceptance(packet)

    assert not result.advancement_allowed
    failed = tuple(check.name for check in result.checks if not check.passed)
    assert failed == (CheckName.FAULT_COMPARATOR_MARGIN,)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_metric_contract_fails_closed_on_nonfinite_values(invalid: float) -> None:
    with pytest.raises(ValidationError):
        MetricEstimate(
            support=1,
            estimate=invalid,
            interval_lower=0.0,
            interval_upper=1.0,
        )


def test_metric_contract_rejects_empty_support_and_inconsistent_intervals() -> None:
    with pytest.raises(ValidationError, match="zero-valued N/A marker"):
        MetricEstimate(
            support=0,
            estimate=0.5,
            interval_lower=0.4,
            interval_upper=0.6,
        )
    assert (
        MetricEstimate(
            support=0,
            estimate=0.0,
            interval_lower=0.0,
            interval_upper=0.0,
        ).support
        == 0
    )
    with pytest.raises(ValidationError, match="must contain"):
        MetricEstimate(
            support=1,
            estimate=0.5,
            interval_lower=0.6,
            interval_upper=0.7,
        )


def test_metric_and_result_checksums_reject_tampering() -> None:
    packet = _packet(
        DevelopmentView.IID_VALIDATION,
        version=RemediationVersion.V03,
    )
    packet_payload = packet.model_dump(mode="python", round_trip=True)
    artifacts = cast(dict[str, Any], packet_payload["artifacts"])
    artifacts["prediction_artifact_sha256"] = "9" * 64
    with pytest.raises(ValidationError, match="checksum"):
        DevelopmentViewMetrics.model_validate(packet_payload)

    result = evaluate_v03_acceptance(packet)
    result_payload = result.model_dump(mode="python", round_trip=True)
    result_payload["advancement_allowed"] = False
    with pytest.raises(ValidationError, match=r"advancement|checksum"):
        V03AcceptanceResult.model_validate(result_payload)


def test_contracts_reject_coercion_unknown_fields_and_support_mismatch() -> None:
    payload = _estimate(0.5).model_dump(mode="python", round_trip=True)
    payload["support"] = "100"
    with pytest.raises(ValidationError, match="int_type"):
        MetricEstimate.model_validate(payload)

    payload = _estimate(0.5).model_dump(mode="python", round_trip=True)
    payload["unknown"] = True
    with pytest.raises(ValidationError, match="extra_forbidden"):
        MetricEstimate.model_validate(payload)

    metrics_payload = _metric_set().model_dump(mode="python", round_trip=True)
    comparator = cast(dict[str, Any], metrics_payload["strongest_fault_comparator_macro_f1"])
    comparator["support"] = 99
    with pytest.raises(ValidationError, match="supports must match"):
        SemanticMetricSet.model_validate(metrics_payload)

    metrics_payload = _metric_set().model_dump(mode="python", round_trip=True)
    action_comparator = cast(
        dict[str, Any], metrics_payload["strongest_action_comparator_macro_f1"]
    )
    action_comparator["support"] = 99
    with pytest.raises(ValidationError, match="action model and comparator"):
        SemanticMetricSet.model_validate(metrics_payload)


def test_view_contract_rejects_version_support_and_composition_drift() -> None:
    with pytest.raises(ValidationError, match=r"v0\.3 metric packets"):
        bind_development_view_metrics(
            contract_version=RemediationVersion.V03,
            view=DevelopmentView.RENDERER_SHADOW,
            sample_count=100,
            artifacts=_artifacts(),
            metrics=_metric_set(),
        )
    with pytest.raises(ValidationError, match=r"v0\.4 metric packets"):
        bind_development_view_metrics(
            contract_version=RemediationVersion.V04,
            view=DevelopmentView.IID_VALIDATION,
            sample_count=100,
            artifacts=_artifacts(),
            metrics=_metric_set(),
        )

    payload = _metric_set().model_dump(mode="python", round_trip=True)
    constrained_parse = cast(dict[str, Any], payload["constrained_parse_rate"])
    constrained_parse["support"] = 99
    with pytest.raises(ValidationError, match="view-wide metric supports"):
        bind_development_view_metrics(
            contract_version=RemediationVersion.V03,
            view=DevelopmentView.IID_VALIDATION,
            sample_count=100,
            artifacts=_artifacts(),
            metrics=SemanticMetricSet.model_validate(payload),
        )

    payload = _metric_set().model_dump(mode="python", round_trip=True)
    for field in ("fault_family_macro_f1", "strongest_fault_comparator_macro_f1"):
        cast(dict[str, Any], payload[field])["support"] = 101
    with pytest.raises(ValidationError, match="task metric support"):
        bind_development_view_metrics(
            contract_version=RemediationVersion.V03,
            view=DevelopmentView.IID_VALIDATION,
            sample_count=100,
            artifacts=_artifacts(),
            metrics=SemanticMetricSet.model_validate(payload),
        )

    with pytest.raises(ValidationError, match="only the composition"):
        bind_development_view_metrics(
            contract_version=RemediationVersion.V04,
            view=DevelopmentView.COMPOSITION_SHADOW,
            sample_count=100,
            artifacts=_artifacts(),
            metrics=_metric_set(),
        )
    with pytest.raises(ValidationError, match="composition support"):
        bind_development_view_metrics(
            contract_version=RemediationVersion.V04,
            view=DevelopmentView.COMPOSITION_SHADOW,
            sample_count=100,
            artifacts=_artifacts(),
            metrics=_metric_set(),
            composition_score_interval=_estimate(0.5, support=101),
        )


def test_acceptance_check_rejects_invalid_scope_and_claimed_result() -> None:
    common: dict[str, object] = {
        "name": CheckName.EVIDENCE_F1,
        "relation": CheckRelation.AT_LEAST,
        "observed": 0.7,
        "required": 0.7,
        "passed": True,
    }
    with pytest.raises(ValidationError, match="program checks cannot"):
        AcceptanceCheck.model_validate(
            {
                **common,
                "aggregation": CheckAggregation.PROGRAM,
                "contributing_view": DevelopmentView.RENDERER_SHADOW,
            }
        )
    with pytest.raises(ValidationError, match="require a contributing view"):
        AcceptanceCheck.model_validate(
            {
                **common,
                "aggregation": CheckAggregation.VIEW,
                "contributing_view": None,
            }
        )
    with pytest.raises(ValidationError, match="does not match"):
        AcceptanceCheck.model_validate(
            {
                **common,
                "passed": False,
                "aggregation": CheckAggregation.VIEW,
                "contributing_view": DevelopmentView.RENDERER_SHADOW,
            }
        )


def test_thresholds_cannot_be_relaxed_after_preregistration() -> None:
    payload = DEFAULT_ACCEPTANCE_THRESHOLDS.model_dump(mode="python", round_trip=True)
    payload["minimum_continuation_macro_f1"] = 0.89
    with pytest.raises(ValidationError, match="must remain preregistered"):
        AcceptanceThresholds.model_validate(payload)


def test_v04_passes_all_views_and_does_not_threshold_composition_score() -> None:
    result = evaluate_v04_acceptance(_passing_v03(), _shadow_packets())

    assert result.advancement_allowed
    assert result.composition_score_interval is not None
    assert result.composition_score_interval.estimate == 0.0
    composition = next(
        evaluation
        for evaluation in result.view_evaluations
        if evaluation.view is DevelopmentView.COMPOSITION_SHADOW
    )
    assert not composition.performance_thresholded
    assert tuple(check.name for check in composition.checks) == (
        CheckName.CONSTRAINED_PARSE_RATE,
        CheckName.CONSTRAINED_SCHEMA_VALIDITY_RATE,
    )
    assert all(check.contributing_view is not None for check in result.worst_split_checks)


def test_v04_missing_required_view_fails_closed_with_named_coverage_check() -> None:
    result = evaluate_v04_acceptance(_passing_v03(), _shadow_packets()[:-1])

    assert not result.advancement_allowed
    coverage = next(
        check
        for check in result.program_checks
        if check.name is CheckName.REQUIRED_SHADOW_VIEWS_PRESENT
    )
    assert not coverage.passed
    assert coverage.observed == 5.0
    assert coverage.required == 6.0


def test_v04_requires_a_passing_v03_gate() -> None:
    failed_v03 = evaluate_v03_acceptance(
        _packet(
            DevelopmentView.IID_VALIDATION,
            version=RemediationVersion.V03,
            required_abstention_accuracy=0.79,
        )
    )
    result = evaluate_v04_acceptance(failed_v03, _shadow_packets())

    assert not result.advancement_allowed
    assert result.program_checks[0].name is CheckName.V03_ADVANCEMENT_PREREQUISITE
    assert not result.program_checks[0].passed


def test_v04_worst_split_exposes_and_blocks_the_failing_view() -> None:
    packets = list(_shadow_packets())
    noise_index = REQUIRED_V04_SHADOW_VIEWS.index(DevelopmentView.NOISE_SHADOW)
    packets[noise_index] = _packet(
        DevelopmentView.NOISE_SHADOW,
        version=RemediationVersion.V04,
        evidence_f1=0.69,
    )
    result = evaluate_v04_acceptance(_passing_v03(), tuple(packets))

    assert not result.advancement_allowed
    evidence = next(
        check for check in result.worst_split_checks if check.name is CheckName.EVIDENCE_F1
    )
    assert evidence.contributing_view is DevelopmentView.NOISE_SHADOW
    assert evidence.observed == 0.69
    assert not evidence.passed


def test_component_shadow_marks_absent_continuation_na_but_gates_supported_tasks() -> None:
    packets = list(_shadow_packets())
    component_index = REQUIRED_V04_SHADOW_VIEWS.index(DevelopmentView.COMPONENT_ROLE_SHADOW)

    metrics_payload = _metric_set().model_dump(mode="python", round_trip=True)
    metrics_payload["continuation_macro_f1"] = MetricEstimate(
        support=0,
        estimate=0.0,
        interval_lower=0.0,
        interval_upper=0.0,
    )
    component_metrics = SemanticMetricSet.model_validate(metrics_payload)
    packets[component_index] = bind_development_view_metrics(
        contract_version=RemediationVersion.V04,
        view=DevelopmentView.COMPONENT_ROLE_SHADOW,
        sample_count=100,
        artifacts=_artifacts(),
        metrics=component_metrics,
        composition_score_interval=None,
    )

    passing = evaluate_v04_acceptance(_passing_v03(), tuple(packets))
    component = passing.view_evaluations[component_index]

    assert passing.advancement_allowed
    assert CheckName.CONTINUATION_MACRO_F1 not in component.supported_check_names
    assert CheckName.FAULT_COMPARATOR_MARGIN in component.supported_check_names
    assert CheckName.ACTION_COMPARATOR_MARGIN in component.supported_check_names
    assert CheckName.EVIDENCE_F1 in component.supported_check_names

    failing_payload = component_metrics.model_dump(mode="python", round_trip=True)
    failing_payload["evidence_f1"] = _estimate(0.69)
    packets[component_index] = bind_development_view_metrics(
        contract_version=RemediationVersion.V04,
        view=DevelopmentView.COMPONENT_ROLE_SHADOW,
        sample_count=100,
        artifacts=_artifacts(),
        metrics=SemanticMetricSet.model_validate(failing_payload),
        composition_score_interval=None,
    )
    failing = evaluate_v04_acceptance(_passing_v03(), tuple(packets))
    component_failure = next(
        check
        for check in failing.view_evaluations[component_index].checks
        if check.name is CheckName.EVIDENCE_F1
    )

    assert not failing.advancement_allowed
    assert component_failure.contributing_view is DevelopmentView.COMPONENT_ROLE_SHADOW
    assert not component_failure.passed


def test_v04_composition_still_requires_perfect_constrained_validity() -> None:
    packets = list(_shadow_packets())
    composition_index = REQUIRED_V04_SHADOW_VIEWS.index(DevelopmentView.COMPOSITION_SHADOW)
    packets[composition_index] = _packet(
        DevelopmentView.COMPOSITION_SHADOW,
        version=RemediationVersion.V04,
        constrained_schema_validity_rate=0.99,
    )
    result = evaluate_v04_acceptance(_passing_v03(), tuple(packets))

    assert not result.advancement_allowed
    composition = result.view_evaluations[composition_index]
    assert not composition.passed
    assert composition.checks[1].name is CheckName.CONSTRAINED_SCHEMA_VALIDITY_RATE


def test_v04_rejects_mixed_candidate_bindings_and_result_tampering() -> None:
    packets = list(_shadow_packets())
    packets[-1] = _packet(
        DevelopmentView.NOISE_SHADOW,
        version=RemediationVersion.V04,
        checkpoint="2",
    )
    with pytest.raises(ValidationError, match="one frozen candidate"):
        evaluate_v04_acceptance(_passing_v03(), tuple(packets))

    packets = list(_shadow_packets())
    packets[-1] = _packet(
        DevelopmentView.NOISE_SHADOW,
        version=RemediationVersion.V04,
        tokenizer="3",
    )
    with pytest.raises(ValidationError, match="one frozen candidate"):
        evaluate_v04_acceptance(_passing_v03(), tuple(packets))

    for mismatched_iid in (
        _passing_v03(checkpoint="3"),
        _passing_v03(tokenizer="4"),
        _passing_v03(config="5"),
    ):
        with pytest.raises(ValidationError, match="one frozen candidate"):
            evaluate_v04_acceptance(mismatched_iid, _shadow_packets())

    result = evaluate_v04_acceptance(_passing_v03(), _shadow_packets())
    payload = result.model_dump(mode="python", round_trip=True)
    payload["checksum_sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="checksum"):
        V04AcceptanceResult.model_validate(payload)


def test_v04_rejects_duplicate_or_nonshadow_packets() -> None:
    packets = _shadow_packets()
    with pytest.raises(ValueError, match="unique"):
        evaluate_v04_acceptance(_passing_v03(), (*packets, packets[0]))
    with pytest.raises(ValueError, match="shadow views"):
        evaluate_v04_acceptance(
            _passing_v03(),
            (
                _packet(
                    DevelopmentView.IID_VALIDATION,
                    version=RemediationVersion.V03,
                ),
            ),
        )


def test_evaluators_reject_wrong_contract_types_and_versions() -> None:
    with pytest.raises(TypeError, match="exact DevelopmentViewMetrics"):
        evaluate_v03_acceptance(cast(Any, {}))
    with pytest.raises(TypeError, match="exact AcceptanceThresholds"):
        evaluate_v03_acceptance(
            _packet(DevelopmentView.IID_VALIDATION, version=RemediationVersion.V03),
            thresholds=cast(Any, {}),
        )
    with pytest.raises(ValueError, match=r"v0\.3 metric packet"):
        evaluate_v03_acceptance(
            _packet(DevelopmentView.RENDERER_SHADOW, version=RemediationVersion.V04)
        )

    with pytest.raises(TypeError, match="exact V03AcceptanceResult"):
        evaluate_v04_acceptance(cast(Any, {}), ())
    with pytest.raises(TypeError, match="exact metric-packet tuple"):
        evaluate_v04_acceptance(_passing_v03(), cast(Any, []))
    with pytest.raises(TypeError, match="exact AcceptanceThresholds"):
        evaluate_v04_acceptance(_passing_v03(), (), thresholds=cast(Any, {}))


def test_empty_v04_view_set_fails_closed_without_metric_support() -> None:
    result = evaluate_v04_acceptance(_passing_v03(), ())

    assert not result.advancement_allowed
    assert result.shadow_view_metrics == ()
    assert result.view_evaluations == ()
    assert result.worst_split_checks == ()
    assert result.composition_score_interval is None
    assert not result.program_checks[1].passed
    assert not result.program_checks[2].passed


def test_shadow_evaluation_contract_rejects_gate_shape_drift() -> None:
    result = evaluate_v04_acceptance(_passing_v03(), _shadow_packets())
    evaluation = result.view_evaluations[0]
    payload = evaluation.model_dump(mode="python", round_trip=True)
    payload["performance_thresholded"] = False
    with pytest.raises(ValidationError, match="reported without a threshold"):
        ShadowViewEvaluation.model_validate(payload)

    payload = evaluation.model_dump(mode="python", round_trip=True)
    payload["checks"] = payload["checks"][:-1]
    with pytest.raises(ValidationError, match="bound supported-metric set"):
        ShadowViewEvaluation.model_validate(payload)

    payload = evaluation.model_dump(mode="python", round_trip=True)
    payload["passed"] = False
    with pytest.raises(ValidationError, match="does not match"):
        ShadowViewEvaluation.model_validate(payload)


def test_v04_result_rejects_order_and_reporting_tampering() -> None:
    result = evaluate_v04_acceptance(_passing_v03(), _shadow_packets())

    payload = result.model_dump(mode="python", round_trip=True)
    required = cast(tuple[DevelopmentView, ...], payload["required_shadow_views"])
    payload["required_shadow_views"] = tuple(reversed(required))
    with pytest.raises(ValidationError, match="must remain preregistered"):
        V04AcceptanceResult.model_validate(payload)

    payload = result.model_dump(mode="python", round_trip=True)
    packets = cast(tuple[object, ...], payload["shadow_view_metrics"])
    payload["shadow_view_metrics"] = (packets[1], packets[0], *packets[2:])
    with pytest.raises(ValidationError, match="preregistered order"):
        V04AcceptanceResult.model_validate(payload)

    payload = result.model_dump(mode="python", round_trip=True)
    payload["composition_score_interval"] = None
    with pytest.raises(ValidationError, match="composition report"):
        V04AcceptanceResult.model_validate(payload)

    payload = result.model_dump(mode="python", round_trip=True)
    program_checks = cast(tuple[object, ...], payload["program_checks"])
    payload["program_checks"] = tuple(reversed(program_checks))
    with pytest.raises(ValidationError, match="canonical order"):
        V04AcceptanceResult.model_validate(payload)
