"""Measured compact-target inventory and preregistered generation-cap freeze."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path

from pydantic import Field, model_validator

from reactorbench.evaluation.compact import (
    CompactTargetConstraint,
    parse_compact_target,
    serialize_compact_target,
)
from reactorbench.schemas.base import ContractModel, canonical_json_bytes, canonical_sha256
from reactorbench.schemas.enums import TaskName
from reactorbench.tokenizer import BOS_ID, EOS_ID, PAD_ID, UNK_ID, ProjectTokenizer

from .config import InventoryPolicy, RemediationView
from .data import RemediationExample, SafeDevelopmentDataset
from .serialization import compact_serialized_parts, retained_compact_prompt_tokens


class TaskInventoryMeasurement(ContractModel):
    task_name: TaskName
    example_count: int = Field(ge=1)
    compile_count: int = Field(ge=0)
    compile_rate: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    round_trip_count: int = Field(ge=0)
    round_trip_rate: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    fit_count: int = Field(ge=0)
    target_fit_rate: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    reachable_count: int = Field(ge=0)
    reachability_rate: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    minimum_target_tokens: int = Field(ge=1)
    median_target_tokens: int = Field(ge=1)
    percentile_95_target_tokens: int = Field(ge=1)
    maximum_target_tokens: int = Field(ge=1)
    frozen_generation_cap: int = Field(ge=1, le=512)
    truncated_prompt_count: int = Field(ge=0)
    truncated_prompt_rate: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    minimum_prompt_tokens_retained: int = Field(ge=2)
    task_footer_retained_count: int = Field(ge=0)
    task_footer_retained_rate: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    cap_exhaustion_target_count: int = Field(ge=0)
    cap_exhaustion_target_rate: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def measurements_are_ordered(self) -> TaskInventoryMeasurement:
        lengths = (
            self.minimum_target_tokens,
            self.median_target_tokens,
            self.percentile_95_target_tokens,
            self.maximum_target_tokens,
        )
        if lengths != tuple(sorted(lengths)):
            raise ValueError("target-token measurements are not ordered")
        if self.maximum_target_tokens > self.frozen_generation_cap:
            raise ValueError("frozen generation cap does not fit every measured target")
        if self.truncated_prompt_count > self.example_count:
            raise ValueError("truncated prompt count exceeds task support")
        counted_rates = (
            (self.compile_count, self.compile_rate, "compile"),
            (self.round_trip_count, self.round_trip_rate, "round-trip"),
            (self.fit_count, self.target_fit_rate, "target-fit"),
            (self.reachable_count, self.reachability_rate, "reachability"),
            (
                self.truncated_prompt_count,
                self.truncated_prompt_rate,
                "prompt-truncation",
            ),
            (
                self.task_footer_retained_count,
                self.task_footer_retained_rate,
                "task-footer retention",
            ),
            (
                self.cap_exhaustion_target_count,
                self.cap_exhaustion_target_rate,
                "cap-exhaustion",
            ),
        )
        for count, rate, name in counted_rates:
            if count > self.example_count or rate != count / self.example_count:
                raise ValueError(f"task {name} count/rate differs from task support")
        return self


class CompactInventoryReport(ContractModel):
    report_version: str = "0.2.0"
    source_commit: str = Field(pattern=r"^[0-9a-f]{7,64}$")
    dataset_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tokenizer_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compact_contract_version: str = "0.2.0"
    permitted_views: tuple[RemediationView, ...]
    counts_by_view: tuple[tuple[RemediationView, int], ...]
    example_count: int = Field(ge=1)
    compile_count: int = Field(ge=0)
    compile_rate: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    round_trip_count: int = Field(ge=0)
    fit_count: int = Field(ge=0)
    reachable_count: int = Field(ge=0)
    task_measurements: tuple[TaskInventoryMeasurement, ...]
    prompt_truncation_count: int = Field(ge=0)
    prompt_truncation_rate: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    task_footer_retained_count: int = Field(ge=0)
    task_footer_retained_rate: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    cap_exhaustion_target_count: int = Field(ge=0)
    cap_exhaustion_target_rate: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    target_fit_rate: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    round_trip_rate: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    reachability_rate: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    checksum_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def gate_and_checksum_match(self) -> CompactInventoryReport:
        if (
            self.compile_count != self.example_count
            or self.round_trip_count != self.example_count
            or self.fit_count != self.example_count
            or self.reachable_count != self.example_count
            or self.task_footer_retained_count != self.example_count
            or self.cap_exhaustion_target_count != 0
            or self.compile_rate != 1.0
            or self.target_fit_rate != 1.0
            or self.round_trip_rate != 1.0
            or self.reachability_rate != 1.0
            or self.task_footer_retained_rate != 1.0
            or self.cap_exhaustion_target_rate != 0.0
        ):
            raise ValueError("compact inventory must fail closed unless every target passes")
        canonical_views = tuple(
            view for view in RemediationView if view in set(self.permitted_views)
        )
        if self.permitted_views != canonical_views:
            raise ValueError("inventory permitted views must be unique and canonical")
        if (
            tuple(view for view, _count in self.counts_by_view) != self.permitted_views
            or any(count <= 0 for _view, count in self.counts_by_view)
            or sum(count for _view, count in self.counts_by_view) != self.example_count
        ):
            raise ValueError("inventory view counts must be positive, canonical, and complete")
        tasks = tuple(item.task_name for item in self.task_measurements)
        if tasks != tuple(task for task in TaskName if task in set(tasks)):
            raise ValueError("task inventory measurements are not canonical")
        reconciled = (
            (sum(item.example_count for item in self.task_measurements), self.example_count),
            (sum(item.compile_count for item in self.task_measurements), self.compile_count),
            (
                sum(item.round_trip_count for item in self.task_measurements),
                self.round_trip_count,
            ),
            (sum(item.fit_count for item in self.task_measurements), self.fit_count),
            (sum(item.reachable_count for item in self.task_measurements), self.reachable_count),
            (
                sum(item.truncated_prompt_count for item in self.task_measurements),
                self.prompt_truncation_count,
            ),
            (
                sum(item.task_footer_retained_count for item in self.task_measurements),
                self.task_footer_retained_count,
            ),
            (
                sum(item.cap_exhaustion_target_count for item in self.task_measurements),
                self.cap_exhaustion_target_count,
            ),
        )
        if not self.task_measurements or any(
            observed != expected for observed, expected in reconciled
        ):
            raise ValueError("task inventory counts do not reconcile with report aggregates")
        aggregate_rates = (
            (self.compile_rate, self.compile_count / self.example_count),
            (self.target_fit_rate, self.fit_count / self.example_count),
            (self.round_trip_rate, self.round_trip_count / self.example_count),
            (self.reachability_rate, self.reachable_count / self.example_count),
            (
                self.prompt_truncation_rate,
                self.prompt_truncation_count / self.example_count,
            ),
            (
                self.task_footer_retained_rate,
                self.task_footer_retained_count / self.example_count,
            ),
            (
                self.cap_exhaustion_target_rate,
                self.cap_exhaustion_target_count / self.example_count,
            ),
        )
        if any(observed != expected for observed, expected in aggregate_rates):
            raise ValueError("inventory aggregate count/rate reconciliation failed")
        expected = canonical_sha256(
            self.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        if self.checksum_sha256 != expected:
            raise ValueError("compact inventory report checksum mismatch")
        return self

    @property
    def generation_caps(self) -> dict[TaskName, int]:
        return {item.task_name: item.frozen_generation_cap for item in self.task_measurements}


class CounterfactualCapExtensionReport(ContractModel):
    """Development-only v0.3 cap freeze for the task absent from v0.2 data."""

    report_version: str = "0.3.0"
    source_commit: str = Field(pattern=r"^[0-9a-f]{7,64}$")
    dataset_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tokenizer_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    base_inventory_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compact_contract_version: str = "0.2.0"
    permitted_views: tuple[RemediationView, RemediationView]
    task_name: TaskName
    example_count: int = Field(ge=1)
    train_example_count: int = Field(ge=1)
    validation_example_count: int = Field(ge=1)
    minimum_target_tokens: int = Field(ge=1)
    median_target_tokens: int = Field(ge=1)
    percentile_95_target_tokens: int = Field(ge=1)
    maximum_target_tokens: int = Field(ge=1)
    cap_margin_tokens: int = Field(ge=1, le=64)
    frozen_generation_cap: int = Field(ge=1, le=512)
    compile_count: int = Field(ge=1)
    compile_rate: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    round_trip_count: int = Field(ge=1)
    round_trip_rate: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    fit_count: int = Field(ge=1)
    target_fit_rate: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    reachable_count: int = Field(ge=1)
    reachability_rate: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    prompt_truncation_count: int = Field(ge=0)
    prompt_truncation_rate: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    task_footer_retained_count: int = Field(ge=0)
    task_footer_retained_rate: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    cap_exhaustion_target_count: int = Field(ge=0)
    cap_exhaustion_target_rate: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    checksum_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def boundary_gate_and_checksum_match(self) -> CounterfactualCapExtensionReport:
        if self.permitted_views != (
            RemediationView.IID_TRAIN,
            RemediationView.IID_VALIDATION,
        ):
            raise ValueError("v0.3 cap extension may use only IID train and validation")
        if self.task_name is not TaskName.COUNTERFACTUAL_COMPARE:
            raise ValueError("v0.3 cap extension is limited to counterfactual comparison")
        if self.train_example_count + self.validation_example_count != self.example_count:
            raise ValueError("v0.3 cap extension view counts do not cover its examples")
        lengths = (
            self.minimum_target_tokens,
            self.median_target_tokens,
            self.percentile_95_target_tokens,
            self.maximum_target_tokens,
        )
        if lengths != tuple(sorted(lengths)):
            raise ValueError("v0.3 target-token measurements are not ordered")
        if self.maximum_target_tokens > self.frozen_generation_cap:
            raise ValueError("v0.3 generation cap does not fit every measured target")
        if (
            not (
                self.compile_count
                == self.round_trip_count
                == self.fit_count
                == self.reachable_count
                == self.task_footer_retained_count
                == self.example_count
            )
            or self.cap_exhaustion_target_count != 0
        ):
            raise ValueError(
                "v0.3 cap extension requires complete compile/fit/round-trip/reachability"
            )
        rates = (
            (self.compile_rate, self.compile_count / self.example_count),
            (self.round_trip_rate, self.round_trip_count / self.example_count),
            (self.target_fit_rate, self.fit_count / self.example_count),
            (self.reachability_rate, self.reachable_count / self.example_count),
            (
                self.prompt_truncation_rate,
                self.prompt_truncation_count / self.example_count,
            ),
            (
                self.task_footer_retained_rate,
                self.task_footer_retained_count / self.example_count,
            ),
            (
                self.cap_exhaustion_target_rate,
                self.cap_exhaustion_target_count / self.example_count,
            ),
        )
        if self.prompt_truncation_count > self.example_count or any(
            observed != expected for observed, expected in rates
        ):
            raise ValueError("v0.3 cap extension count/rate reconciliation failed")
        expected = canonical_sha256(
            self.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
        )
        if self.checksum_sha256 != expected:
            raise ValueError("v0.3 cap extension checksum mismatch")
        return self


def _percentile_nearest_rank(values: tuple[int, ...], probability: float) -> int:
    if not values or not 0.0 < probability <= 1.0:
        raise ValueError("percentile requires non-empty values and probability in (0,1]")
    ordered = tuple(sorted(values))
    index = max(0, math.ceil(probability * len(ordered)) - 1)
    return ordered[index]


def audit_compact_target_reachability(
    example: RemediationExample,
    tokenizer: ProjectTokenizer,
    *,
    generation_cap: int,
) -> None:
    """Prove one canonical target is reachable using only its public context.

    Target text is consumed only by this pre-training audit.  The runtime constraint
    receives solely ``CompactTargetContext`` plus the already frozen generation cap,
    matching the inference boundary exactly.
    """

    if type(example) is not RemediationExample or type(tokenizer) is not ProjectTokenizer:
        raise TypeError("reachability audit requires an exact example and tokenizer")
    if type(generation_cap) is not int or not 1 <= generation_cap <= 512:
        raise ValueError("reachability audit requires a valid frozen generation cap")
    target_token_ids = tokenizer.encode(
        example.compact_target,
        add_bos=False,
        add_eos=False,
    )
    if not target_token_ids or len(target_token_ids) + 1 > generation_cap:
        raise ValueError("canonical target and EOS do not fit the frozen generation cap")
    special_ids = frozenset({UNK_ID, BOS_ID, EOS_ID, PAD_ID})
    if any(token_id in special_ids for token_id in target_token_ids):
        raise ValueError("canonical target token path contains a special or unknown token")
    constraint = CompactTargetConstraint(
        example.compact_context,
        maximum_generated_tokens=generation_cap,
    )
    generated: tuple[int, ...] = ()
    for expected_token_id in target_token_ids:
        allowed = constraint.allowed_next_token_ids(tokenizer, generated)
        if not allowed:
            raise ValueError("canonical target token path reaches a constrained dead end")
        current_text = tokenizer.decode(generated) if generated else ""
        if (EOS_ID in allowed) is not constraint.accepts_complete(current_text):
            raise ValueError("EOS reachability differs from compact-prefix completeness")
        if expected_token_id not in allowed:
            raise ValueError("canonical target token is unreachable under the frozen constraint")
        generated = (*generated, expected_token_id)
    if tokenizer.decode(generated) != example.compact_target:
        raise ValueError("canonical target token path does not decode losslessly")
    final_allowed = constraint.allowed_next_token_ids(tokenizer, generated)
    if EOS_ID not in final_allowed:
        raise ValueError("EOS is not reachable after the complete canonical target")


def measure_compact_inventory(
    dataset: SafeDevelopmentDataset,
    tokenizer: ProjectTokenizer,
    policy: InventoryPolicy,
) -> CompactInventoryReport:
    if (
        type(dataset) is not SafeDevelopmentDataset
        or type(tokenizer) is not ProjectTokenizer
        or type(policy) is not InventoryPolicy
    ):
        raise TypeError("inventory measurement requires exact dataset/tokenizer/policy contracts")
    if dataset.manifest.views != tuple(policy.permitted_views):
        raise ValueError("dataset views differ from the v0.2 inventory allowlist")
    target_lengths: dict[TaskName, list[int]] = defaultdict(list)
    compile_counts: Counter[TaskName] = Counter()
    round_trip_counts: Counter[TaskName] = Counter()
    for example in dataset.examples:
        parsed = parse_compact_target(example.compact_target, context=example.compact_context)
        compile_counts[example.task_name] += 1
        if (
            serialize_compact_target(parsed, context=example.compact_context)
            != example.compact_target
        ):
            raise ValueError("compact target did not round-trip canonically")
        round_trip_counts[example.task_name] += 1
        _prompt_text, target_text = compact_serialized_parts(example)
        target_tokens = len(tokenizer.encode(target_text, add_bos=False, add_eos=True))
        target_lengths[example.task_name].append(target_tokens)
    caps = {
        task: min(
            policy.maximum_cap_tokens,
            max(policy.minimum_cap_tokens, max(lengths) + policy.cap_margin_tokens),
        )
        for task, lengths in target_lengths.items()
    }
    if any(caps[task] < max(lengths) for task, lengths in target_lengths.items()):
        raise ValueError("preregistered maximum cap cannot fit every compact target")
    measurements: list[TaskInventoryMeasurement] = []
    for task in TaskName:
        lengths = tuple(target_lengths.get(task, ()))
        if not lengths:
            continue
        task_examples = tuple(item for item in dataset.examples if item.task_name is task)
        retained_counts: list[int] = []
        truncated = reachable = footer_retained = 0
        fit_count = sum(length <= caps[task] for length in lengths)
        cap_exhaustion_count = sum(length > caps[task] for length in lengths)
        for example in task_examples:
            retained_ids, _original_count, was_truncated = retained_compact_prompt_tokens(
                example,
                tokenizer,
                context_length=policy.context_length,
                generation_cap=caps[task],
            )
            retained_counts.append(len(retained_ids))
            truncated += int(was_truncated)
            footer_retained += 1
            audit_compact_target_reachability(
                example,
                tokenizer,
                generation_cap=caps[task],
            )
            reachable += 1
        support = len(task_examples)
        measurements.append(
            TaskInventoryMeasurement(
                task_name=task,
                example_count=support,
                compile_count=compile_counts[task],
                compile_rate=compile_counts[task] / support,
                round_trip_count=round_trip_counts[task],
                round_trip_rate=round_trip_counts[task] / support,
                fit_count=fit_count,
                target_fit_rate=fit_count / support,
                reachable_count=reachable,
                reachability_rate=reachable / support,
                minimum_target_tokens=min(lengths),
                median_target_tokens=_percentile_nearest_rank(lengths, 0.5),
                percentile_95_target_tokens=_percentile_nearest_rank(lengths, 0.95),
                maximum_target_tokens=max(lengths),
                frozen_generation_cap=caps[task],
                truncated_prompt_count=truncated,
                truncated_prompt_rate=truncated / support,
                minimum_prompt_tokens_retained=min(retained_counts),
                task_footer_retained_count=footer_retained,
                task_footer_retained_rate=footer_retained / support,
                cap_exhaustion_target_count=cap_exhaustion_count,
                cap_exhaustion_target_rate=cap_exhaustion_count / support,
            )
        )
    count = len(dataset.examples)
    compile_count = sum(item.compile_count for item in measurements)
    round_trip_count = sum(item.round_trip_count for item in measurements)
    fit_count = sum(item.fit_count for item in measurements)
    reachable_count = sum(item.reachable_count for item in measurements)
    total_truncated = sum(item.truncated_prompt_count for item in measurements)
    footer_retained_count = sum(item.task_footer_retained_count for item in measurements)
    cap_exhaustion_count = sum(item.cap_exhaustion_target_count for item in measurements)
    draft = CompactInventoryReport.model_construct(
        report_version="0.2.0",
        source_commit=dataset.manifest.source_commit,
        dataset_manifest_sha256=dataset.manifest.checksum_sha256,
        tokenizer_manifest_sha256=tokenizer.manifest.checksum_sha256,
        compact_contract_version="0.2.0",
        permitted_views=dataset.manifest.views,
        counts_by_view=dataset.manifest.counts_by_view,
        example_count=count,
        compile_count=compile_count,
        compile_rate=compile_count / count,
        round_trip_count=round_trip_count,
        fit_count=fit_count,
        reachable_count=reachable_count,
        task_measurements=tuple(measurements),
        prompt_truncation_count=total_truncated,
        prompt_truncation_rate=total_truncated / count,
        task_footer_retained_count=footer_retained_count,
        task_footer_retained_rate=footer_retained_count / count,
        cap_exhaustion_target_count=cap_exhaustion_count,
        cap_exhaustion_target_rate=cap_exhaustion_count / count,
        target_fit_rate=fit_count / count,
        round_trip_rate=round_trip_count / count,
        reachability_rate=reachable_count / count,
        checksum_sha256="0" * 64,
    )
    checksum = canonical_sha256(
        draft.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
    )
    report = CompactInventoryReport(
        **draft.model_dump(mode="python", exclude={"checksum_sha256"}),
        checksum_sha256=checksum,
    )
    validate_compact_inventory_dataset_counts(report, dataset)
    return report


def measure_counterfactual_cap_extension(
    dataset: SafeDevelopmentDataset,
    tokenizer: ProjectTokenizer,
    policy: InventoryPolicy,
    base_report: CompactInventoryReport,
) -> CounterfactualCapExtensionReport:
    """Freeze the sole new v0.3 task cap without observing any shadow/final row."""

    if (
        type(dataset) is not SafeDevelopmentDataset
        or type(tokenizer) is not ProjectTokenizer
        or type(policy) is not InventoryPolicy
        or type(base_report) is not CompactInventoryReport
    ):
        raise TypeError("v0.3 cap measurement requires exact inventory contracts")
    expected_views = (RemediationView.IID_TRAIN, RemediationView.IID_VALIDATION)
    if dataset.manifest.views != expected_views or dataset.manifest.dataset_version != "0.3.0":
        raise ValueError("v0.3 cap extension requires the isolated train/validation dataset")
    if base_report.permitted_views != expected_views:
        raise ValueError("base inventory report has an incompatible development boundary")
    if TaskName.COUNTERFACTUAL_COMPARE in base_report.generation_caps:
        raise ValueError("base inventory already defines the counterfactual generation cap")
    if tokenizer.manifest.checksum_sha256 != base_report.tokenizer_manifest_sha256:
        raise ValueError("v0.3 cap extension tokenizer differs from the base inventory")

    examples = tuple(
        example
        for example in dataset.examples
        if example.task_name is TaskName.COUNTERFACTUAL_COMPARE
    )
    train_count = sum(example.view is RemediationView.IID_TRAIN for example in examples)
    validation_count = sum(example.view is RemediationView.IID_VALIDATION for example in examples)
    if not examples or not train_count or not validation_count:
        raise ValueError("v0.3 cap extension requires counterfactual train and validation support")

    lengths: list[int] = []
    compile_count = round_trip_count = 0
    for example in examples:
        parsed = parse_compact_target(example.compact_target, context=example.compact_context)
        compile_count += 1
        if (
            serialize_compact_target(parsed, context=example.compact_context)
            != example.compact_target
        ):
            raise ValueError("v0.3 counterfactual target did not round-trip canonically")
        round_trip_count += 1
        _prompt_text, target_text = compact_serialized_parts(example)
        lengths.append(len(tokenizer.encode(target_text, add_bos=False, add_eos=True)))
    frozen_cap = min(
        policy.maximum_cap_tokens,
        max(policy.minimum_cap_tokens, max(lengths) + policy.cap_margin_tokens),
    )
    if frozen_cap < max(lengths):
        raise ValueError("v0.3 counterfactual target cannot fit the preregistered cap policy")

    reachable_count = footer_retained_count = prompt_truncation_count = 0
    for example in examples:
        retained_ids, _original_count, was_truncated = retained_compact_prompt_tokens(
            example,
            tokenizer,
            context_length=policy.context_length,
            generation_cap=frozen_cap,
        )
        if len(retained_ids) < 2:
            raise RuntimeError("v0.3 compact prompt retained an invalid token boundary")
        prompt_truncation_count += int(was_truncated)
        footer_retained_count += 1
        audit_compact_target_reachability(
            example,
            tokenizer,
            generation_cap=frozen_cap,
        )
        reachable_count += 1

    ordered = tuple(sorted(lengths))
    count = len(examples)
    fit_count = sum(length <= frozen_cap for length in ordered)
    cap_exhaustion_count = sum(length > frozen_cap for length in ordered)
    draft = CounterfactualCapExtensionReport.model_construct(
        source_commit=dataset.manifest.source_commit,
        dataset_manifest_sha256=dataset.manifest.checksum_sha256,
        tokenizer_manifest_sha256=tokenizer.manifest.checksum_sha256,
        base_inventory_report_sha256=base_report.checksum_sha256,
        permitted_views=expected_views,
        task_name=TaskName.COUNTERFACTUAL_COMPARE,
        example_count=count,
        train_example_count=train_count,
        validation_example_count=validation_count,
        minimum_target_tokens=min(ordered),
        median_target_tokens=_percentile_nearest_rank(ordered, 0.5),
        percentile_95_target_tokens=_percentile_nearest_rank(ordered, 0.95),
        maximum_target_tokens=max(ordered),
        cap_margin_tokens=policy.cap_margin_tokens,
        frozen_generation_cap=frozen_cap,
        compile_count=compile_count,
        compile_rate=compile_count / count,
        round_trip_count=round_trip_count,
        round_trip_rate=round_trip_count / count,
        fit_count=fit_count,
        target_fit_rate=fit_count / count,
        reachable_count=reachable_count,
        reachability_rate=reachable_count / count,
        prompt_truncation_count=prompt_truncation_count,
        prompt_truncation_rate=prompt_truncation_count / count,
        task_footer_retained_count=footer_retained_count,
        task_footer_retained_rate=footer_retained_count / count,
        cap_exhaustion_target_count=cap_exhaustion_count,
        cap_exhaustion_target_rate=cap_exhaustion_count / count,
        checksum_sha256="0" * 64,
    )
    checksum = canonical_sha256(
        draft.model_dump(mode="json", round_trip=True, exclude={"checksum_sha256"})
    )
    report = CounterfactualCapExtensionReport(
        **draft.model_dump(mode="python", exclude={"checksum_sha256"}),
        checksum_sha256=checksum,
    )
    validate_counterfactual_cap_dataset_counts(report, dataset)
    return report


def validate_compact_inventory_dataset_counts(
    report: CompactInventoryReport,
    dataset: SafeDevelopmentDataset,
) -> None:
    """Reconcile a serialized base report's count tables with validated raw rows."""

    if type(report) is not CompactInventoryReport or type(dataset) is not SafeDevelopmentDataset:
        raise TypeError("inventory count validation requires exact report and dataset contracts")
    if report.dataset_manifest_sha256 != dataset.manifest.checksum_sha256:
        raise ValueError("inventory report is bound to another dataset manifest")
    if report.counts_by_view != dataset.manifest.counts_by_view:
        raise ValueError("inventory report view counts differ from raw examples")
    expected_task_counts = tuple(
        (measurement.task_name, measurement.example_count)
        for measurement in report.task_measurements
    )
    if expected_task_counts != dataset.manifest.counts_by_task:
        raise ValueError("inventory report task counts differ from raw examples")


def validate_counterfactual_cap_dataset_counts(
    report: CounterfactualCapExtensionReport,
    dataset: SafeDevelopmentDataset,
) -> None:
    """Reconcile the single-task cap extension with its validated raw rows."""

    if (
        type(report) is not CounterfactualCapExtensionReport
        or type(dataset) is not SafeDevelopmentDataset
    ):
        raise TypeError("cap count validation requires exact report and dataset contracts")
    if report.dataset_manifest_sha256 != dataset.manifest.checksum_sha256:
        raise ValueError("cap extension is bound to another dataset manifest")
    examples = tuple(
        item for item in dataset.examples if item.task_name is TaskName.COUNTERFACTUAL_COMPARE
    )
    expected = (
        len(examples),
        sum(item.view is RemediationView.IID_TRAIN for item in examples),
        sum(item.view is RemediationView.IID_VALIDATION for item in examples),
    )
    observed = (
        report.example_count,
        report.train_example_count,
        report.validation_example_count,
    )
    if observed != expected:
        raise ValueError("cap extension view counts differ from raw examples")


def write_compact_inventory_report(report: CompactInventoryReport, path: Path) -> None:
    if type(report) is not CompactInventoryReport or not isinstance(path, Path):
        raise TypeError("inventory report write requires an exact report and Path")
    if path.exists() or path.is_symlink():
        raise FileExistsError("compact inventory report must not overwrite an existing path")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError("compact inventory temporary path already exists")
    try:
        with temporary.open("xb") as stream:
            stream.write(
                canonical_json_bytes(report.model_dump(mode="json", round_trip=True)) + b"\n"
            )
            stream.flush()
            import os

            os.fsync(stream.fileno())
        temporary.replace(path)
    except Exception:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()
        raise


def write_counterfactual_cap_extension_report(
    report: CounterfactualCapExtensionReport, path: Path
) -> None:
    if type(report) is not CounterfactualCapExtensionReport or not isinstance(path, Path):
        raise TypeError("cap extension write requires an exact report and Path")
    if path.exists() or path.is_symlink():
        raise FileExistsError("cap extension report must not overwrite an existing path")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError("cap extension temporary path already exists")
    try:
        with temporary.open("xb") as stream:
            stream.write(
                canonical_json_bytes(report.model_dump(mode="json", round_trip=True)) + b"\n"
            )
            stream.flush()
            import os

            os.fsync(stream.fileno())
        temporary.replace(path)
    except Exception:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()
        raise


__all__ = [
    "CompactInventoryReport",
    "CounterfactualCapExtensionReport",
    "TaskInventoryMeasurement",
    "audit_compact_target_reachability",
    "measure_compact_inventory",
    "measure_counterfactual_cap_extension",
    "validate_compact_inventory_dataset_counts",
    "validate_counterfactual_cap_dataset_counts",
    "write_compact_inventory_report",
    "write_counterfactual_cap_extension_report",
]
