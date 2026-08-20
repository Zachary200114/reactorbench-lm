from __future__ import annotations

import pytest

from reactorbench.dataset.quality import QualityRecord, TaskShortcutRecord, audit_quality
from reactorbench.schemas.enums import SplitName, TaskName


def _provenance(example_id: str) -> dict[str, object]:
    return {
        "dataset_version": "0.1.0",
        "generator_commit": "abcdef1",
        "scenario_schema_version": "0.1.0",
        "renderer_version": "0.1.0",
        "seed": 1,
        "scenario_id": f"scenario-{example_id}",
        "plant_variant_id": "ASTER-A",
        "fault_family_ids": (),
        "template_family_ids": ("compact-log-v1",),
        "split_name": SplitName.IID_TRAIN.value,
        "task_name": TaskName.FAULT_FAMILY.value,
    }


def _quality(example_id: str) -> QualityRecord:
    return QualityRecord(
        example_id=example_id,
        split_name=SplitName.IID_TRAIN,
        text=f"[T+001][o-0000] synthetic observation {example_id}.",
        template_family_id="compact-log-v1",
        alias_family_id="canonical-v1",
        provenance=_provenance(example_id),
    )


def _task(
    record_id: str,
    *,
    render_id: str,
    task_name: TaskName,
    alias: str,
    target: str,
) -> TaskShortcutRecord:
    return TaskShortcutRecord(
        record_id=record_id,
        prompt_render_ids=(render_id,),
        task_name=task_name,
        template_family_id="compact-log-v1",
        alias_family_id=alias,
        target_labels=(target,),
        context_flags=("corruption:none",),
    )


def test_shortcut_contingencies_are_task_scoped_and_report_nonfailures() -> None:
    quality = (_quality("render-a"), _quality("render-b"), _quality("render-c"))
    tasks = (
        _task(
            "projection-a-fault",
            render_id="render-a",
            task_name=TaskName.FAULT_FAMILY,
            alias="heldout-v1",
            target="ALPHA",
        ),
        _task(
            "projection-b-fault",
            render_id="render-b",
            task_name=TaskName.FAULT_FAMILY,
            alias="heldout-v1",
            target="ALPHA",
        ),
        _task(
            "projection-c-fault",
            render_id="render-c",
            task_name=TaskName.FAULT_FAMILY,
            alias="canonical-v1",
            target="BETA",
        ),
        _task(
            "projection-a-action",
            render_id="render-a",
            task_name=TaskName.NEXT_ACTION,
            alias="heldout-v1",
            target="ACTION_A",
        ),
        _task(
            "projection-b-action",
            render_id="render-b",
            task_name=TaskName.NEXT_ACTION,
            alias="heldout-v1",
            target="ACTION_B",
        ),
        TaskShortcutRecord(
            record_id="projection-c-action",
            prompt_render_ids=("render-c",),
            task_name=TaskName.NEXT_ACTION,
            template_family_id="compact-log-v1",
            alias_family_id="canonical-v1",
            target_labels=("ACTION_B",),
            context_flags=(
                "corruption:none",
                "semantic:standby-state:available",
            ),
        ),
    )

    report = audit_quality(quality, task_records=tasks)

    finding = next(
        item
        for item in report.shortcut_findings
        if item.feature_name == "alias_family_id" and item.feature_value == "heldout-v1"
    )
    assert finding.task_name is TaskName.FAULT_FAMILY
    assert finding.feature_class == "renderer_nuisance"
    assert finding.sole_target == "ALPHA"
    assert finding.support == 2
    action_contingency = next(
        item
        for item in report.shortcut_contingencies
        if item.task_name is TaskName.NEXT_ACTION
        and item.feature_name == "alias_family_id"
        and item.feature_value == "heldout-v1"
    )
    assert action_contingency.target_counts == (("ACTION_A", 1), ("ACTION_B", 1))
    semantic = next(
        item
        for item in report.shortcut_contingencies
        if item.feature_value == "semantic:standby-state:available"
    )
    assert semantic.feature_class == "semantic_context"
    assert semantic.target_counts == (("ACTION_B", 1),)
    assert not any(
        item.feature_value == "semantic:standby-state:available"
        for item in report.shortcut_findings
    )
    assert not any(item.task_name is TaskName.NEXT_ACTION for item in report.shortcut_findings)
    assert report.task_record_count == len(tasks)
    assert tuple(item.record_id for item in report.audited_task_records) == tuple(
        sorted(task.record_id for task in tasks)
    )
    assert not report.passed


def test_task_shortcut_records_must_reference_audited_render_ids() -> None:
    quality = (_quality("render-a"),)
    task = _task(
        "projection-a",
        render_id="render-missing",
        task_name=TaskName.FAULT_FAMILY,
        alias="canonical-v1",
        target="ALPHA",
    )

    with pytest.raises(ValueError, match="audited rendered candidates"):
        audit_quality(quality, task_records=(task,))


def test_implicit_task_audit_rejects_grouped_task_provenance() -> None:
    record = _quality("render-a")
    grouped = record.model_copy(
        update={
            "provenance": {
                **record.provenance,
                "task_name": (TaskName.FAULT_FAMILY.value, TaskName.NEXT_ACTION.value),
            }
        }
    )
    with pytest.raises(ValueError, match="one scalar provenance task_name"):
        audit_quality((grouped,))


@pytest.mark.parametrize(
    "context_flags",
    [
        (),
        ("corruption:none", "corruption:safe_reorder"),
    ],
)
def test_task_shortcut_record_requires_exactly_one_corruption_plan(
    context_flags: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match="exactly one corruption plan"):
        TaskShortcutRecord(
            record_id="invalid-corruption-plan",
            prompt_render_ids=("render-a",),
            task_name=TaskName.FAULT_FAMILY,
            template_family_id="compact-log-v1",
            alias_family_id="canonical-v1",
            target_labels=("ALPHA",),
            context_flags=context_flags,
        )


def test_implicit_task_audit_preserves_explicit_corruption_plan() -> None:
    record = _quality("render-a").model_copy(update={"context_flags": ("corruption:safe_reorder",)})

    report = audit_quality((record,))

    corruption = tuple(
        item
        for item in report.shortcut_contingencies
        if item.feature_name == "context_flag" and item.feature_class == "renderer_nuisance"
    )
    assert tuple(item.feature_value for item in corruption) == ("corruption:safe_reorder",)


def test_joint_renderer_plan_shortcuts_are_detected_when_marginals_are_balanced() -> None:
    quality: list[QualityRecord] = []
    tasks: list[TaskShortcutRecord] = []
    combinations = (
        ("template-a", "alias-a", "ALPHA"),
        ("template-a", "alias-b", "BETA"),
        ("template-b", "alias-a", "BETA"),
        ("template-b", "alias-b", "ALPHA"),
    )
    for combination_index, (template, alias, target) in enumerate(combinations):
        for repeat in range(2):
            render_id = f"joint-{combination_index}-{repeat}"
            quality.append(_quality(render_id))
            tasks.append(
                TaskShortcutRecord(
                    record_id=f"task-{render_id}",
                    prompt_render_ids=(render_id,),
                    task_name=TaskName.FAULT_FAMILY,
                    template_family_id=template,
                    alias_family_id=alias,
                    target_labels=(target,),
                    context_flags=("corruption:none",),
                )
            )

    report = audit_quality(tuple(quality), task_records=tuple(tasks))

    assert not any(
        finding.feature_name in {"template_family_id", "alias_family_id"}
        for finding in report.shortcut_findings
    )
    joint = tuple(
        finding
        for finding in report.shortcut_findings
        if finding.feature_name == "template_alias_plan"
    )
    assert len(joint) == 4
    assert {finding.support for finding in joint} == {2}
