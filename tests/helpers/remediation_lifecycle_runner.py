"""Separate-process driver for the bounded remediation lifecycle integration test.

The driver injects tiny deterministic stage callbacks into the real public command
layer.  It never imports a dataset, opens held-out evidence, or invokes scientific
pipeline actions.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import reactorbench.remediation.cli as cli
from reactorbench.remediation.config import PIPELINE_STAGES, PipelineConfig, config_sha256
from reactorbench.remediation.orchestration import (
    PipelineState,
    StageAction,
    StageContext,
    StageOutcome,
)

SOURCE_COMMIT = "abcdef0123456789"
HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
RUN_NAME = "lifecycle-run"
SIMULATED_STAGE = "v02_inventory_and_caps"
SIMULATED_TOTAL_UNITS = 5
FIRST_ATTEMPT_COMPLETED_UNITS = 2
STOP_WAIT_SECONDS = 5.0


def _config() -> PipelineConfig:
    return PipelineConfig.model_validate(
        {
            "pipeline_version": "0.4.0",
            "run_name": RUN_NAME,
            "run_root": "work",
            "v02_config_path": "configs/v02.toml",
            "v02_config_sha256": HASH_A,
            "v03_config_path": "configs/v03.toml",
            "v03_config_sha256": HASH_B,
            "v04_config_path": "configs/v04.toml",
            "v04_config_sha256": HASH_C,
            "stage_order": list(PIPELINE_STAGES),
            "heartbeat_interval_seconds": 5,
            "maximum_status_bytes": 64 * 1024,
            "maximum_event_log_bytes": 1024 * 1024,
            "maximum_pipeline_seconds": 3600,
            "maximum_run_bytes": 4 * 1024 * 1024,
            "maximum_process_rss_bytes": 512 * 1024**2,
            "stop_before_final_evaluation": True,
        }
    )


def _write_json_exclusive(path: Path, payload: Mapping[str, object]) -> None:
    encoded = (
        json.dumps(
            dict(payload),
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        + b"\n"
    )
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    directory_descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def _read_completed_units(path: Path) -> int:
    raw_payload: object = json.loads(path.read_text(encoding="ascii"))
    if type(raw_payload) is not dict:
        raise ValueError("simulated lifecycle checkpoint failed strict validation")
    payload = cast(dict[str, object], raw_payload)
    completed_value = payload.get("completed_units")
    if (
        set(payload) != {"completed_units", "stage", "total_units"}
        or payload.get("stage") != SIMULATED_STAGE
        or payload.get("total_units") != SIMULATED_TOTAL_UNITS
        or type(completed_value) is not int
    ):
        raise ValueError("simulated lifecycle checkpoint failed strict validation")
    if not 0 < completed_value < SIMULATED_TOTAL_UNITS:
        raise ValueError("simulated lifecycle checkpoint is outside its unit bound")
    return completed_value


def _simple_action(stage: str) -> StageAction:
    def action(context: StageContext) -> StageOutcome:
        _write_json_exclusive(
            context.attempt_directory / "action-invocation.json",
            {"attempt": context.attempt_directory.name, "stage": stage},
        )
        return StageOutcome(summary=f"{stage} simulated successfully.")

    return action


def _simulated_multistep_action(context: StageContext) -> StageOutcome:
    _write_json_exclusive(
        context.attempt_directory / "action-invocation.json",
        {"attempt": context.attempt_directory.name, "stage": SIMULATED_STAGE},
    )
    attempt_number = int(context.attempt_directory.name.removeprefix("attempt-"))
    if attempt_number == 1:
        for completed_units in range(1, FIRST_ATTEMPT_COMPLETED_UNITS + 1):
            context.progress.report(
                stage=SIMULATED_STAGE,
                stage_index=2,
                stage_total=len(PIPELINE_STAGES),
                completed_units=completed_units,
                total_units=SIMULATED_TOTAL_UNITS,
                message="Simulated lifecycle work advanced.",
            )
        checkpoint = context.attempt_directory / "lifecycle-checkpoint.json"
        _write_json_exclusive(
            checkpoint,
            {
                "completed_units": FIRST_ATTEMPT_COMPLETED_UNITS,
                "stage": SIMULATED_STAGE,
                "total_units": SIMULATED_TOTAL_UNITS,
            },
        )
        checkpoint_reference = checkpoint.relative_to(context.run_directory).as_posix()
        context.progress.checkpoint(
            checkpoint=checkpoint_reference,
            stage=SIMULATED_STAGE,
            message="Simulated durable checkpoint saved.",
        )
        context.progress.report(
            stage=SIMULATED_STAGE,
            stage_index=2,
            stage_total=len(PIPELINE_STAGES),
            completed_units=FIRST_ATTEMPT_COMPLETED_UNITS,
            total_units=SIMULATED_TOTAL_UNITS,
            latest_checkpoint=checkpoint_reference,
            message="Simulated stage is ready for lifecycle controls.",
        )
        deadline = time.monotonic() + STOP_WAIT_SECONDS
        polls = 0
        while time.monotonic() < deadline:
            if context.stop_requested():
                raise KeyboardInterrupt
            polls += 1
            if polls % 25 == 0:
                context.progress.report(
                    stage=SIMULATED_STAGE,
                    stage_index=2,
                    stage_total=len(PIPELINE_STAGES),
                    completed_units=FIRST_ATTEMPT_COMPLETED_UNITS,
                    total_units=SIMULATED_TOTAL_UNITS,
                    latest_checkpoint=checkpoint_reference,
                    message="Simulated stage remains safely resumable.",
                )
            time.sleep(0.01)
        raise RuntimeError("bounded lifecycle test did not receive a stop request")

    previous_checkpoint = (
        context.attempt_directory.parent / "attempt-0001" / "lifecycle-checkpoint.json"
    )
    completed_units = _read_completed_units(previous_checkpoint)
    _write_json_exclusive(
        context.attempt_directory / "resumed-from-checkpoint.json",
        {
            "completed_units": completed_units,
            "source_checkpoint": previous_checkpoint.relative_to(context.run_directory).as_posix(),
            "stage": SIMULATED_STAGE,
        },
    )
    for unit in range(completed_units + 1, SIMULATED_TOTAL_UNITS + 1):
        context.progress.report(
            stage=SIMULATED_STAGE,
            stage_index=2,
            stage_total=len(PIPELINE_STAGES),
            completed_units=unit,
            total_units=SIMULATED_TOTAL_UNITS,
            message="Simulated resumed work advanced.",
        )
    return StageOutcome(summary="Simulated stage resumed from durable unit two.")


def _stage_actions(
    *, project_root: Path, config: PipelineConfig, source_commit: str
) -> Mapping[str, StageAction]:
    del project_root
    if config.run_name != RUN_NAME or source_commit != SOURCE_COMMIT:
        raise ValueError("lifecycle test bindings differ from their frozen values")
    return {
        stage: (_simulated_multistep_action if stage == SIMULATED_STAGE else _simple_action(stage))
        for stage in PIPELINE_STAGES
    }


def _stop_path(*, project_root: Path, config: PipelineConfig) -> Path:
    return project_root / config.run_root / config.run_name / "STOP_REQUESTED"


def _request_stop(path: Path) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(b"stop requested by lifecycle integration test\n")
        stream.flush()
        os.fsync(stream.fileno())


def _archive_stop(path: Path) -> Path | None:
    if not path.exists():
        return None
    archive_directory = path.parent / "stop-requests"
    archive_directory.mkdir(mode=0o750, exist_ok=True)
    destination = archive_directory / "acknowledged-stop-request"
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("lifecycle stop archive already exists")
    path.replace(destination)
    return destination


def _write_terminal_review(
    *,
    project_root: Path,
    config: PipelineConfig,
    source_commit: str,
    state: PipelineState,
) -> object:
    review_directory = project_root / config.run_root / config.run_name / "terminal-review"
    review_directory.mkdir(mode=0o750, exist_ok=False)
    manifest = review_directory / "review-bundle.json"
    summary = review_directory / "review-summary.md"
    _write_json_exclusive(
        manifest,
        {
            "pipeline_state_sha256": state.checksum_sha256,
            "source_commit": source_commit,
            "status": state.status,
        },
    )
    descriptor = os.open(summary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(b"# Simulated terminal review\n\nThe run stopped safely.\n")
        stream.flush()
        os.fsync(stream.fileno())
    return SimpleNamespace(manifest_path=manifest, summary_path=summary)


def _install_test_runtime(project_root: Path) -> None:
    config = _config()
    loaded = cli.LoadedPipeline(
        config=config,
        config_path=project_root / "frozen.toml",
        checksum_sha256=config_sha256(config),
    )
    module = SimpleNamespace(
        build_stage_actions=_stage_actions,
        pipeline_stop_file=_stop_path,
        request_pipeline_stop=_request_stop,
        archive_pipeline_stop=_archive_stop,
        write_terminal_review_bundle=_write_terminal_review,
    )

    def load_configuration(root: Path, relative_path: str) -> cli.LoadedPipeline:
        del relative_path
        if root != project_root:
            raise ValueError("lifecycle project root changed unexpectedly")
        return loaded

    def source_commit(root: Path) -> str:
        if root != project_root:
            raise ValueError("lifecycle project root changed unexpectedly")
        return SOURCE_COMMIT

    patch.object(cli, "_load_configuration", load_configuration).start()
    patch.object(cli, "_source_commit", source_commit).start()
    patch.object(cli, "_pipeline_module", lambda: module).start()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=Path)
    parser.add_argument(
        "command",
        choices=("start", "status", "stop", "resume", "dry-run"),
    )
    arguments = parser.parse_args(argv)
    project_root = arguments.project_root.resolve(strict=True)
    _install_test_runtime(project_root)
    return cli.main(
        [arguments.command],
        project_root=project_root,
        enforce_venv=False,
    )


if __name__ == "__main__":
    raise SystemExit(main())
