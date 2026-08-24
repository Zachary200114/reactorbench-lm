from __future__ import annotations

import json
from dataclasses import replace
from typing import cast

import pytest

from reactorbench.remediation.local_monitor import (
    MAX_ACTIVITY_CHARACTERS,
    MAX_ACTIVITY_ENTRIES,
    RUN_NAME,
    ActivityBuffer,
    AlreadyActiveError,
    CommandOutcome,
    GuiAction,
    LongRunLauncher,
    MonitorState,
    MonitorStatus,
    StatusParseError,
    StatusReader,
    _bounded_decode,
    _operation_json,
    _parser,
    _run_short_command,
    _safe_action_message,
    _status_payload,
    _trusted_paths,
    button_policy,
    command_for,
    parse_status_output,
)

SOURCE_COMMIT = "8f6d491188164c1c81c86f350516d1019c843105"


def _running_status_output(*, stop_request: str = "none") -> str:
    return "\n".join(
        (
            "Pipeline status: running",
            f"Run: {RUN_NAME}",
            f"Source commit: {SOURCE_COMMIT}",
            "Next stage: v02_development_gate",
            "Interruptions: 1",
            "Latest progress: heartbeat at v02_development_gate (event 364)",
            "Message: pipeline remains active",
            "Stage position: 5/16",
            "Work completed: 128/252",
            "Elapsed seconds: 3627.6",
            "Latest verified update UTC: 2026-08-24T20:15:30+00:00",
            "Latest metric: validation_nll=0.14841478",
            "Estimated seconds remaining: 7200.0",
            "Latest checkpoint: stages/03-v02_development_training/checkpoint.json",
            f"Stop request: {stop_request}",
            "",
        )
    )


def test_verified_status_parses_every_gui_field_and_progress_fraction() -> None:
    status = parse_status_output(_running_status_output())

    assert status.state is MonitorState.RUNNING
    assert status.verified is True
    assert status.run_exists is True
    assert status.current_stage == "v02_development_gate"
    assert (status.stage_index, status.stage_total) == (5, 16)
    assert (status.completed_units, status.total_units) == (128, 252)
    assert status.latest_event == "heartbeat"
    assert status.event_sequence == 364
    assert status.elapsed_seconds == 3627.6
    assert status.eta_seconds == 7200.0
    assert status.metric_name == "validation_nll"
    assert status.metric_value == pytest.approx(0.14841478)
    assert status.interruptions == 1
    assert status.stop_requested is False
    assert status.latest_update_utc == "2026-08-24T20:15:30+00:00"
    assert status.work_percent == pytest.approx(50.79365079)
    assert status.overall_percent == pytest.approx(28.17460317)
    version_name, version_stage_index, version_stage_total, version_percent = (
        status.version_progress
    )
    assert (version_name, version_stage_index, version_stage_total) == ("v0.2", 4, 4)
    assert version_percent == pytest.approx(87.69841270)


@pytest.mark.parametrize(
    ("stage_index", "expected"),
    [
        (1, ("Setup", 1, 1, 0.0)),
        (2, ("v0.2", 1, 4, 0.0)),
        (5, ("v0.2", 4, 4, 75.0)),
        (6, ("v0.3", 1, 5, 0.0)),
        (10, ("v0.3", 5, 5, 80.0)),
        (11, ("v0.4", 1, 5, 0.0)),
        (15, ("v0.4", 5, 5, 80.0)),
        (16, ("Finalization", 1, 1, 0.0)),
    ],
)
def test_version_progress_covers_every_frozen_pipeline_group(
    stage_index: int,
    expected: tuple[str, int, int, float],
) -> None:
    status = replace(
        parse_status_output(_running_status_output()),
        stage_index=stage_index,
        completed_units=None,
        total_units=None,
    )

    assert status.version_progress == expected


def test_ready_status_without_progress_is_valid_but_cannot_start_again() -> None:
    output = "\n".join(
        (
            "Pipeline status: ready",
            f"Run: {RUN_NAME}",
            f"Source commit: {SOURCE_COMMIT}",
            "Next stage: preflight",
            "Interruptions: 0",
            "Progress reporter: not started",
            "Stop request: none",
            "",
        )
    )

    status = parse_status_output(output)
    policy = button_policy(status, operation_in_progress=False, local_launcher_active=False)

    assert status.state is MonitorState.RUNNING
    assert status.latest_event == "not started"
    assert policy.start is False
    assert policy.stop is True
    assert policy.resume is False


@pytest.mark.parametrize(
    ("state", "pipeline_status", "expected_stop", "expected_resume"),
    [
        (MonitorState.RUNNING, "running", True, False),
        (MonitorState.STOPPED, "stopped", False, True),
        (MonitorState.BLOCKED, "blocked", False, False),
        (MonitorState.FAILED, "failed", False, False),
        (MonitorState.COMPLETED, "completed", False, False),
    ],
)
def test_terminal_and_active_states_enable_only_safe_actions(
    state: MonitorState,
    pipeline_status: str,
    expected_stop: bool,
    expected_resume: bool,
) -> None:
    status = replace(
        parse_status_output(_running_status_output()),
        state=state,
        pipeline_status=pipeline_status,
    )

    policy = button_policy(status, operation_in_progress=False, local_launcher_active=False)

    assert policy.start is False
    assert policy.stop is expected_stop
    assert policy.resume is expected_resume


def test_not_started_is_the_only_state_that_enables_start() -> None:
    status = MonitorStatus.not_started(SOURCE_COMMIT)
    policy = button_policy(status, operation_in_progress=False, local_launcher_active=False)

    assert policy.start is True
    assert policy.stop is False
    assert policy.resume is False
    assert policy.open_finder is False

    busy = button_policy(status, operation_in_progress=True, local_launcher_active=False)
    assert busy.start is False
    assert busy.stop is False
    assert busy.resume is False


def test_pending_stop_and_unverified_status_disable_mutating_controls() -> None:
    pending = parse_status_output(_running_status_output(stop_request="pending"))
    assert (
        button_policy(pending, operation_in_progress=False, local_launcher_active=False).stop
        is False
    )

    invalid = MonitorStatus.failed_verification("Status evidence failed safely.")
    policy = button_policy(invalid, operation_in_progress=False, local_launcher_active=False)
    assert policy.start is False
    assert policy.stop is False
    assert policy.resume is False
    assert policy.copy_status is False


@pytest.mark.parametrize(
    "changed",
    [
        "Run: another-run",
        "Stage position: 5/15",
        "Latest verified update UTC: 2026-08-24T20:15:30+05:00",
        "Unknown field: unsafe",
    ],
)
def test_changed_or_mismatched_status_fails_closed(changed: str) -> None:
    lines = _running_status_output().splitlines()
    key = changed.split(": ", 1)[0]
    replacement_index = next(
        (index for index, line in enumerate(lines) if line.startswith(f"{key}: ")),
        None,
    )
    if replacement_index is None:
        lines.insert(-1, changed)
    else:
        lines[replacement_index] = changed

    with pytest.raises(StatusParseError):
        parse_status_output("\n".join((*lines, "")))


def test_duplicate_status_fields_and_unsafe_checkpoint_are_refused() -> None:
    duplicate = _running_status_output().replace(
        "Stop request: none\n",
        "Run: duplicate\nStop request: none\n",
    )
    with pytest.raises(StatusParseError):
        parse_status_output(duplicate)

    unsafe = _running_status_output().replace(
        "stages/03-v02_development_training/checkpoint.json",
        "../checkpoint.json",
    )
    with pytest.raises(StatusParseError):
        parse_status_output(unsafe)


def test_status_reader_maps_only_the_exact_missing_run_response_to_not_started() -> None:
    calls: list[GuiAction] = []

    def missing(action: GuiAction) -> CommandOutcome:
        calls.append(action)
        return CommandOutcome(
            returncode=3,
            stdout="",
            stderr="Command refused: Pipeline run does not exist; use the start command.\n",
        )

    status = StatusReader(missing, lambda: SOURCE_COMMIT).read()
    assert calls == [GuiAction.STATUS]
    assert status == MonitorStatus.not_started(SOURCE_COMMIT)

    def ambiguous(_action: GuiAction) -> CommandOutcome:
        return CommandOutcome(returncode=3, stdout="", stderr="different missing input\n")

    refused = StatusReader(ambiguous, lambda: SOURCE_COMMIT).read()
    assert refused.verified is False
    assert refused.state is MonitorState.FAILED


def test_command_routes_are_fixed_and_reject_strings_paths_and_extra_arguments() -> None:
    paths = _trusted_paths()
    assert command_for(GuiAction.DRY_RUN) == (str(paths.dry_run_wrapper), "--dry-run")
    assert command_for(GuiAction.STATUS) == (str(paths.status_wrapper),)
    assert command_for(GuiAction.START) == (
        "/usr/bin/caffeinate",
        "-i",
        str(paths.start_wrapper),
    )
    assert command_for(GuiAction.RESUME) == (
        "/usr/bin/caffeinate",
        "-i",
        str(paths.resume_wrapper),
    )
    assert command_for(GuiAction.OPEN_FINDER) == ("/usr/bin/open", str(paths.run_directory))
    assert not any("--config" in item for action in GuiAction for item in command_for(action))

    with pytest.raises(TypeError):
        command_for(cast(GuiAction, "start"))
    with pytest.raises(TypeError):
        _run_short_command(cast(GuiAction, "status"))


class _FakeProcess:
    def __init__(self, pid: int = 4321) -> None:
        self.pid = pid
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode


def test_long_launcher_refuses_duplicate_process_and_close_does_not_kill_child() -> None:
    commands: list[tuple[str, ...]] = []
    process = _FakeProcess()

    def spawn(command: tuple[str, ...]) -> _FakeProcess:
        commands.append(command)
        return process

    launcher = LongRunLauncher(spawn)
    assert launcher.launch(GuiAction.START) == 4321
    assert commands == [command_for(GuiAction.START)]
    assert launcher.has_active_process() is True

    with pytest.raises(AlreadyActiveError):
        launcher.launch(GuiAction.RESUME)

    launcher.window_closed()
    assert process.returncode is None
    assert launcher.has_active_process() is True


def test_long_launcher_refuses_non_lifecycle_action() -> None:
    launcher = LongRunLauncher(lambda _command: _FakeProcess())
    with pytest.raises(ValueError, match="only Start and Resume"):
        launcher.launch(GuiAction.DRY_RUN)


def test_output_log_and_public_errors_are_bounded_and_do_not_echo_private_detail() -> None:
    bounded = _bounded_decode(b"x" * 1_000, maximum_bytes=128)
    assert len(bounded.encode("utf-8")) <= 128
    assert bounded.endswith("[output truncated safely]")

    outcome = CommandOutcome(
        returncode=5,
        stdout="",
        stderr="private traceback at /Users/private/secret/checkpoint",
    )
    message = _safe_action_message(GuiAction.DRY_RUN, outcome)
    assert "local integrity verification failed" in message
    assert "private" not in message
    assert "Traceback" not in message

    activity = ActivityBuffer()
    for index in range(500):
        activity.append(f"entry {index}: " + "x" * 700)
    assert len(activity.entries()) <= MAX_ACTIVITY_ENTRIES
    assert len(activity.text()) <= MAX_ACTIVITY_CHARACTERS


def test_native_window_payload_contains_exact_safe_policy_and_no_arbitrary_paths() -> None:
    payload = _status_payload(MonitorStatus.not_started(SOURCE_COMMIT))

    assert payload["state"] == "Not started"
    assert payload["run_name"] == RUN_NAME
    assert payload["source_commit"] == SOURCE_COMMIT
    assert payload["overall_percent"] == 0.0
    assert payload["version_name"] == "Setup"
    assert payload["version_percent"] == 0.0
    assert payload["version_stage_index"] == 0
    assert payload["version_stage_total"] == 1
    assert payload["work_percent"] == 0.0
    assert payload["policy"] == {
        "copy_status": True,
        "dry_run": True,
        "open_finder": False,
        "refresh": True,
        "resume": False,
        "start": True,
        "stop": False,
    }
    assert "path" not in " ".join(payload)


def test_json_operation_uses_safe_message_without_echoing_command_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    private_detail = "/Users/private/secret/checkpoint"

    def refused(action: GuiAction) -> CommandOutcome:
        assert action is GuiAction.DRY_RUN
        return CommandOutcome(returncode=5, stdout="", stderr=private_detail)

    monkeypatch.setattr("reactorbench.remediation.local_monitor._run_short_command", refused)

    assert _operation_json(GuiAction.DRY_RUN) == 5
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "kind": "dry-run",
        "message": (
            "Local operation did not complete: local integrity verification failed (exit 5)."
        ),
        "ok": False,
        "pid": None,
        "returncode": 5,
    }
    assert private_detail not in json.dumps(payload)


def test_controller_parser_has_only_closed_action_flags() -> None:
    arguments = _parser().parse_args(["--snapshot-json"])
    assert arguments.snapshot_json is True

    with pytest.raises(SystemExit) as missing:
        _parser().parse_args([])
    assert missing.value.code == 2

    with pytest.raises(SystemExit) as arbitrary:
        _parser().parse_args(["--snapshot-json", "--config", "../unsafe.toml"])
    assert arbitrary.value.code == 2
