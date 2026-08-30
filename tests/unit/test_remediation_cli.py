"""Focused command-layer tests using only temporary run state and tiny callbacks."""

from __future__ import annotations

import io
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest

import reactorbench.remediation.cli as cli
from reactorbench.remediation.config import PIPELINE_STAGES, PipelineConfig, config_sha256
from reactorbench.remediation.orchestration import (
    PipelineState,
    PipelineStore,
    StageAction,
    StageContext,
    StageOutcome,
)
from reactorbench.remediation.progress import (
    ProgressIOError,
    ProgressMetric,
    ProgressReporter,
    ProgressSnapshot,
)
from reactorbench.schemas.base import canonical_json_bytes

SOURCE_COMMIT = "abcdef0"
HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
SCRIPTS = (
    "run_phase6_pipeline.sh",
    "check_phase6_status.sh",
    "resume_phase6_pipeline.sh",
    "stop_phase6_pipeline.sh",
    "run_phase6_evaluation.sh",
)


def _project_root(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="ascii")
    (tmp_path / "src/reactorbench/remediation").mkdir(parents=True)
    return tmp_path


def _config(run_name: str = "cli-run") -> PipelineConfig:
    return PipelineConfig.model_validate(
        {
            "pipeline_version": "0.4.0",
            "run_name": run_name,
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
            "maximum_run_bytes": 1024 * 1024,
            "maximum_process_rss_bytes": 256 * 1024**2,
            "stop_before_final_evaluation": True,
        }
    )


def _loaded(project_root: Path, config: PipelineConfig) -> cli.LoadedPipeline:
    path = project_root / "frozen.toml"
    path.write_text("frozen=true\n", encoding="ascii")
    return cli.LoadedPipeline(
        config=config,
        config_path=path,
        checksum_sha256=config_sha256(config),
    )


def _success_action(stage: str, calls: list[tuple[str, str]]) -> StageAction:
    def action(context: StageContext) -> StageOutcome:
        calls.append((stage, context.attempt_directory.name))
        return StageOutcome(summary=f"{stage} complete.")

    return action


def _actions(
    calls: list[tuple[str, str]], overrides: Mapping[str, StageAction] | None = None
) -> dict[str, StageAction]:
    selected = {} if overrides is None else dict(overrides)
    return {stage: selected.get(stage, _success_action(stage, calls)) for stage in PIPELINE_STAGES}


def _patch_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    project_root: Path,
    config: PipelineConfig,
    actions: Mapping[str, StageAction],
    source_commit: str = SOURCE_COMMIT,
    terminal_reviews: list[str] | None = None,
) -> cli.LoadedPipeline:
    loaded = _loaded(project_root, config)
    stop_path = project_root / config.run_root / config.run_name / "STOP_REQUESTED"

    def load_configuration(root: Path, relative_path: str) -> cli.LoadedPipeline:
        assert root == project_root
        assert relative_path == cli.DEFAULT_PIPELINE_CONFIG
        return loaded

    def build_actions(
        *, project_root: Path, config: PipelineConfig, source_commit: str
    ) -> Mapping[str, StageAction]:
        assert project_root == loaded.config_path.parent
        assert config == loaded.config
        assert 7 <= len(source_commit) <= 64
        return actions

    def stop_file(*, project_root: Path, config: PipelineConfig) -> Path:
        assert project_root == loaded.config_path.parent
        assert config == loaded.config
        return stop_path

    def request_stop(path: Path) -> None:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(b"stop requested\n")

    def archive_stop(path: Path) -> Path | None:
        if not path.exists():
            return None
        archive = path.parent / "stop-requests"
        archive.mkdir(exist_ok=True)
        destination = archive / "acknowledged-stop-request"
        path.replace(destination)
        return destination

    def write_terminal_review(
        *,
        project_root: Path,
        config: PipelineConfig,
        source_commit: str,
        state: PipelineState,
    ) -> object:
        assert 7 <= len(source_commit) <= 64
        status = state.status
        if terminal_reviews is not None:
            terminal_reviews.append(status)
        review_directory = project_root / config.run_root / config.run_name / "terminal-review"
        review_directory.mkdir(mode=0o750, exist_ok=True)
        manifest = review_directory / "review-bundle.json"
        summary = review_directory / "review-summary.md"
        manifest.write_bytes(b"{}\n")
        summary.write_text(f"# {status}\n", encoding="ascii")
        return SimpleNamespace(manifest_path=manifest, summary_path=summary)

    monkeypatch.setattr(cli, "_load_configuration", load_configuration)
    monkeypatch.setattr(cli, "_source_commit", lambda root: source_commit)
    monkeypatch.setattr(cli, "_build_stage_actions", build_actions)
    monkeypatch.setattr(cli, "_pipeline_stop_file", stop_file)
    monkeypatch.setattr(cli, "_request_pipeline_stop", request_stop)
    monkeypatch.setattr(cli, "_archive_pipeline_stop", archive_stop)
    monkeypatch.setattr(cli, "_write_terminal_review_bundle", write_terminal_review)
    return loaded


def _main(arguments: list[str], project_root: Path) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    result = cli.main(
        arguments,
        project_root=project_root,
        enforce_venv=False,
        stdout=stdout,
        stderr=stderr,
    )
    return result, stdout.getvalue(), stderr.getvalue()


def _create_store(project_root: Path, loaded: cli.LoadedPipeline) -> PipelineStore:
    return PipelineStore.create(
        project_root=project_root,
        config=loaded.config,
        pipeline_config_sha256=loaded.checksum_sha256,
        source_commit=SOURCE_COMMIT,
        command=("python", "-m", "reactorbench.remediation", "start"),
    )


def test_start_status_dry_run_and_completed_resume_are_understandable_and_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project_root(tmp_path)
    config = _config()
    calls: list[tuple[str, str]] = []
    _patch_runtime(
        monkeypatch,
        project_root=root,
        config=config,
        actions=_actions(calls),
    )

    code, output, errors = _main(["start"], root)
    assert code == cli.ExitCode.OK
    assert "Starting development pipeline" in output
    assert "Pipeline returned status: completed" in output
    assert errors == ""
    assert [stage for stage, _ in calls] == list(PIPELINE_STAGES)

    code, _, errors = _main(["start"], root)
    assert code == cli.ExitCode.STATE
    assert "use the resume command" in errors

    code, output, errors = _main(["status"], root)
    assert code == cli.ExitCode.OK
    assert "Pipeline status: completed" in output
    assert "Latest progress: completed" in output
    assert f"Stage position: {len(PIPELINE_STAGES)}/{len(PIPELINE_STAGES)}" in output
    assert "Elapsed seconds:" in output
    assert "Latest verified update UTC:" in output
    assert "Latest checkpoint:" in output
    assert "Stop request: none" in output
    assert errors == ""

    state_path = root / "work/cli-run/pipeline-state.json"
    before = state_path.read_bytes()
    code, output, errors = _main(["dry-run"], root)
    assert code == cli.ExitCode.OK
    assert "without changes: completed" in output
    assert state_path.read_bytes() == before
    assert errors == ""

    code, output, errors = _main(["resume"], root)
    assert code == cli.ExitCode.OK
    assert "Resuming development pipeline" in output
    assert state_path.read_bytes() == before
    assert len(calls) == len(PIPELINE_STAGES)
    assert errors == ""


def test_dry_run_before_start_validates_plan_without_creating_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project_root(tmp_path)
    config = _config()
    _patch_runtime(
        monkeypatch,
        project_root=root,
        config=config,
        actions=_actions([]),
    )

    code, output, errors = _main(["dry-run"], root)
    assert code == cli.ExitCode.OK
    assert "no training, data generation, or evaluation" in output
    assert f"Frozen stages: {len(PIPELINE_STAGES)}" in output
    assert "start would create a new run" in output
    assert not (root / "work/cli-run").exists()
    assert errors == ""


@pytest.mark.parametrize("command", ["start", "dry-run"])
def test_pipeline_preflight_runtime_failures_are_sanitized(
    command: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project_root(tmp_path)
    config = _config()
    loaded = _loaded(root, config)
    private_detail = "/Users/private/research/tokenizer-manifest.json"

    def fail_preflight(**_kwargs: object) -> Mapping[str, StageAction]:
        raise RuntimeError(f"source binding failed at {private_detail}")

    monkeypatch.setattr(cli, "_load_configuration", lambda *_args: loaded)
    monkeypatch.setattr(cli, "_source_commit", lambda _root: SOURCE_COMMIT)
    monkeypatch.setattr(
        cli,
        "_pipeline_module",
        lambda: SimpleNamespace(build_stage_actions=fail_preflight),
    )

    code, output, errors = _main([command], root)

    assert code == cli.ExitCode.CONFIGURATION
    assert output == ""
    assert "Pipeline preflight failed safely" in errors
    assert "Traceback" not in errors
    assert private_detail not in errors
    assert not (root / "work/cli-run").exists()


def test_stop_is_exclusive_and_resume_archives_stale_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project_root(tmp_path)
    config = _config()
    calls: list[tuple[str, str]] = []
    loaded = _patch_runtime(
        monkeypatch,
        project_root=root,
        config=config,
        actions=_actions(calls),
    )
    _create_store(root, loaded)

    code, output, errors = _main(["stop"], root)
    assert code == cli.ExitCode.OK
    assert "next safe boundary" in output
    assert errors == ""
    stop_path = root / "work/cli-run/STOP_REQUESTED"
    assert stop_path.read_bytes() == b"stop requested\n"

    code, output, _ = _main(["stop"], root)
    assert code == cli.ExitCode.OK
    assert "already pending" in output

    code, output, errors = _main(["resume"], root)
    assert code == cli.ExitCode.OK
    assert "Previous stop request archived" in output
    assert not stop_path.exists()
    assert (stop_path.parent / "stop-requests/acknowledged-stop-request").is_file()
    assert errors == ""

    code, _, errors = _main(["stop"], root)
    assert code == cli.ExitCode.STATE
    assert "not active" in errors


def test_blocked_stopped_and_failed_pipeline_results_use_distinct_exit_codes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project_root(tmp_path)
    terminal_reviews: list[str] = []

    def blocked(context: StageContext) -> StageOutcome:
        del context
        return StageOutcome(summary="Development gate blocked.", advancement_allowed=False)

    blocked_config = _config("blocked-run")
    _patch_runtime(
        monkeypatch,
        project_root=root,
        config=blocked_config,
        actions=_actions([], {"preflight": blocked}),
        terminal_reviews=terminal_reviews,
    )
    code, output, _ = _main(["start"], root)
    assert code == cli.ExitCode.BLOCKED
    assert "status: blocked" in output
    assert "Terminal review manifest:" in output

    def interrupted(context: StageContext) -> StageOutcome:
        del context
        raise KeyboardInterrupt

    stopped_config = _config("stopped-run")
    _patch_runtime(
        monkeypatch,
        project_root=root,
        config=stopped_config,
        actions=_actions([], {"preflight": interrupted}),
        terminal_reviews=terminal_reviews,
    )
    code, output, _ = _main(["start"], root)
    assert code == cli.ExitCode.STOPPED
    assert "status: stopped" in output
    assert "Terminal review manifest:" in output

    def failed(context: StageContext) -> StageOutcome:
        del context
        raise RuntimeError("private implementation detail")

    failed_config = _config("failed-run")
    _patch_runtime(
        monkeypatch,
        project_root=root,
        config=failed_config,
        actions=_actions([], {"preflight": failed}),
        terminal_reviews=terminal_reviews,
    )
    code, output, errors = _main(["start"], root)
    assert code == cli.ExitCode.PIPELINE_FAILED
    assert "Terminal review manifest:" in output
    assert "failed safely" in errors
    assert "private implementation detail" not in errors
    assert terminal_reviews == ["blocked", "stopped", "failed"]


def test_status_refuses_source_drift_and_tampered_or_symlink_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project_root(tmp_path)
    config = _config()
    loaded = _patch_runtime(
        monkeypatch,
        project_root=root,
        config=config,
        actions=_actions([]),
    )
    store = _create_store(root, loaded)

    monkeypatch.setattr(cli, "_source_commit", lambda root: "1234567")
    code, _, errors = _main(["status"], root)
    assert code == cli.ExitCode.CONFIGURATION
    assert "different config or source commit" in errors

    monkeypatch.setattr(cli, "_source_commit", lambda root: SOURCE_COMMIT)
    progress = store.run_directory / "status.json"
    progress.symlink_to(store.state_path)
    code, _, errors = _main(["status"], root)
    assert code == cli.ExitCode.STATE
    assert "Progress status is unsafe" in errors


def test_progress_io_failure_is_mapped_to_a_safe_cli_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project_root(tmp_path)
    config = _config("progress-failure")
    _patch_runtime(
        monkeypatch,
        project_root=root,
        config=config,
        actions=_actions([]),
    )
    private_detail = "/private/secret/progress.jsonl"

    def fail_enter(self: ProgressReporter) -> ProgressReporter:
        del self
        try:
            raise OSError(private_detail)
        except OSError as error:
            raise ProgressIOError("private progress detail") from error

    monkeypatch.setattr(ProgressReporter, "__enter__", fail_enter)
    code, output, errors = _main(["start"], root)

    assert code == cli.ExitCode.STATE
    assert output.startswith("Starting development pipeline:")
    assert "local state or boundary validation failed safely" in errors
    assert private_detail not in errors
    assert "private progress detail" not in errors
    assert "Traceback" not in errors


def test_status_rejects_a_canonical_snapshot_that_differs_from_the_event_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project_root(tmp_path)
    config = _config("status-tail-mismatch")
    _patch_runtime(
        monkeypatch,
        project_root=root,
        config=config,
        actions=_actions([]),
    )
    code, _, errors = _main(["start"], root)
    assert code == cli.ExitCode.OK
    assert errors == ""

    status_path = root / config.run_root / config.run_name / "status.json"
    snapshot = ProgressSnapshot.model_validate_json(status_path.read_bytes(), strict=True)
    changed = snapshot.model_copy(update={"message": "canonically rewritten status"})
    status_path.write_bytes(
        canonical_json_bytes(changed.model_dump(mode="json", round_trip=True)) + b"\n"
    )

    code, output, errors = _main(["status"], root)
    assert code == cli.ExitCode.STATE
    assert output == ""
    assert "does not match the verified event-log tail" in errors


def test_status_display_includes_work_metric_and_eta_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project_root(tmp_path)
    config = _config("detailed-status")
    _patch_runtime(
        monkeypatch,
        project_root=root,
        config=config,
        actions=_actions([]),
    )
    code, _, errors = _main(["start"], root)
    assert code == cli.ExitCode.OK
    assert errors == ""

    store = PipelineStore(
        root / config.run_root / config.run_name,
        maximum_state_bytes=config.maximum_status_bytes,
    )
    state = store.load_state()
    snapshot = ProgressSnapshot.model_validate_json(
        (store.run_directory / "status.json").read_bytes(), strict=True
    ).model_copy(
        update={
            "completed_units": 40,
            "total_units": 100,
            "latest_metric": ProgressMetric(name="validation_loss", value=0.25),
            "eta_seconds": 90.0,
        }
    )
    output = io.StringIO()
    cli._print_status(state, snapshot, output=output)

    rendered = output.getvalue()
    assert "Work completed: 40/100" in rendered
    assert "Latest metric: validation_loss=0.25" in rendered
    assert "Estimated seconds remaining: 90.0" in rendered


def test_final_evaluation_calls_executor_once_only_after_visible_prerequisites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project_root(tmp_path)
    config = _config()
    calls: list[tuple[str, str]] = []
    _patch_runtime(
        monkeypatch,
        project_root=root,
        config=config,
        actions=_actions(calls),
    )
    final_calls: list[tuple[str, bool]] = []

    def run_final(
        *,
        project_root: Path,
        config: PipelineConfig,
        source_commit: str,
        explicit_confirmation: bool,
    ) -> object:
        assert source_commit == SOURCE_COMMIT
        final_calls.append((config.run_name, explicit_confirmation))
        ledger = project_root / config.run_root / config.run_name / cli.FINAL_ACCESS_LEDGER
        ledger.write_bytes(b"claimed\n")
        return object()

    monkeypatch.setattr(cli, "_run_final_evaluation", run_final)

    code, _, errors = _main(["final-evaluation"], root)
    assert code == cli.ExitCode.FINAL_EVALUATION_LOCKED
    assert "explicit confirmation" in errors
    assert final_calls == []

    code, _, errors = _main(
        ["final-evaluation", "--confirm-final-evaluation", "--historical-golden"], root
    )
    assert code == cli.ExitCode.FINAL_EVALUATION_LOCKED
    assert "Historical golden" in errors
    assert final_calls == []

    code, _, errors = _main(["final-evaluation", "--confirm-final-evaluation"], root)
    assert code == cli.ExitCode.FINAL_EVALUATION_LOCKED
    assert "evidence is unavailable" in errors
    assert final_calls == []

    assert _main(["start"], root)[0] == cli.ExitCode.OK
    code, _, errors = _main(["final-evaluation", "--confirm-final-evaluation"], root)
    assert code == cli.ExitCode.FINAL_EVALUATION_LOCKED
    assert "markers are absent" in errors
    assert final_calls == []

    run_directory = root / "work/cli-run"
    for name in (
        cli.FINAL_READY_MARKER,
        cli.OWNER_REVIEW_MARKER,
        cli.FRESH_EXTENSION_MARKER,
    ):
        (run_directory / name).write_bytes(b"{}\n")
    code, output, errors = _main(["final-evaluation", "--confirm-final-evaluation"], root)
    assert code == cli.ExitCode.OK
    assert "Final evaluation completed" in output
    assert errors == ""
    assert final_calls == [(config.run_name, True)]
    assert (run_directory / cli.FINAL_ACCESS_LEDGER).is_file()

    code, _, errors = _main(["final-evaluation", "--confirm-final-evaluation"], root)
    assert code == cli.ExitCode.FINAL_EVALUATION_LOCKED
    assert "already exists" in errors
    assert final_calls == [(config.run_name, True)]


def test_locked_final_executor_is_bounded_at_the_main_command_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project_root(tmp_path)
    config = _config()
    _patch_runtime(
        monkeypatch,
        project_root=root,
        config=config,
        actions=_actions([]),
    )
    assert _main(["start"], root)[0] == cli.ExitCode.OK
    run_directory = root / "work/cli-run"
    for name in (
        cli.FINAL_READY_MARKER,
        cli.OWNER_REVIEW_MARKER,
        cli.FRESH_EXTENSION_MARKER,
    ):
        (run_directory / name).write_bytes(b"{}\n")

    class Blocked(RuntimeError):
        pass

    sensitive_detail = "/private/internal/checkpoint.safetensors"

    def locked_executor(**_kwargs: object) -> object:
        raise Blocked(sensitive_detail)

    monkeypatch.setattr(
        cli,
        "_pipeline_module",
        lambda: SimpleNamespace(
            run_final_evaluation=locked_executor,
            FinalEvaluationBlockedError=Blocked,
        ),
    )

    code, output, errors = _main(
        ["final-evaluation", "--confirm-final-evaluation"],
        root,
    )

    assert code == cli.ExitCode.FINAL_EVALUATION_LOCKED
    assert output == ""
    assert "access remains locked" in errors
    assert sensitive_detail not in errors
    assert "Traceback" not in errors
    assert not (run_directory / cli.FINAL_ACCESS_LEDGER).exists()


def test_root_path_venv_and_config_argument_boundaries_fail_safely(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project_root(tmp_path)
    assert cli.find_project_root(root) == root.resolve()
    with pytest.raises(cli.CliFailure, match="safe relative"):
        cli._safe_project_file(root, "../outside.toml")
    linked = root / "linked.toml"
    target = root / "target.toml"
    target.write_text("x=1\n", encoding="ascii")
    linked.symlink_to(target)
    with pytest.raises(cli.CliFailure, match="symbolic link"):
        cli._safe_project_file(root, "linked.toml")

    config = _config()
    _patch_runtime(
        monkeypatch,
        project_root=root,
        config=config,
        actions=_actions([]),
    )
    stdout = io.StringIO()
    stderr = io.StringIO()
    code = cli.main(
        ["dry-run"],
        project_root=root,
        enforce_venv=True,
        stdout=stdout,
        stderr=stderr,
    )
    assert code == cli.ExitCode.CONFIGURATION
    assert "Project .venv is missing" in stderr.getvalue()


def test_pipeline_api_adapters_enforce_shapes_and_map_final_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project_root(tmp_path)
    config = _config()
    loaded = _loaded(root, config)
    store = _create_store(root, loaded)
    state = store.load_state()
    stop_path = store.run_directory / "STOP_REQUESTED"
    calls: list[str] = []

    class Blocked(RuntimeError):
        pass

    def build_actions(
        *, project_root: Path, config: PipelineConfig, source_commit: str
    ) -> Mapping[str, StageAction]:
        assert project_root == root
        assert config == loaded.config
        assert source_commit == SOURCE_COMMIT
        return _actions([])

    def stop_file(*, project_root: Path, config: PipelineConfig) -> Path:
        assert project_root == root
        assert config == loaded.config
        return stop_path

    def request_stop(path: Path) -> None:
        assert path == stop_path
        calls.append("stop")

    def archive_stop(path: Path) -> Path | None:
        assert path == stop_path
        calls.append("archive")
        return None

    review_directory = store.run_directory / "adapter-review"
    review_directory.mkdir()
    manifest = review_directory / "manifest.json"
    summary = review_directory / "summary.md"
    manifest.write_bytes(b"{}\n")
    summary.write_text("# review\n", encoding="ascii")

    def write_review(
        *,
        project_root: Path,
        config: PipelineConfig,
        source_commit: str,
        state: PipelineState,
    ) -> object:
        assert (project_root, config, source_commit, state.status) == (
            root,
            loaded.config,
            SOURCE_COMMIT,
            "ready",
        )
        return SimpleNamespace(manifest_path=manifest, summary_path=summary)

    sentinel = object()

    def run_final(
        *,
        project_root: Path,
        config: PipelineConfig,
        source_commit: str,
        explicit_confirmation: bool,
    ) -> object:
        assert (project_root, config, source_commit, explicit_confirmation) == (
            root,
            loaded.config,
            SOURCE_COMMIT,
            True,
        )
        return sentinel

    module = SimpleNamespace(
        build_stage_actions=build_actions,
        pipeline_stop_file=stop_file,
        request_pipeline_stop=request_stop,
        archive_pipeline_stop=archive_stop,
        write_terminal_review_bundle=write_review,
        run_final_evaluation=run_final,
        FinalEvaluationBlockedError=Blocked,
    )
    monkeypatch.setattr(cli, "_pipeline_module", lambda: module)

    assert (
        tuple(
            cli._build_stage_actions(
                project_root=root,
                config=config,
                source_commit=SOURCE_COMMIT,
            )
        )
        == PIPELINE_STAGES
    )
    assert cli._pipeline_stop_file(project_root=root, config=config) == stop_path
    cli._request_pipeline_stop(stop_path)
    assert cli._archive_pipeline_stop(stop_path) is None
    review = cli._write_terminal_review_bundle(
        project_root=root,
        config=config,
        source_commit=SOURCE_COMMIT,
        state=state,
    )
    assert review.manifest_path == manifest
    assert (
        cli._run_final_evaluation(
            project_root=root,
            config=config,
            source_commit=SOURCE_COMMIT,
            explicit_confirmation=True,
        )
        is sentinel
    )
    assert calls == ["stop", "archive"]

    module.run_final_evaluation = lambda **_kwargs: (_ for _ in ()).throw(Blocked())
    with pytest.raises(cli.CliFailure) as blocked:
        cli._run_final_evaluation(
            project_root=root,
            config=config,
            source_commit=SOURCE_COMMIT,
            explicit_confirmation=True,
        )
    assert blocked.value.exit_code == cli.ExitCode.FINAL_EVALUATION_LOCKED

    module.run_final_evaluation = lambda **_kwargs: (_ for _ in ()).throw(RuntimeError())
    with pytest.raises(cli.CliFailure) as failed:
        cli._run_final_evaluation(
            project_root=root,
            config=config,
            source_commit=SOURCE_COMMIT,
            explicit_confirmation=True,
        )
    assert failed.value.exit_code == cli.ExitCode.STATE

    monkeypatch.setattr(cli, "_pipeline_module", lambda: SimpleNamespace())
    with pytest.raises(cli.CliFailure, match="action factory"):
        cli._build_stage_actions(
            project_root=root,
            config=config,
            source_commit=SOURCE_COMMIT,
        )
    with pytest.raises(cli.CliFailure, match="stop control"):
        cli._pipeline_stop_file(project_root=root, config=config)
    with pytest.raises(cli.CliFailure, match="review writer"):
        cli._write_terminal_review_bundle(
            project_root=root,
            config=config,
            source_commit=SOURCE_COMMIT,
            state=state,
        )
    with pytest.raises(cli.CliFailure, match="executor"):
        cli._run_final_evaluation(
            project_root=root,
            config=config,
            source_commit=SOURCE_COMMIT,
            explicit_confirmation=True,
        )


def test_low_level_path_source_stop_and_progress_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project_root(tmp_path)
    for unsafe in ("", "/absolute", ".", "a/../b", "a\\b", "é.toml"):
        with pytest.raises(cli.CliFailure):
            cli._safe_relative_argument(unsafe)
    with pytest.raises(cli.CliFailure, match="missing"):
        cli._safe_project_file(root, "missing.toml")
    with pytest.raises(TypeError):
        cli.find_project_root("not-a-path")  # type: ignore[arg-type]
    untrusted = tmp_path / "untrusted"
    untrusted.mkdir()
    with pytest.raises(cli.CliFailure, match="trusted ReactorBench"):
        cli.find_project_root(untrusted)

    completed = subprocess.CompletedProcess(
        args=("git",), returncode=1, stdout="not-a-commit\n", stderr=""
    )
    monkeypatch.setattr(
        "reactorbench.remediation.cli.subprocess.run",
        lambda *_args, **_kwargs: completed,
    )
    with pytest.raises(cli.CliFailure, match="verify the local Git commit"):
        cli._source_commit(root)

    marker = root / "marker"
    assert cli._stop_requested(marker) is False
    marker.write_bytes(b"")
    with pytest.raises(cli.CliFailure, match="invalid"):
        cli._stop_requested(marker)
    marker.unlink()
    target = root / "target"
    target.write_bytes(b"safe\n")
    marker.symlink_to(target)
    with pytest.raises(cli.CliFailure, match="unsafe"):
        cli._stop_requested(marker)

    config = _config()
    loaded = _loaded(root, config)
    store = _create_store(root, loaded)
    assert cli._progress_snapshot(store, config) is None
    progress = store.run_directory / "status.json"
    progress.write_bytes(b"{}\n")
    with pytest.raises(cli.CliFailure, match="strict verification"):
        cli._progress_snapshot(store, config)


def test_ready_status_stop_races_and_incomplete_final_remain_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project_root(tmp_path)
    config = _config()
    loaded = _patch_runtime(
        monkeypatch,
        project_root=root,
        config=config,
        actions=_actions([]),
    )
    _create_store(root, loaded)

    code, output, errors = _main(["status"], root)
    assert code == cli.ExitCode.OK
    assert "Progress reporter: not started" in output
    assert "Stop request: none" in output
    assert errors == ""

    stop_path = root / "work/cli-run/STOP_REQUESTED"
    stop_path.write_bytes(b"pending\n")
    code, output, _ = _main(["status"], root)
    assert code == cli.ExitCode.OK
    assert "Stop request: pending" in output
    stop_path.unlink()

    def raced(_path: Path) -> None:
        raise FileExistsError

    monkeypatch.setattr(cli, "_request_pipeline_stop", raced)
    code, output, _ = _main(["stop"], root)
    assert code == cli.ExitCode.OK
    assert "already pending" in output

    def invalid(_path: Path) -> None:
        raise ValueError

    monkeypatch.setattr(cli, "_request_pipeline_stop", invalid)
    code, _, errors = _main(["stop"], root)
    assert code == cli.ExitCode.STATE
    assert "could not be written safely" in errors

    code, _, errors = _main(["final-evaluation", "--confirm-final-evaluation"], root)
    assert code == cli.ExitCode.FINAL_EVALUATION_LOCKED
    assert "not complete" in errors


def test_default_committed_pipeline_config_and_references_are_strictly_bound() -> None:
    root = Path(__file__).resolve().parents[2]
    loaded = cli._load_configuration(root, cli.DEFAULT_PIPELINE_CONFIG)
    assert loaded.config.pipeline_version == "0.4.0"
    assert loaded.config.run_name == "phase6-remediation-v0.4.0-targeted-02"
    assert loaded.config.stop_before_final_evaluation is True
    assert loaded.checksum_sha256 == config_sha256(loaded.config)
    assert loaded.config.v03_config_path.endswith("phase6-remediation-v0.3.2-focused.toml")
    assert loaded.config.reuse_v02_prefix is not None


def test_shell_wrappers_are_executable_syntax_checked_bounded_and_final_locked() -> None:
    root = Path(__file__).resolve().parents[2]
    paths = tuple(root / "scripts" / name for name in SCRIPTS)
    for path in paths:
        assert path.stat().st_mode & 0o111
        text = path.read_text(encoding="utf-8")
        assert "set -euo pipefail" in text
        assert 'project_venv="$project_root/.venv"' in text
        assert 'python_executable="$project_venv/bin/python"' in text
    syntax = subprocess.run(  # noqa: S603 - fixed /bin/bash and repository scripts
        ("/bin/bash", "-n", *(str(path) for path in paths)),
        check=False,
        capture_output=True,
        text=True,
    )
    assert syntax.returncode == 0, syntax.stderr

    for path in paths[:-1]:
        rejected = subprocess.run(  # noqa: S603 - fixed /bin/bash and repository scripts
            ("/bin/bash", str(path), "unexpected"),
            check=False,
            capture_output=True,
            text=True,
        )
        assert rejected.returncode == cli.ExitCode.USAGE
    evaluation = paths[-1]
    no_confirmation = subprocess.run(  # noqa: S603 - fixed /bin/bash and repository script
        ("/bin/bash", str(evaluation)),
        check=False,
        capture_output=True,
        text=True,
    )
    assert no_confirmation.returncode == cli.ExitCode.USAGE
    historical = subprocess.run(  # noqa: S603 - fixed /bin/bash and repository script
        ("/bin/bash", str(evaluation), "--historical-golden"),
        check=False,
        capture_output=True,
        text=True,
    )
    assert historical.returncode == cli.ExitCode.FINAL_EVALUATION_LOCKED
    assert "prohibited" in historical.stderr
