"""Verify built distributions and an isolated, no-network wheel installation."""

from __future__ import annotations

import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIST_DIRECTORY = ROOT / "dist"
PACKAGE_DATA_PREFIX = "reactorbench/_data"


def _project_version() -> str:
    with (ROOT / "pyproject.toml").open("rb") as pyproject_file:
        pyproject = tomllib.load(pyproject_file)
    version = pyproject["project"]["version"]
    if not isinstance(version, str):
        raise TypeError("project.version must be a string")
    return version


def _expected_resource_files() -> dict[str, bytes]:
    expected = {
        f"{PACKAGE_DATA_PREFIX}/configs/default.toml": (
            ROOT / "configs" / "default.toml"
        ).read_bytes()
    }
    snapshot_root = ROOT / "schemas" / "aster" / "v0"
    for path in snapshot_root.rglob("*"):
        if path.is_file():
            relative_path = path.relative_to(snapshot_root).as_posix()
            expected[f"{PACKAGE_DATA_PREFIX}/schemas/aster/v0/{relative_path}"] = path.read_bytes()
    return expected


def _verify_wheel(wheel_path: Path, expected_resources: dict[str, bytes]) -> None:
    with zipfile.ZipFile(wheel_path) as wheel:
        members = wheel.namelist()
        if len(members) != len(set(members)):
            raise AssertionError("wheel contains duplicate archive members")
        packaged_resources = {
            member for member in members if member.startswith(f"{PACKAGE_DATA_PREFIX}/")
        }
        if packaged_resources != set(expected_resources):
            raise AssertionError("wheel package resources do not match reviewed root assets")
        for member, expected_bytes in expected_resources.items():
            if wheel.read(member) != expected_bytes:
                raise AssertionError(f"wheel resource drifted from its reviewed source: {member}")


def _verify_sdist(
    sdist_path: Path,
    *,
    version: str,
    expected_resources: dict[str, bytes],
) -> None:
    archive_root = f"reactorbench_lm-{version}"
    expected_sources = {
        f"{archive_root}/{member.removeprefix(f'{PACKAGE_DATA_PREFIX}/')}": content
        for member, content in expected_resources.items()
    }
    with tarfile.open(sdist_path, mode="r:gz") as sdist:
        members = sdist.getmembers()
        member_names = [member.name for member in members]
        if len(member_names) != len(set(member_names)):
            raise AssertionError("sdist contains duplicate archive members")
        if any(
            name == f"{archive_root}/tests" or name.startswith(f"{archive_root}/tests/")
            for name in member_names
        ):
            raise AssertionError(
                "sdist must not ship tests without the complete research fixture tree"
            )
        members_by_name = {member.name: member for member in members}
        for member_name, expected_bytes in expected_sources.items():
            member = members_by_name.get(member_name)
            if member is None:
                raise AssertionError(f"sdist is missing a wheel source asset: {member_name}")
            extracted = sdist.extractfile(member)
            if extracted is None or extracted.read() != expected_bytes:
                raise AssertionError(f"sdist asset drifted from its reviewed source: {member_name}")


def _verify_isolated_install(wheel_path: Path) -> None:
    verification = """
from importlib.resources import as_file
from pathlib import Path
import sys

installation = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(installation))

import reactorbench
from reactorbench.config import load_project_config
from reactorbench.resources import canonical_schema_snapshot_resource, default_config_resource
from reactorbench.schemas import load_snapshot, schema_documents

package_path = Path(reactorbench.__file__).resolve()
assert package_path.is_relative_to(installation)
with as_file(default_config_resource()) as config_path:
    assert load_project_config(config_path).project_name == "ReactorBench-LM"
with as_file(canonical_schema_snapshot_resource()) as snapshot_path:
    documents, _manifest = load_snapshot(snapshot_path)
    assert documents == schema_documents()
"""
    with tempfile.TemporaryDirectory(prefix="reactorbench-wheel-") as temporary_directory:
        installation = Path(temporary_directory) / "site-packages"
        base_python = Path(sys.base_prefix) / (
            "python.exe" if sys.platform == "win32" else "bin/python"
        )
        subprocess.run(  # noqa: S603
            [
                str(base_python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-cache-dir",
                "--no-deps",
                "--no-index",
                "--target",
                str(installation),
                str(wheel_path),
            ],
            check=True,
        )
        subprocess.run(  # noqa: S603
            [sys.executable, "-I", "-c", verification, str(installation)],
            check=True,
            cwd=temporary_directory,
        )


def main() -> None:
    version = _project_version()
    wheel_path = DIST_DIRECTORY / f"reactorbench_lm-{version}-py3-none-any.whl"
    sdist_path = DIST_DIRECTORY / f"reactorbench_lm-{version}.tar.gz"
    if not wheel_path.is_file() or not sdist_path.is_file():
        raise FileNotFoundError("expected wheel and sdist artifacts; run the build target first")

    expected_resources = _expected_resource_files()
    _verify_wheel(wheel_path, expected_resources)
    _verify_sdist(sdist_path, version=version, expected_resources=expected_resources)
    _verify_isolated_install(wheel_path)


if __name__ == "__main__":
    main()
