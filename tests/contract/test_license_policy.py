from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_zero_clause_license_matches_the_owner_policy_and_package_metadata() -> None:
    license_text = (ROOT / "LICENSE").read_text("ascii")
    normalized = " ".join(license_text.split())
    assert license_text.startswith("BSD Zero Clause License\n\nCopyright (C) 2026 Zachary Ryan\n")
    assert (
        "Permission to use, copy, modify, and/or distribute this software for any "
        "purpose with or without fee is hereby granted."
    ) in normalized
    assert 'THE SOFTWARE IS PROVIDED "AS IS"' in normalized
    assert "provided that" not in normalized.lower()

    configuration = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))
    assert configuration["project"]["license"] == "0BSD"
    assert configuration["project"]["license-files"] == ["LICENSE"]
    assert "/LICENSE" in configuration["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]


def test_readme_explains_that_credit_is_optional_without_relicensing_dependencies() -> None:
    readme = (ROOT / "README.md").read_text("utf-8")
    normalized = " ".join(readme.split())
    assert "## License" in readme
    assert "[0BSD License](LICENSE)" in readme
    assert "Credit is appreciated, but it is not required." in normalized
    assert "Third-party dependencies and referenced material" in normalized
