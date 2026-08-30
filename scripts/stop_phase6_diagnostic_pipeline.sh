#!/usr/bin/env bash
set -euo pipefail

if (( $# != 0 )); then
  echo "Usage: scripts/stop_phase6_diagnostic_pipeline.sh" >&2
  exit 2
fi
script_source=${BASH_SOURCE[0]}
if [[ -L "$script_source" ]]; then
  echo "Refusing to run through a symbolic-link script." >&2
  exit 4
fi
script_dir=$(CDPATH= cd -- "$(dirname -- "$script_source")" && pwd -P)
project_root=$(CDPATH= cd -- "$script_dir/.." && pwd -P)
python_executable="$project_root/.venv/bin/python"
config="configs/experiments/phase6-remediation-pipeline-v0.4.0-targeted-05-diagnostic-01.toml"
if [[ ! -x "$python_executable" ]]; then
  echo "Project .venv Python is not executable." >&2
  exit 4
fi
cd -- "$project_root"
exec "$python_executable" -m reactorbench.remediation stop --config "$config"
