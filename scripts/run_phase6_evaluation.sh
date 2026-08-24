#!/usr/bin/env bash
set -euo pipefail

for argument in "$@"; do
  if [[ $argument == "--historical-golden" ]]; then
    echo "Historical golden evidence is prohibited for the fresh final evaluation." >&2
    exit 10
  fi
done
if (( $# != 1 )) || [[ $1 != "--confirm-final-evaluation" ]]; then
  echo "Usage: scripts/run_phase6_evaluation.sh --confirm-final-evaluation" >&2
  echo "This future-only command also requires ready, owner-review, and fresh-extension markers." >&2
  exit 2
fi

script_source=${BASH_SOURCE[0]}
if [[ -L "$script_source" ]]; then
  echo "Refusing to run through a symbolic-link script." >&2
  exit 4
fi
script_dir=$(CDPATH= cd -- "$(dirname -- "$script_source")" && pwd -P)
project_root=$(CDPATH= cd -- "$script_dir/.." && pwd -P)
project_venv="$project_root/.venv"
python_executable="$project_venv/bin/python"

if [[ -L "$project_venv" || ! -d "$project_venv" || -L "$project_venv/bin" ]]; then
  echo "Project .venv is missing or unsafe." >&2
  exit 4
fi
if [[ ! -x "$python_executable" ]]; then
  echo "Project .venv Python is not executable." >&2
  exit 4
fi

cd -- "$project_root"
exec "$python_executable" -m reactorbench.remediation final-evaluation \
  --confirm-final-evaluation
