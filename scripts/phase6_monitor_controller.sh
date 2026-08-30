#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )); then
  echo "Usage: scripts/phase6_monitor_controller.sh <allowlisted-action>" >&2
  exit 2
fi
case $1 in
  --smoke | --snapshot-json | --readiness-check | --start-detached | --request-stop | --resume-detached | --open-finder | --diagnostic-snapshot-json | --diagnostic-readiness-check | --diagnostic-start-detached | --diagnostic-request-stop | --diagnostic-resume-detached | --diagnostic-open-finder)
    ;;
  *)
    echo "Command refused: monitor action is not allowlisted." >&2
    exit 2
    ;;
esac

script_source=${BASH_SOURCE[0]}
if [[ -L "$script_source" ]]; then
  echo "Refusing to run through a symbolic-link script." >&2
  exit 4
fi
script_dir=$(CDPATH= cd -- "$(dirname -- "$script_source")" && pwd -P)
project_root=$(CDPATH= cd -- "$script_dir/.." && pwd -P)
project_python="$project_root/.venv/bin/python"
controller="$project_root/src/reactorbench/remediation/local_monitor.py"

if [[ ! -x "$project_python" ]]; then
  echo "Project .venv Python is not executable." >&2
  exit 4
fi
if [[ -L "$controller" || ! -f "$controller" ]]; then
  echo "The local Phase 6 monitor controller is missing or unsafe." >&2
  exit 4
fi

"$project_python" "$controller" "$1"
