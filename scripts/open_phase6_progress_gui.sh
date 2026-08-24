#!/usr/bin/env bash
set -euo pipefail

if (( $# > 1 )); then
  echo "Usage: scripts/open_phase6_progress_gui.sh [--smoke]" >&2
  exit 2
fi
if (( $# == 1 )) && [[ $1 != "--smoke" ]]; then
  echo "Usage: scripts/open_phase6_progress_gui.sh [--smoke]" >&2
  exit 2
fi

script_source=${BASH_SOURCE[0]}
if [[ -L "$script_source" ]]; then
  echo "Refusing to run through a symbolic-link script." >&2
  exit 4
fi
script_dir=$(CDPATH= cd -- "$(dirname -- "$script_source")" && pwd -P)
project_root=$(CDPATH= cd -- "$script_dir/.." && pwd -P)
gui_source="$project_root/src/reactorbench/remediation/local_monitor.py"
swift_source="$project_root/src/reactorbench/remediation/Phase6RunMonitor.swift"
info_plist="$project_root/src/reactorbench/remediation/Phase6RunMonitor-Info.plist"
project_python="$project_root/.venv/bin/python"
controller_wrapper="$project_root/scripts/phase6_monitor_controller.sh"
swift_compiler=/usr/bin/swiftc
open_tool=/usr/bin/open

if [[ -L "$gui_source" || ! -f "$gui_source" ]]; then
  echo "The local Phase 6 monitor controller is missing or unsafe." >&2
  exit 4
fi
if [[ -L "$swift_source" || ! -f "$swift_source" ]]; then
  echo "The local Phase 6 monitor window is missing or unsafe." >&2
  exit 4
fi
if [[ -L "$info_plist" || ! -f "$info_plist" ]]; then
  echo "The local Phase 6 monitor metadata is missing or unsafe." >&2
  exit 4
fi
if [[ ! -x "$project_python" ]]; then
  echo "Project .venv Python is not executable." >&2
  exit 4
fi
if [[ -L "$controller_wrapper" || ! -x "$controller_wrapper" ]]; then
  echo "The local Phase 6 monitor wrapper is missing or unsafe." >&2
  exit 4
fi
if [[ -L "$swift_compiler" || ! -x "$swift_compiler" ]]; then
  echo "The trusted macOS Swift compiler is unavailable." >&2
  exit 4
fi
if [[ -L "$open_tool" || ! -x "$open_tool" ]]; then
  echo "The trusted macOS application launcher is unavailable." >&2
  exit 4
fi

if (( $# == 1 )); then
  exec "$controller_wrapper" --smoke
fi

monitor_temp=$(mktemp -d /tmp/reactorbench-phase6-monitor.XXXXXX)
monitor_app="$monitor_temp/ReactorBench-LM Phase 6 Monitor.app"
monitor_binary="$monitor_app/Contents/MacOS/ReactorBenchPhase6Monitor"

cleanup_monitor_temp() {
  rm -rf -- "$monitor_temp"
}
trap cleanup_monitor_temp EXIT HUP INT TERM

mkdir -p "$monitor_app/Contents/MacOS"
cp "$info_plist" "$monitor_app/Contents/Info.plist"
"$swift_compiler" \
  -module-cache-path "$monitor_temp/module-cache" \
  -o "$monitor_binary" \
  "$swift_source"
"$open_tool" -W -n "$monitor_app"
