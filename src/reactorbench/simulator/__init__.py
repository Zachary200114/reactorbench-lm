"""Phase 2 deterministic Aster-A simulation entry points."""

from .content_guard import (
    ContentFinding,
    ProhibitedContentError,
    assert_no_prohibited_content,
    scan_prohibited_content,
)
from .core import (
    ASTER_A_SPEC,
    GENERATOR_VERSION,
    SimulationTrace,
    UnsupportedScenarioError,
    build_load_transient_scenario,
    build_pump_degradation_scenario,
    build_sensor_drift_scenario,
    build_sensor_noise_scenario,
    build_sensor_stuck_load_scenario,
    build_stable_scenario,
    generate_trace,
)

__all__ = [
    "ASTER_A_SPEC",
    "GENERATOR_VERSION",
    "ContentFinding",
    "ProhibitedContentError",
    "SimulationTrace",
    "UnsupportedScenarioError",
    "assert_no_prohibited_content",
    "build_load_transient_scenario",
    "build_pump_degradation_scenario",
    "build_sensor_drift_scenario",
    "build_sensor_noise_scenario",
    "build_sensor_stuck_load_scenario",
    "build_stable_scenario",
    "generate_trace",
    "scan_prohibited_content",
]
