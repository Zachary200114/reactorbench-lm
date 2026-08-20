"""From-scratch decoder-only Transformer and safe checkpoint interfaces."""

from .checkpoint import CheckpointManifest, load_checkpoint, save_checkpoint
from .config import (
    Phase4Config,
    Phase4Paths,
    SmokeTrainingConfig,
    TokenizerConfig,
    TransformerConfig,
    load_phase4_config,
    resolve_project_path,
)
from .transformer import (
    AttentionCache,
    CausalBatch,
    TransformerLM,
    causal_language_model_loss,
    exact_parameter_count,
    initialized_model,
    shift_next_token_targets,
)

__all__ = [
    "AttentionCache",
    "CausalBatch",
    "CheckpointManifest",
    "Phase4Config",
    "Phase4Paths",
    "SmokeTrainingConfig",
    "TokenizerConfig",
    "TransformerConfig",
    "TransformerLM",
    "causal_language_model_loss",
    "exact_parameter_count",
    "initialized_model",
    "load_checkpoint",
    "load_phase4_config",
    "resolve_project_path",
    "save_checkpoint",
    "shift_next_token_targets",
]
