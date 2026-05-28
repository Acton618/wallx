from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class TokenScorePredictor(nn.Module):
    """Small per-token scorer used by VisPruner predictor strategies."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim or max(1, input_dim // 4))
        self.dropout = float(dropout)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, token_features: torch.Tensor) -> torch.Tensor:
        return self.net(token_features).squeeze(-1)

    def config_dict(self) -> Dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
        }


def load_token_score_predictor(
    checkpoint_path: str,
    default_input_dim: int,
    default_hidden_dim: Optional[int] = None,
    default_dropout: float = 0.0,
    map_location: str = "cpu",
    strict: bool = True,
) -> TokenScorePredictor:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        cfg = checkpoint.get("config", {}) or {}
        state_dict = checkpoint["model_state_dict"]
    else:
        cfg = {}
        state_dict = checkpoint

    model = TokenScorePredictor(
        input_dim=int(cfg.get("input_dim", default_input_dim)),
        hidden_dim=cfg.get("hidden_dim", default_hidden_dim),
        dropout=float(cfg.get("dropout", default_dropout)),
    )
    model.load_state_dict(state_dict, strict=strict)
    return model
