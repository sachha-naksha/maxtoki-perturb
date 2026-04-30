"""BioNeMo MaxToki PyTorch adapter (backbone + temporal head).

The HF release is pretraining-only - it has no temporal head. The published
BioNeMo distcp checkpoint is the only place the temporal head lives. This
module wraps both into a single ``MaxTokiTemporal`` module exposing:

    backbone(token_ids) -> hidden states (B, L, H)
    temporal_head(hidden) -> temporal scalar / vector (B, T)

Two load paths are supported:

    1. ``from_distcp(distcp_dir, variant)`` - loads the BioNeMo Megatron
       distributed checkpoint, including the temporal regression head.
       Requires bionemo-framework + Megatron-LM + apex/te installed.

    2. ``from_hf_with_temporal(hf_repo, head_state_dict)`` - loads the HF
       Llama backbone via transformers, then loads a separately-saved
       temporal head state_dict on top. Useful if you've already converted
       the distcp once.

NOTE: ``from_distcp`` is a stub. Wiring up the exact NeMo / Megatron loading
needs the upstream maxToki repo (in particular the temporal-head module
definition and the BioNeMo recipe yaml). Drop the upstream repo into the
working tree and I'll fill in the TODOs flagged below.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn

Variant = Literal["217m", "1b"]


@dataclass
class MaxTokiConfig:
    variant: Variant
    vocab_size: int = 20275
    hidden_size: int = 1232          # 217m default; 1b -> 2304
    num_layers: int = 11             # 217m -> 11; 1b -> 20
    num_heads: int = 8               # 217m -> 8; 1b -> 16
    num_kv_heads: int = 8            # 217m -> 8; 1b -> 8 (GQA)
    head_dim: int = 154              # 217m -> 154; 1b -> 144
    rope: str = "standard"           # 217m -> standard; 1b -> "llama3"
    max_seq_len: int = 4096
    pad_id: int = 0
    bos_id: int = 2
    eos_id: int = 3
    # Temporal head spec (filled in from upstream repo):
    temporal_pooling: str = "eos"     # how the head pools backbone hidden states
    temporal_out_dim: int = 1         # 1 if scalar age, else trajectory dim

    @classmethod
    def for_variant(cls, variant: Variant) -> "MaxTokiConfig":
        if variant == "217m":
            return cls(variant="217m")
        if variant == "1b":
            return cls(
                variant="1b",
                hidden_size=2304,
                num_layers=20,
                num_heads=16,
                num_kv_heads=8,
                head_dim=144,
                rope="llama3",
            )
        raise ValueError(f"unknown variant: {variant}")


class TemporalHead(nn.Module):
    """Temporal regression head as published in MaxToki.

    Architecture placeholder - matches the typical BioNeMo regression head
    shape. Replace ``__init__`` and ``forward`` once the upstream module is
    inspected (see TODO in ``MaxTokiTemporal.from_distcp``).
    """

    def __init__(self, hidden_size: int, out_dim: int = 1, pooling: str = "eos"):
        super().__init__()
        self.pooling = pooling
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, out_dim),
        )

    def pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        # hidden: (B, L, H)
        if self.pooling == "eos":
            # take the last non-pad token per row
            if attention_mask is None:
                return hidden[:, -1]
            lengths = attention_mask.long().sum(dim=1) - 1
            return hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths]
        if self.pooling == "mean":
            if attention_mask is None:
                return hidden.mean(dim=1)
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        raise ValueError(f"unknown pooling: {self.pooling}")

    def forward(
        self, hidden: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        pooled = self.pool(hidden, attention_mask)
        return self.proj(pooled)


class MaxTokiTemporal(nn.Module):
    def __init__(self, backbone: nn.Module, head: TemporalHead, config: MaxTokiConfig):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.config = config

    @torch.inference_mode()
    def temporal(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run backbone + temporal head, return (B, temporal_out_dim)."""
        hidden = self._backbone_hidden(input_ids, attention_mask)
        return self.head(hidden, attention_mask)

    def _backbone_hidden(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None
    ) -> torch.Tensor:
        # The HF Llama backbone returns BaseModelOutput with .last_hidden_state.
        # If you wire BioNeMo Megatron here, adapt to its forward signature.
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )
        return out.last_hidden_state  # (B, L, H)

    # --- Loaders ---

    @classmethod
    def from_hf_with_temporal(
        cls,
        hf_repo: str,
        temporal_state_dict_path: str | Path,
        variant: Variant = "217m",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "MaxTokiTemporal":
        from transformers import AutoModel  # type: ignore

        cfg = MaxTokiConfig.for_variant(variant)
        backbone = AutoModel.from_pretrained(hf_repo, torch_dtype=dtype)
        head = TemporalHead(cfg.hidden_size, cfg.temporal_out_dim, cfg.temporal_pooling)
        sd = torch.load(temporal_state_dict_path, map_location="cpu")
        head.load_state_dict(sd)
        backbone = backbone.to(device).eval()
        head = head.to(device=device, dtype=dtype).eval()
        return cls(backbone, head, cfg)

    @classmethod
    def from_distcp(
        cls,
        distcp_dir: str | Path,
        variant: Variant = "217m",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "MaxTokiTemporal":
        """Load the full BioNeMo distcp (backbone + temporal head).

        TODO(needs upstream maxToki repo):
            * import the BioNeMo MegatronBaseModel subclass MaxToki uses
            * build the model from the recipe yaml (config_dir / config_name)
            * call ``dist_checkpointing.load(...)`` from ``megatron.core``
              with sharded_state_dict from the Megatron model
            * extract / wrap the ``temporal_head`` module exactly as the
              upstream code defines it (so MSE values match the paper)
            * detach the LM head (we don't need it for temporal MSE)

        Until that's wired, this stub raises so we don't silently produce
        wrong numbers.
        """
        raise NotImplementedError(
            "from_distcp requires the upstream NVIDIA-Digital-Bio/maxToki repo "
            "to be importable - share it and I'll fill this in. In the meantime, "
            "use from_hf_with_temporal once you've extracted the temporal head."
        )
