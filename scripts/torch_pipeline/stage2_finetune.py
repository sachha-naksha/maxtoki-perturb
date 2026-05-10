"""Stage 2 finetune — peak-aware sparsity attention bias on a frozen MaxToki 217M backbone.

Consumes the prep bundle produced by FIREFate/moscot/maxtoki_multimodal_prep.ipynb
(default: ``$BUNDLE_DIR=/projects/bhdw/asachan/tmp/maxtoki_finetune_skm``):

    cells.parquet     one row / cell with input_ids, gene_positions, keep_mask, age, sample
    genes.parquet     gene_idx -> ensg + symbol  (used to remap stripped-dict ids to the full BioNeMo dict)
    manifest.json     attn-sparsity recipe + paths

The bundle was tokenized with the stripped maxtoki_mlx token_dictionary.json (vocab=20277);
the BioNeMo distcp checkpoint uses the full token_dictionary.json (vocab=23277, contains
<boq>/<eoq>/numeric tokens). 20275 of 20277 shared tokens have different integer ids,
so we rebuild every cell's input_ids in full-dict space at trainer load time using the
ensg<->id mapping from genes.parquet.

Per-cell key-mask -> additive (B,1,1,Lk) attention bias plumbed via
``extra_block_kwargs={"attention_bias": key_bias}`` -> Megatron TransformerBlock ->
TEDotProductAttention ``core_attention_bias`` (post-scale, additive, composes with
the implicit causal mask). 0 at attendable positions, -1e4 (bf16-safe) at masked-out.

The trainer mirrors the 0-shot eval trajectory shape (see e.g.
``configs/pdk4_evenly_seq8k.yaml``):
    K=3 evenly-spaced YM2 (34yo) cells sorted by Pseudotime as context;
    every OM6/OM9 (80yo) cell as a query;
    time_lapse_target = Pseudotime[query] - Pseudotime[last_context].
Same shape used at training and inference, so the in-silico inhibition delta-t
is read from a head that was actually trained on this trajectory format.

Two loss heads (composable):
    Path A — supervise labels[numeric_position] via the LM head's CE on numeric
             tokens (categorical, 3000 bins). Cheap diagnostic that the numeric
             position carries gradient. Off by default (--w-num 0.0).
    Path B — TimeBetweenHead(LayerNorm -> Linear -> GELU -> Dropout -> Linear)
             on hidden state at <eoq>, regressed against z-scored time_lapse via
             MSE. Continuous prediction surface; varies per query cell. THIS is
             what the inhibition delta-t reads at inference. On by default.

The pseudotime values come from the source h5ad (the prep bundle does not store
them). Pass --h5ad to override the default path.

Smoke (1 GPU, 5 steps):  apptainer exec --nv ... stage2_finetune.py --smoke
Real run:                stage2_finetune.py --max-steps 1000 --batch-size 2
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Trajectory assembler + sampler + key-mask helpers
# (transcribed from FIREFate/moscot/maxtoki_multimodal_prep.ipynb cells 9-11)
# ---------------------------------------------------------------------------

def sample_trajectory_specs_within_sample_pseudotime(
    samples_str: np.ndarray,
    pseudotime: np.ndarray,
    K: int = 3,
    n_traj: int | None = None,
    seed: int = 0,
) -> list[dict]:
    """Default training distribution: every cell in every sample takes a turn
    as query; context = K cells *evenly-spaced across the full pseudotime
    range of the same sample*, sorted ascending (so ``ctx[-1]`` is the
    highest-pseudotime cell ≈ pt=100).

    Mirrors the user's intent: "the finetuning looks at all the cells (YM2,
    OM6, OM9) and finetunes based on their peak-RNA sparsity attention bias
    for both NextCell and TimeBetween." Same trajectory shape as the 0-shot
    inhibition evals (``configs/pdk4_evenly_seq8k.yaml``: K=3 evenly-spaced
    along the full reference trajectory).

    Why "full range" not "earlier only"? Pseudotime here is integer 1-100
    with ~13 cells per bin per donor. Picking K evenly-spaced from
    *strictly earlier* always lands the last context one bin before the
    query, so ``time_lapse = query.pt - last_ctx.pt`` collapses to ~1 for
    nearly every row (T_STD ≈ 0.016) — no signal to regress against.
    Spanning the full range gives ``time_lapse = query.pt - 100`` ∈ [-99, 0],
    a continuous per-cell-varying target the regression head can actually fit.

    Note: ctx may include cells with ``pseudotime > query.pseudotime``. That
    is OK — the model is not given any explicit ordering between context
    and query beyond the sequence ordering, and the time_lapse is a signed
    pseudotime delta. Same convention as the eval pipeline.
    """
    rng = np.random.default_rng(seed)
    n = len(samples_str)
    candidates = np.arange(n)
    if n_traj is not None and n_traj < n:
        candidates = rng.choice(candidates, size=n_traj, replace=False)

    # Precompute per-sample pseudotime-sorted index arrays + their evenly-
    # spaced K positions. The K context indices are the SAME for every query
    # in the sample (mirrors eval, where the reference triplet is fixed
    # across queries).
    per_sample_ctx: dict[str, list[int]] = {}
    per_sample_last_pt: dict[str, float] = {}
    for s in np.unique(samples_str):
        in_sample = np.where(samples_str == s)[0]
        in_sample = in_sample[~np.isnan(pseudotime[in_sample])]
        if len(in_sample) < K:
            continue
        order = np.argsort(pseudotime[in_sample], kind="stable")
        sorted_idx = in_sample[order]
        positions = np.linspace(0, len(sorted_idx) - 1, K).round().astype(int)
        ctx = sorted_idx[positions].astype(int).tolist()
        per_sample_ctx[s] = ctx
        per_sample_last_pt[s] = float(pseudotime[ctx[-1]])

    specs = []
    for q in candidates:
        s   = samples_str[q]
        ptq = pseudotime[q]
        if np.isnan(ptq) or s not in per_sample_ctx:
            continue
        ctx = per_sample_ctx[s]
        # Skip rows where query *is* one of the context cells (degenerate;
        # the trajectory would have the query duplicated in context).
        if int(q) in ctx:
            continue
        specs.append({
            "query_idx":    int(q),
            "context_idxs": list(ctx),
            "time_lapse":   float(ptq - per_sample_last_pt[s]),
        })
    return specs


def sample_trajectory_specs_pool_evenly(
    ages_str: np.ndarray,
    pseudotime: np.ndarray,
    context_age: str,
    query_age: str,
    K: int = 3,
    n_traj: int | None = None,
    seed: int = 0,
) -> tuple[list[dict], list[int]]:
    """Mirror the 0-shot eval (configs/pdk4_evenly_seq8k.yaml):
        - K context cells: cells with ``age == context_age`` sorted by
          ``pseudotime``, then K evenly-spaced indices picked. Same triplet
          reused across all queries (matches eval; lets the model condition
          on a fixed reference trajectory).
        - Query cells: every cell with ``age == query_age`` (NaN-pseudotime
          dropped).
        - time_lapse = pseudotime[query] - pseudotime[last_context].
          Last context = highest-pseudotime cell in the picked set.

    Returns (specs, context_idxs) so the trainer can log + persist the chosen
    fixed context."""
    rng = np.random.default_rng(seed)

    ctx_pool = np.where(ages_str == context_age)[0]
    ctx_pool = ctx_pool[~np.isnan(pseudotime[ctx_pool])]
    if len(ctx_pool) == 0:
        raise ValueError(f"no cells with age={context_age!r} (and non-NaN pseudotime)")
    order = np.argsort(pseudotime[ctx_pool], kind="stable")
    ctx_sorted = ctx_pool[order]
    if K >= len(ctx_sorted):
        ctx_idxs = ctx_sorted.astype(int).tolist()
    else:
        positions = np.linspace(0, len(ctx_sorted) - 1, K).round().astype(int)
        ctx_idxs = ctx_sorted[positions].astype(int).tolist()
    last_ctx_pt = float(pseudotime[ctx_idxs[-1]])

    query_pool = np.where(ages_str == query_age)[0]
    query_pool = query_pool[~np.isnan(pseudotime[query_pool])]
    if n_traj is not None and n_traj < len(query_pool):
        query_pool = rng.choice(query_pool, size=n_traj, replace=False)

    specs = [
        {
            "query_idx":    int(q),
            "context_idxs": list(ctx_idxs),
            "time_lapse":   float(pseudotime[q] - last_ctx_pt),
        }
        for q in query_pool
    ]
    return specs, ctx_idxs


def sample_trajectory_specs(
    ages_str: np.ndarray,
    samples_str: np.ndarray,
    K: int,
    n_traj: int | None,
    age_to_pseudotime: dict[str, float] | None = None,
    seed: int = 0,
    same_sample_only: bool = False,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    unique_ages = sorted(np.unique(ages_str))
    if age_to_pseudotime is None:
        try:
            age_to_pseudotime = {a: float(a) for a in unique_ages}
        except ValueError:
            age_to_pseudotime = {a: float(i) for i, a in enumerate(unique_ages)}
    pt = np.array([age_to_pseudotime[a] for a in ages_str])

    candidates = np.arange(len(ages_str))
    if n_traj is not None and n_traj < len(candidates):
        candidates = rng.choice(candidates, size=n_traj, replace=False)

    specs = []
    for q in candidates:
        q_pt = pt[q]
        earlier = np.where(pt < q_pt)[0]
        if same_sample_only:
            earlier = earlier[samples_str[earlier] == samples_str[q]]
        if len(earlier) < K:
            continue
        earlier_pt = pt[earlier]
        unique_earlier = np.unique(earlier_pt)
        if len(unique_earlier) >= K:
            picked = rng.choice(unique_earlier, size=K, replace=False)
            ctx = [int(rng.choice(earlier[earlier_pt == a])) for a in picked]
        else:
            ctx = rng.choice(earlier, size=K, replace=False).astype(int).tolist()
        last_ctx_pt = max(pt[c] for c in ctx)
        specs.append({
            "query_idx":    int(q),
            "context_idxs": [int(c) for c in ctx],
            "time_lapse":   float(q_pt - last_ctx_pt),
        })
    return specs


def assemble_trajectory_for_finetune(
    cells_df: pd.DataFrame,
    query_idx: int,
    context_idxs: list[int],
    time_lapse: float,
    bos_id: int,
    eos_id: int,
    boq_id: int,
    eoq_id: int,
    dummy_numeric_id: int,
    seq_length: int,
) -> dict[str, Any]:
    """Stitch K context cells + 1 query cell into a single Stage 2 row.

    `cells_df.input_ids` / `keep_mask` here must be in **full-dict id space**
    (id-remap done at load time). Specials are always attendable."""
    ids, genes_at, keep_at = [], [], []

    for c in context_idxs:
        c_ids = list(cells_df.input_ids.iloc[c])
        c_gp  = list(cells_df.gene_positions.iloc[c])
        c_km  = list(cells_df.keep_mask.iloc[c])
        ids.extend(c_ids); genes_at.extend(c_gp); keep_at.extend(c_km)

    q_ids = list(cells_df.input_ids.iloc[query_idx])
    q_gp  = list(cells_df.gene_positions.iloc[query_idx])
    q_km  = list(cells_df.keep_mask.iloc[query_idx])
    if len(q_ids) >= 2 and q_ids[0] == bos_id and q_ids[-1] == eos_id:
        q_ids, q_gp, q_km = q_ids[1:-1], q_gp[1:-1], q_km[1:-1]

    ids.append(boq_id);            genes_at.append(-1); keep_at.append(True)
    query_start = len(ids)
    ids.extend(q_ids);             genes_at.extend(q_gp); keep_at.extend(q_km)
    query_end = len(ids)
    ids.append(eoq_id);            genes_at.append(-1); keep_at.append(True)
    numeric_position = len(ids)
    ids.append(dummy_numeric_id);  genes_at.append(-1); keep_at.append(True)

    if len(ids) > seq_length:
        excess = len(ids) - seq_length
        cut = 0
        for j, t in enumerate(ids):
            if t == eos_id and j + 1 >= excess:
                cut = j + 1
                break
        if cut == 0:
            cut = excess
        ids = ids[cut:]; genes_at = genes_at[cut:]; keep_at = keep_at[cut:]
        query_start      -= cut
        query_end        -= cut
        numeric_position -= cut

    return {
        "input_ids":         np.asarray(ids, dtype=np.int64),
        "gene_at_position":  np.asarray(genes_at, dtype=np.int64),
        "keep_mask":         np.asarray(keep_at, dtype=bool),
        "query_start":       query_start,
        "query_end":         query_end,
        "numeric_position":  numeric_position,
        "time_lapse_target": float(time_lapse),
    }


# ---------------------------------------------------------------------------
# Bundle loading + id remap (stripped maxtoki_mlx dict -> full BioNeMo dict)
# ---------------------------------------------------------------------------

def load_per_cell_pseudotime(h5ad_path: Path, cell_ids: np.ndarray,
                              col: str = "Pseudotime") -> np.ndarray:
    """Read ``adata.obs[col]`` from the source h5ad and align to cells.parquet
    by ``cell_id``. The bundle does not store pseudotime; the h5ad is the
    canonical source.

    Returns a float64 array of length ``len(cell_ids)`` (NaN for cells the
    h5ad doesn't have)."""
    import anndata as ad
    a = ad.read_h5ad(h5ad_path)
    if col not in a.obs.columns:
        raise KeyError(f"{col!r} not in {h5ad_path} obs columns: {list(a.obs.columns)}")
    pt_by_cell = dict(zip(a.obs_names.astype(str), a.obs[col].astype(float).values))
    pt = np.array([pt_by_cell.get(str(c), np.nan) for c in cell_ids], dtype=np.float64)
    n_missing = int(np.isnan(pt).sum())
    if n_missing:
        print(f"[pseudotime] WARNING: {n_missing}/{len(pt)} cells missing {col!r}; "
              f"trajectories with NaN pseudotime will be dropped")
    return pt


def load_bundle_and_remap(bundle_dir: Path, full_token_dict: dict[str, int]):
    """Load cells.parquet + genes.parquet and remap every input_id from the
    stripped maxtoki_mlx dict (used during prep) to the full BioNeMo dict.

    Specials: stripped <bos>=2, <eos>=3, <pad>=0; full <bos>=23276, <eos>=2,
    <pad>=0. We map by symbolic name, not by integer.

    The `gene_positions` field tells us which positions are gene tokens; for
    those we look up ensg in genes.parquet and re-encode with full_token_dict[ensg].
    """
    cells = pq.read_table(bundle_dir / "cells.parquet").to_pandas()
    genes = pq.read_table(bundle_dir / "genes.parquet").to_pandas()
    # gene_idx -> ensg lookup (the prep wrote genes.parquet sorted by gene_idx)
    ensg_by_gene_idx: dict[int, str] = dict(zip(genes.gene_idx.astype(int),
                                                 genes.ensg.astype(str)))

    # Stripped specials we know from the prep manifest.
    STRIPPED_BOS, STRIPPED_EOS = 2, 3
    full_bos = int(full_token_dict["<bos>"])
    full_eos = int(full_token_dict["<eos>"])

    n_unmapped = 0
    new_input_ids = []
    for _, row in cells.iterrows():
        ids = list(row.input_ids)
        gpos = list(row.gene_positions)
        out = []
        for tid, gp in zip(ids, gpos):
            if gp >= 0:
                ensg = ensg_by_gene_idx.get(int(gp))
                full_id = full_token_dict.get(ensg) if ensg else None
                if full_id is None:
                    n_unmapped += 1
                    out.append(0)  # pad — should not occur if genes.parquet matches
                else:
                    out.append(int(full_id))
            else:
                # special token (was <bos>/<eos> in stripped dict)
                if tid == STRIPPED_BOS:
                    out.append(full_bos)
                elif tid == STRIPPED_EOS:
                    out.append(full_eos)
                else:
                    n_unmapped += 1
                    out.append(0)
        new_input_ids.append(np.asarray(out, dtype=np.int64))
    cells["input_ids"] = new_input_ids
    if n_unmapped:
        print(f"[load_bundle_and_remap] WARNING: {n_unmapped} tokens unmapped (set to <pad>=0)")
    return cells, genes


# ---------------------------------------------------------------------------
# Torch Dataset
# ---------------------------------------------------------------------------

def time_to_numeric_token_id(time_lapse: float, numeric_id_to_value: dict[int, int]) -> int:
    """Pick the numeric token whose integer value is closest to ``time_lapse``.

    For SKM (ages {34, 80}) the targets are integers in years; the BioNeMo
    numeric token grid spans -1500..1499 so every reachable lapse has an exact
    or near-exact bin."""
    return min(numeric_id_to_value.items(),
               key=lambda kv: abs(kv[1] - time_lapse))[0]


class TrajectoryDataset(Dataset):
    def __init__(
        self,
        cells_df: pd.DataFrame,
        specs: list[dict],
        seq_length: int,
        bos_id: int, eos_id: int, boq_id: int, eoq_id: int,
        dummy_numeric_id: int,
    ):
        self.cells = cells_df
        self.specs = specs
        self.seq_length = seq_length
        self.bos_id = bos_id; self.eos_id = eos_id
        self.boq_id = boq_id; self.eoq_id = eoq_id
        self.dummy_numeric_id = dummy_numeric_id

    def __len__(self) -> int:
        return len(self.specs)

    def __getitem__(self, idx: int) -> dict:
        s = self.specs[idx]
        return assemble_trajectory_for_finetune(
            self.cells,
            query_idx        = s["query_idx"],
            context_idxs     = s["context_idxs"],
            time_lapse       = s["time_lapse"],
            bos_id           = self.bos_id,
            eos_id           = self.eos_id,
            boq_id           = self.boq_id,
            eoq_id           = self.eoq_id,
            dummy_numeric_id = self.dummy_numeric_id,
            seq_length       = self.seq_length,
        )


def collate_trajectories(
    batch: list[dict],
    pad_id: int,
    numeric_id_to_value: dict[int, int] | None = None,
) -> dict[str, torch.Tensor]:
    """Pad to the longest row in batch. Pad positions are masked out from
    both the loss (loss_mask=0) and attention (key_bias=-1e4).

    Loss-mask layout (Path A — supervise gene tokens AND numeric token via the
    LM head's shared CE):
        loss_mask[query_start : query_end]   = 1   # NextCell CE
        loss_mask[numeric_position]          = 1   # TimeBetweenCells CE (Path A)

    Megatron's compute_language_model_loss does the standard causal shift
    internally: labels[i] is scored against logits[i-1]. So labels[query_start]
    is predicted at <boq> (position query_start - 1), and labels[numeric_position]
    is predicted at <eoq> (position numeric_position - 1). Both correct.

    We additionally return ``gene_loss_mask`` and ``num_loss_mask`` (subsets of
    ``loss_mask``) so the trainer can log the two CE components separately.
    """
    L = max(len(r["input_ids"]) for r in batch)
    B = len(batch)

    input_ids       = torch.full((B, L), pad_id, dtype=torch.long)
    keep_mask       = torch.zeros((B, L), dtype=torch.bool)
    loss_mask       = torch.zeros((B, L), dtype=torch.float32)
    gene_loss_mask  = torch.zeros((B, L), dtype=torch.float32)
    num_loss_mask   = torch.zeros((B, L), dtype=torch.float32)
    labels          = torch.full((B, L), pad_id, dtype=torch.long)
    seq_lens        = torch.zeros(B, dtype=torch.long)
    numeric_pos     = torch.zeros(B, dtype=torch.long)
    time_lapse      = torch.zeros(B, dtype=torch.float32)
    numeric_label   = torch.zeros(B, dtype=torch.long)

    for i, r in enumerate(batch):
        ell = len(r["input_ids"])
        seq_lens[i] = ell
        input_ids[i, :ell] = torch.from_numpy(r["input_ids"])
        keep_mask[i, :ell] = torch.from_numpy(r["keep_mask"])
        # default labels = input_ids (Megatron handles the causal shift)
        labels[i, :ell] = torch.from_numpy(r["input_ids"])

        qs, qe = r["query_start"], r["query_end"]
        np_pos = r["numeric_position"]
        gene_loss_mask[i, qs:qe] = 1.0
        loss_mask[i, qs:qe]      = 1.0

        if numeric_id_to_value is not None:
            num_id = time_to_numeric_token_id(r["time_lapse_target"], numeric_id_to_value)
            labels[i, np_pos]     = num_id
            num_loss_mask[i, np_pos] = 1.0
            loss_mask[i, np_pos]     = 1.0
            numeric_label[i] = num_id

        numeric_pos[i] = np_pos
        time_lapse[i]  = r["time_lapse_target"]

    position_ids = torch.arange(L).unsqueeze(0).expand(B, -1).contiguous()
    return {
        "input_ids":      input_ids,
        "position_ids":   position_ids,
        "labels":         labels,
        "loss_mask":      loss_mask,
        "gene_loss_mask": gene_loss_mask,
        "num_loss_mask":  num_loss_mask,
        "keep_mask":      keep_mask,
        "seq_lens":       seq_lens,
        "numeric_pos":    numeric_pos,
        "numeric_label":  numeric_label,
        "time_lapse":     time_lapse,
    }


# ---------------------------------------------------------------------------
# Backbone setup
# ---------------------------------------------------------------------------

def init_single_rank_dist(seed: int = 0):
    """Megatron's distcp restore needs torch.distributed + parallel_state init,
    even on single GPU. We bring up a single-rank NCCL group on localhost and
    register the model-parallel-rng tracker (otherwise VocabParallelEmbedding
    init blows up with 'cuda rng state model-parallel-rng is not added')."""
    import torch.distributed as dist
    from megatron.core import parallel_state, tensor_parallel

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29501")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        dist.init_process_group(backend="nccl", world_size=1, rank=0)
    torch.cuda.set_device(0)
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )
    tensor_parallel.model_parallel_cuda_manual_seed(seed)


def _patch_sdpa_attention_for_bias() -> None:
    """MaxToki 217M (head_dim=154) uses ``SDPADotProductAttention`` because
    TE Flash requires head_dim%8==0. Upstream's SDPA forward asserts
    ``attention_bias is None``, which kills our key-mask plumbing. Patch the
    forward to compose ``attention_bias`` with the implicit causal mask via
    SDPA's ``attn_mask`` arg (and turn ``is_causal=False`` so SDPA respects it).

    With an explicit attn_mask SDPA may drop to the memory-efficient backend
    (slower than Flash). Acceptable for finetune; the math result is identical.

    Idempotent."""
    from bionemo.maxtoki import sdpa_attention as _sa
    from megatron.core.transformer.enums import AttnMaskType

    if getattr(_sa.SDPADotProductAttention, "_attention_bias_patched", False):
        return

    def forward(self, query, key, value, attention_mask,
                attn_mask_type=None, attention_bias=None, packed_seq_params=None):
        assert packed_seq_params is None, (
            "Packed sequence is not supported by SDPADotProductAttention.")

        # GQA expand
        ratio = (self.num_attention_heads_per_partition
                 // self.num_query_groups_per_partition)
        if ratio > 1:
            key   = key.repeat_interleave(ratio, dim=2)
            value = value.repeat_interleave(ratio, dim=2)

        # [sq, b, np, hn] -> [b, np, sq, hn]
        q = query.permute(1, 2, 0, 3)
        k = key.permute(1, 2, 0, 3)
        v = value.permute(1, 2, 0, 3)

        eff_type    = attn_mask_type if attn_mask_type is not None else self.attn_mask_type
        is_causal   = eff_type == AttnMaskType.causal
        dropout_p   = self.config.attention_dropout if self.training else 0.0

        if attention_bias is None:
            context = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=dropout_p,
                is_causal=is_causal, scale=self.softmax_scale,
            )
        else:
            B, Hq, Sq, _ = q.shape
            Sk = k.shape[2]
            mask = attention_bias.to(q.dtype)        # (B,1,1,Sk) or broadcastable
            if is_causal:
                offset = Sk - Sq                      # >=0; usually 0
                causal = torch.full((Sq, Sk), float("-inf"),
                                    device=q.device, dtype=q.dtype)
                causal = torch.triu(causal, diagonal=1 + offset)
                mask = mask + causal.unsqueeze(0).unsqueeze(0)  # broadcast (1,1,Sq,Sk)
            context = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=dropout_p,
                is_causal=False, scale=self.softmax_scale,
            )

        # [b, np, sq, hn] -> [sq, b, np, hn] -> [sq, b, hp]
        context = context.permute(2, 0, 1, 3).contiguous()
        new_shape = context.size()[:-2] + (self.hidden_size_per_partition,)
        return context.view(*new_shape)

    _sa.SDPADotProductAttention.forward = forward
    _sa.SDPADotProductAttention._attention_bias_patched = True
    print("[stage2] patched SDPADotProductAttention.forward to accept attention_bias")


def build_model(ckpt_dir: Path, tokenizer_path: Path, seq_length: int):
    from bionemo.maxtoki.tokenizer import MaxTokiTokenizer
    from bionemo.maxtoki.model import MaxTokiMultitaskFineTuneConfig

    _patch_sdpa_attention_for_bias()

    full_dict = json.loads(Path(tokenizer_path).read_text())
    tok = MaxTokiTokenizer(token_dictionary={k: int(v) for k, v in full_dict.items()})

    cfg = MaxTokiMultitaskFineTuneConfig(
        initial_ckpt_path=str(ckpt_dir),
        seq_length=seq_length,
        # Megatron's default backward path expects each ColumnParallelLinear to
        # accumulate into ``weight.main_grad`` (the distributed-optimizer's fp32
        # buffer). We're running a vanilla AdamW outside Megatron's distopt, so
        # ``main_grad`` is never allocated -> backward NPEs at layers.py:561.
        # Disabling fusion routes grads through the standard ``.grad`` path.
        gradient_accumulation_fusion=False,
    )
    model = cfg.configure_model(tokenizer=tok)
    model = model.cuda().to(torch.bfloat16)
    return model, tok, full_dict


def freeze_backbone_keep_lm_head(model: torch.nn.Module) -> int:
    """Freeze every param. Then unfreeze only the LM output projection (the
    minimal trainable surface that lets CE backprop reach an actual update).

    Megatron's ``ColumnParallelLinear`` backward writes weight-grads into
    ``weight.main_grad`` (the distributed-optimizer's fp32 accumulator) when
    ``ctx.gradient_accumulation_fusion`` is True. The setting is restored from
    the BioNeMo checkpoint regardless of what we put on the config, so we
    pre-allocate ``main_grad`` on every trainable param. After backward we
    copy ``main_grad`` into ``.grad`` so the vanilla AdamW can consume it
    (see ``flush_main_grad_to_grad`` below).

    Returns the number of trainable params."""
    for p in model.parameters():
        p.requires_grad_(False)

    n_trainable = 0
    for name, p in model.named_parameters():
        # MCore GPT LM output is `output_layer.*` (ColumnParallelLinear)
        if name.startswith("output_layer"):
            p.requires_grad_(True)
            p.main_grad = torch.zeros_like(p, dtype=torch.float32)
            n_trainable += p.numel()
    return n_trainable


def flush_main_grad_to_grad(model: torch.nn.Module) -> None:
    """Move accumulated grads from Megatron's ``main_grad`` accumulator into
    the standard ``.grad`` field, then zero ``main_grad`` for the next step."""
    for p in model.parameters():
        if p.requires_grad and getattr(p, "main_grad", None) is not None:
            p.grad = p.main_grad.to(p.dtype)
            p.main_grad.zero_()


def save_heads_checkpoint(out_path: Path, model, time_head, T_MEAN: float, T_STD: float):
    """Snapshot the trainable surfaces (LM head + TimeBetweenHead) plus
    z-score stats so the inference adapter can de-normalize predictions."""
    state = {
        "T_MEAN": T_MEAN,
        "T_STD":  T_STD,
        "lm_head_state_dict": {
            n: p.detach().cpu()
            for n, p in model.named_parameters() if n.startswith("output_layer")
        },
    }
    if time_head is not None:
        state["time_head_state_dict"] = {
            n: p.detach().cpu() for n, p in time_head.state_dict().items()
        }
    torch.save(state, out_path)


# ---------------------------------------------------------------------------
# Path B — TimeBetweenHead + decoder hidden-state capture
# ---------------------------------------------------------------------------

class TimeBetweenHead(nn.Module):
    """Continuous regression head consuming TWO signals from the last block:
        (a) hidden state at <eoq>           — the canonical "predict next" position
        (b) mean-pool of query gene-token  — direct per-cell content signal
                                             (without this, the head only sees
                                             whatever the <eoq> aggregator
                                             encoded, which from a frozen
                                             backbone may be impoverished)
    Concatenated input is (2H,). Output is *z-scored* time-lapse;
    de-normalize with ``T_MEAN, T_STD`` for inference.

    Architecture: 2 LayerNorm+Linear+GELU blocks with skip + a final scalar
    projection. ~5x the params of the bioRxiv minimal head, but still tiny
    relative to the 217M backbone."""
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.in_norm = nn.LayerNorm(2 * hidden_dim)
        self.fc1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, h_eoq: torch.Tensor, h_query_pool: torch.Tensor) -> torch.Tensor:
        # h_eoq, h_query_pool: (B, H)  ->  concat (B, 2H)  ->  scalar (B,)
        x = torch.cat([h_eoq, h_query_pool], dim=-1)
        x = self.in_norm(x)
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = x + F.gelu(self.fc2(self.norm2(x)))   # residual
        x = self.dropout(x)
        return self.out(x).squeeze(-1)


@torch.no_grad()
def run_validation(
    model, time_head, decoder_capture,
    val_specs: list[dict], cells_df: pd.DataFrame,
    bos_id: int, eos_id: int, boq_id: int, eoq_id: int,
    pad_id: int, dummy_numeric_id: int,
    seq_length: int, batch_size: int, max_rows: int, no_attn_bias: bool,
    T_MEAN: float, T_STD: float,
    numeric_id_to_value: dict[int, int] | None,
) -> dict:
    """Forward pass on the val set; report MSE, Pearson r, and de-z-scored
    prediction stats. Cheap because no backward + capped to ``max_rows``."""
    if not val_specs:
        return {}
    specs = val_specs[:max_rows]
    ds = TrajectoryDataset(
        cells_df, specs, seq_length=seq_length,
        bos_id=bos_id, eos_id=eos_id, boq_id=boq_id, eoq_id=eoq_id,
        dummy_numeric_id=dummy_numeric_id,
    )
    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False,
        collate_fn=lambda b: collate_trajectories(
            b, pad_id=pad_id, numeric_id_to_value=numeric_id_to_value,
        ),
    )
    was_training = model.training
    model.eval()
    if time_head is not None:
        time_head.eval()

    all_pred, all_targ = [], []
    for batch in dl:
        ids       = batch["input_ids"].cuda(non_blocking=True)
        pos       = batch["position_ids"].cuda(non_blocking=True)
        gene_mask = batch["gene_loss_mask"].cuda(non_blocking=True)
        keep      = batch["keep_mask"].cuda(non_blocking=True)
        num_pos   = batch["numeric_pos"].cuda(non_blocking=True)
        time_lapse = batch["time_lapse"]

        if no_attn_bias:
            extra = None
        else:
            B, L = ids.shape
            key_bias = torch.zeros(B, 1, 1, L, dtype=torch.bfloat16, device=ids.device)
            key_bias[~keep[:, None, None, :]] = -1e4
            extra = {"attention_bias": key_bias}

        labels = ids
        loss_mask = torch.zeros_like(ids, dtype=torch.float32)
        _ = model(input_ids=ids, position_ids=pos, attention_mask=None,
                  labels=labels, loss_mask=loss_mask, extra_block_kwargs=extra)
        h_sbh = decoder_capture["h"]
        h_bsh = h_sbh.transpose(0, 1).contiguous()
        B = h_bsh.size(0)
        eoq_pos = (num_pos - 1).clamp_min(0)
        h_eoq = h_bsh[torch.arange(B, device=h_bsh.device), eoq_pos]
        qmask = gene_mask.to(h_bsh.dtype)
        denom = qmask.sum(dim=1, keepdim=True).clamp_min(1.0)
        h_qpool = (h_bsh * qmask.unsqueeze(-1)).sum(dim=1) / denom
        t_pred_z = time_head(h_eoq, h_qpool).float()
        t_pred_real = (t_pred_z * T_STD + T_MEAN).cpu().numpy()
        all_pred.append(t_pred_real)
        all_targ.append(time_lapse.numpy())
    if was_training:
        model.train()
        if time_head is not None:
            time_head.train()
    pred = np.concatenate(all_pred)
    targ = np.concatenate(all_targ)
    targ_z = (targ - T_MEAN) / T_STD
    pred_z = (pred - T_MEAN) / T_STD
    mse = float(np.mean((pred_z - targ_z) ** 2))
    r = float(np.corrcoef(pred, targ)[0, 1]) if pred.std() > 0 else float("nan")
    return {
        "n":            len(pred),
        "mse_z":        mse,
        "pearson_r":    r,
        "pred_mean":    float(pred.mean()),
        "pred_std":     float(pred.std()),
        "targ_mean":    float(targ.mean()),
        "targ_std":     float(targ.std()),
    }


def attach_decoder_hidden_capture(model: torch.nn.Module) -> tuple[dict, Any]:
    """Forward-hook ``model.decoder`` (the Megatron TransformerBlock) and stash
    its output hidden states in a captured dict. Hidden layout from MCore is
    ``[seq, batch, hidden]``. Returns (captured_dict, handle)."""
    captured: dict[str, torch.Tensor] = {}

    def hook(module, inp, out):
        h = out[0] if isinstance(out, (tuple, list)) else out
        captured["h"] = h

    handle = model.decoder.register_forward_hook(hook)
    return captured, handle


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-dir", type=Path,
                    default=Path("/projects/bhdw/asachan/tmp/maxtoki_finetune_skm"))
    ap.add_argument("--h5ad", type=Path,
                    default=Path("/projects/bhdw/asachan/tmp/atac_rna_pairing_skm_prep/rna_sub.h5ad"),
                    help="Source RNA h5ad — only used to pull per-cell Pseudotime "
                         "(the bundle does not store it).")
    ap.add_argument("--pseudotime-col", default="Pseudotime",
                    help="adata.obs column for the per-cell pseudotime metric.")
    ap.add_argument("--ckpt-dir", type=Path,
                    default=Path("/projects/bhdw/asachan/models/MaxToki/MaxToki-217M-bionemo"))
    ap.add_argument("--tokenizer-path", type=Path,
                    default=Path("/projects/bhdw/asachan/models/MaxToki/MaxToki-217M-bionemo/context/token_dictionary.json"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/projects/bhdw/asachan/methods/maxtoki-perturb/out/finetune_skm_smoke"))
    ap.add_argument("--seq-length", type=int, default=8192)
    ap.add_argument("--k-context", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--n-traj", type=int, default=None,
                    help="Cap the number of sampled trajectories (None = use every cell that qualifies)")
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="Default LR (used for LM head if --lr-lm-head not given).")
    ap.add_argument("--lr-lm-head",   type=float, default=None,
                    help="LR for the LM head (output_layer.*). Defaults to --lr. "
                         "Pretrained — keep low (1e-5 to 1e-4).")
    ap.add_argument("--lr-time-head", type=float, default=1e-3,
                    help="LR for the TimeBetweenHead. Randomly init'd from scratch "
                         "— needs ~10-100x the LM head LR.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true",
                    help="Smoke: 5 steps, batch=1, n-traj=64, seq-length=4096.")
    ap.add_argument("--no-attn-bias", action="store_true",
                    help="Disable the keep_mask attention bias (ablation).")

    # ---- multitask loss weights (defaults: tilt the gradient bus toward
    # time prediction; gene CE is a regularizer, not the main objective).
    ap.add_argument("--w-ce",  type=float, default=0.1,
                    help="NextCell gene-token CE weight. 0 = freeze LM head "
                         "entirely; the time prediction won't depend on it.")
    ap.add_argument("--w-num", type=float, default=1.0,
                    help="Path A numeric-token CE weight (categorical readout via "
                         "the LM head). Routes time signal through the well-trained "
                         "LM head's output projection — cheap and effective.")
    ap.add_argument("--w-mse", type=float, default=1.0,
                    help="Path B TimeBetweenHead MSE weight (continuous readout via "
                         "regression head on hidden state at <eoq>).")
    ap.add_argument("--no-time-head", action="store_true",
                    help="Skip building the regression head entirely.")

    # ---- validation
    ap.add_argument("--val-frac", type=float, default=0.15,
                    help="Fraction of trajectory specs held out for validation. "
                         "0 = no validation.")
    ap.add_argument("--val-every", type=int, default=100,
                    help="Run validation forward pass every N training steps.")
    ap.add_argument("--val-max-rows", type=int, default=128,
                    help="Cap validation cells per pass (keeps val cheap).")

    ap.add_argument("--save-every", type=int, default=0,
                    help="Periodically save stage2_heads_step{N}.pt every N steps. "
                         "0 = only save at end.")
    args = ap.parse_args()
    if args.lr_lm_head is None:
        args.lr_lm_head = args.lr

    if args.smoke:
        args.max_steps = 5
        args.batch_size = 1
        args.n_traj = 64
        args.seq_length = 4096

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- backbone first (it inits torch.distributed via env)
    init_single_rank_dist(seed=args.seed)
    print(f"[stage2] loading backbone from {args.ckpt_dir}")
    model, tok, full_dict = build_model(args.ckpt_dir, args.tokenizer_path, args.seq_length)
    n_trainable = freeze_backbone_keep_lm_head(model)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[stage2] params total={n_total/1e6:.1f}M  trainable={n_trainable/1e6:.2f}M "
          f"({100*n_trainable/n_total:.2f}%)")

    pad_id  = int(full_dict["<pad>"])
    bos_id  = int(full_dict["<bos>"])
    eos_id  = int(full_dict["<eos>"])
    boq_id  = int(full_dict["<boq>"])
    eoq_id  = int(full_dict["<eoq>"])
    # dummy numeric token: pick the one mapping to value 0
    nums = {int(v): int(full_dict[v]) for v in full_dict if isinstance(v, str) and v.lstrip("-").isdigit()}
    if 0 in nums:
        dummy_numeric_id = nums[0]
    else:
        dummy_numeric_id = next(iter(nums.values()))
    print(f"[stage2] specials  bos={bos_id} eos={eos_id} boq={boq_id} eoq={eoq_id} "
          f"pad={pad_id} dummy_numeric={dummy_numeric_id}")
    # Path A target table: numeric_token_id -> integer time-lapse value (years).
    # nums above already maps value -> id; invert.
    numeric_id_to_value = {tid: val for val, tid in nums.items()}
    n_num_bins = len(numeric_id_to_value)
    val_min = min(numeric_id_to_value.values())
    val_max = max(numeric_id_to_value.values())
    print(f"[stage2] numeric vocab: {n_num_bins} bins, value range [{val_min}, {val_max}]; "
          f"random-CE baseline ~{np.log(n_num_bins):.2f}")

    # ---- bundle (with id remap into full-dict space)
    print(f"[stage2] loading bundle {args.bundle_dir}")
    cells_df, genes_df = load_bundle_and_remap(args.bundle_dir, full_dict)
    print(f"[stage2] cells={len(cells_df):,}  genes={len(genes_df):,}")

    # ---- per-cell pseudotime (from source h5ad; the bundle doesn't carry it)
    samples_str = cells_df["sample"].astype(str).values
    cell_ids    = cells_df["cell_id"].astype(str).values
    pseudotime  = load_per_cell_pseudotime(args.h5ad, cell_ids, col=args.pseudotime_col)
    print(f"[stage2] pseudotime ({args.pseudotime_col}): "
          f"min={np.nanmin(pseudotime):.2f} max={np.nanmax(pseudotime):.2f} "
          f"valid={int((~np.isnan(pseudotime)).sum())}/{len(pseudotime)}")

    # ---- trajectory sampling: every cell as query, K earlier-pseudotime cells
    # from the same sample as context. Mirrors the user's intent: "the
    # finetuning looks at all the cells (YM2, OM6, OM9) and finetunes based
    # on their peak-RNA sparsity attention bias for both NextCell and
    # TimeBetween."
    specs = sample_trajectory_specs_within_sample_pseudotime(
        samples_str, pseudotime, K=args.k_context, n_traj=args.n_traj, seed=args.seed,
    )
    print(f"[stage2] sampled {len(specs)} trajectory specs "
          f"(within-sample, K={args.k_context} earlier-pseudotime cells)")
    if not specs:
        raise RuntimeError("no trajectories sampled — check --k-context vs sample sizes")
    by_sample: dict[str, int] = {}
    for s in specs:
        by_sample[samples_str[s["query_idx"]]] = by_sample.get(samples_str[s["query_idx"]], 0) + 1
    print(f"[stage2] per-sample trajectory counts: {by_sample}")

    # ---- train / val split (z-score stats computed on TRAIN only so val is honest)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(specs))
    n_val = int(args.val_frac * len(specs)) if args.val_frac > 0 else 0
    val_specs   = [specs[i] for i in perm[:n_val]]
    train_specs = [specs[i] for i in perm[n_val:]]
    t_arr = np.array([s["time_lapse"] for s in train_specs], dtype=np.float64)
    T_MEAN = float(t_arr.mean())
    T_STD  = float(t_arr.std()) or 1.0
    print(f"[stage2] split: train={len(train_specs)}  val={len(val_specs)}")
    print(f"[stage2] time_lapse target (train): mean={T_MEAN:.3f} std={T_STD:.3f} "
          f"min={t_arr.min():.2f} max={t_arr.max():.2f}")
    if val_specs:
        v_arr = np.array([s["time_lapse"] for s in val_specs], dtype=np.float64)
        print(f"[stage2] time_lapse target (val):   mean={v_arr.mean():.3f} "
              f"std={v_arr.std():.3f}")
    specs = train_specs   # the dataloader iterates train-only

    ds = TrajectoryDataset(
        cells_df, specs, seq_length=args.seq_length,
        bos_id=bos_id, eos_id=eos_id, boq_id=boq_id, eoq_id=eoq_id,
        dummy_numeric_id=dummy_numeric_id,
    )
    # Path A only ships supervised numeric labels if w_num > 0. Otherwise we
    # leave the numeric position out of the LM CE entirely.
    path_a_id_to_value = numeric_id_to_value if args.w_num > 0 else None
    dl = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, drop_last=True,
        collate_fn=lambda b: collate_trajectories(
            b, pad_id=pad_id, numeric_id_to_value=path_a_id_to_value,
        ),
    )

    # ---- Path B regression head (optional)
    time_head = None
    decoder_capture = None
    decoder_hook_handle = None
    if not args.no_time_head:
        hidden_dim = int(model.config.hidden_size)
        time_head = TimeBetweenHead(hidden_dim=hidden_dim, dropout=0.1)
        time_head = time_head.cuda().to(torch.bfloat16)
        decoder_capture, decoder_hook_handle = attach_decoder_hidden_capture(model)
        print(f"[stage2] TimeBetweenHead built (hidden_dim={hidden_dim}, "
              f"params={sum(p.numel() for p in time_head.parameters())/1e6:.2f}M)")

    # Optimizer with per-group LR: the LM head is pretrained (low LR),
    # the time head is randomly init'd (much higher LR). When --w-ce=0 the
    # LM head won't get gradient anyway; we still register it (cheap) so the
    # final checkpoint doesn't lose it.
    param_groups = []
    lm_params = [p for p in model.parameters() if p.requires_grad]
    if lm_params:
        param_groups.append({"params": lm_params, "lr": args.lr_lm_head, "name": "lm_head"})
    if time_head is not None:
        param_groups.append({"params": list(time_head.parameters()),
                             "lr": args.lr_time_head, "name": "time_head"})
    optim = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), weight_decay=0.0)
    print(f"[stage2] optimizer LRs: " +
          ", ".join(f"{g['name']}={g['lr']}" for g in param_groups))

    # ---- training loop
    model.train()
    if time_head is not None:
        time_head.train()
    step = 0
    history = []   # list of dicts (per-step loss components)
    for batch in dl:
        if step >= args.max_steps:
            break
        ids        = batch["input_ids"].cuda(non_blocking=True)
        pos        = batch["position_ids"].cuda(non_blocking=True)
        lab        = batch["labels"].cuda(non_blocking=True)
        lmask      = batch["loss_mask"].cuda(non_blocking=True)
        gene_mask  = batch["gene_loss_mask"].cuda(non_blocking=True)
        num_mask   = batch["num_loss_mask"].cuda(non_blocking=True)
        keep       = batch["keep_mask"].cuda(non_blocking=True)
        num_pos    = batch["numeric_pos"].cuda(non_blocking=True)
        time_lapse = batch["time_lapse"].cuda(non_blocking=True)

        # additive (B,1,1,Lk) attention bias: 0 attendable, -1e4 masked-out
        if args.no_attn_bias:
            extra = None
        else:
            B, L = ids.shape
            key_bias = torch.zeros(B, 1, 1, L, dtype=torch.bfloat16, device=ids.device)
            key_bias[~keep[:, None, None, :]] = -1e4
            extra = {"attention_bias": key_bias}

        out = model(
            input_ids=ids,
            position_ids=pos,
            attention_mask=None,                  # implicit causal via layer spec
            labels=lab,
            loss_mask=lmask,
            extra_block_kwargs=extra,
        )
        # `lm_outputs` is per-token CE (already loss-masked inside MCoreGPT).
        ce_pertok = out["lm_outputs"]

        # Per-component CE for diagnostics + loss assembly.
        gene_denom = gene_mask.sum().clamp_min(1.0)
        num_denom  = num_mask.sum().clamp_min(1.0)
        ce_gene = (ce_pertok * gene_mask).sum() / gene_denom
        ce_num  = (ce_pertok * num_mask).sum()  / num_denom

        # Path B MSE: regression head sees TWO signals from the last block:
        #  - hidden at <eoq> (= numeric_position - 1; under causal-LM shifting,
        #    this is the position whose logit predicts the numeric token)
        #  - mean-pool of query gene-token hidden states (per-cell content,
        #    bypasses the <eoq> bottleneck)
        mse_loss = torch.tensor(0.0, device=ids.device)
        t_pred_z = None
        if time_head is not None and decoder_capture is not None:
            h_sbh = decoder_capture["h"]                  # (S, B, H), bf16
            h_bsh = h_sbh.transpose(0, 1).contiguous()    # (B, S, H)
            B = h_bsh.size(0)
            eoq_pos = (num_pos - 1).clamp_min(0)
            h_eoq = h_bsh[torch.arange(B, device=h_bsh.device), eoq_pos]  # (B, H)
            # Mean-pool over query gene tokens. gene_loss_mask is 1 exactly on
            # [query_start:query_end], so we get a clean per-row pooling weight.
            qmask = gene_loss_mask.to(h_bsh.dtype)        # (B, S)
            denom = qmask.sum(dim=1, keepdim=True).clamp_min(1.0)
            h_qpool = (h_bsh * qmask.unsqueeze(-1)).sum(dim=1) / denom  # (B, H)
            t_pred_z = time_head(h_eoq, h_qpool).float()  # (B,) z-scored
            t_target_z = ((time_lapse - T_MEAN) / T_STD).float()
            mse_loss = F.mse_loss(t_pred_z, t_target_z)

        loss = args.w_ce * ce_gene + args.w_num * ce_num + args.w_mse * mse_loss

        loss.backward()
        flush_main_grad_to_grad(model)

        # gnorm sanity -- compute *after* the main_grad flush but *before* zero_grad.
        gnorm_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                gnorm_sq += float(p.grad.detach().float().pow(2).sum())
        head_gnorm_sq = 0.0
        if time_head is not None:
            for p in time_head.parameters():
                if p.grad is not None:
                    head_gnorm_sq += float(p.grad.detach().float().pow(2).sum())
        gnorm = (gnorm_sq + head_gnorm_sq) ** 0.5

        optim.step()
        optim.zero_grad(set_to_none=True)

        kept_frac = float(keep.float().mean())
        # Per-cell variance of t_pred -- KEY signal that the head is reading
        # the query cell, not collapsing to a constant.
        if t_pred_z is not None and t_pred_z.numel() > 1:
            t_pred_std_z   = float(t_pred_z.detach().float().std())
            t_target_std_z = float(((time_lapse - T_MEAN) / T_STD).std())
            t_pred_real_mean = float((t_pred_z.detach().float() * T_STD + T_MEAN).mean())
        else:
            t_pred_std_z = t_target_std_z = t_pred_real_mean = float("nan")

        history.append({
            "step":             step,
            "loss":             float(loss),
            "ce_gene":          float(ce_gene),
            "ce_num":           float(ce_num),
            "mse":              float(mse_loss),
            "t_pred_std_z":     t_pred_std_z,
            "t_target_std_z":   t_target_std_z,
            "t_pred_real_mean": t_pred_real_mean,
            "L":                int(ids.shape[1]),
            "keep_frac":        kept_frac,
            "gnorm":            gnorm,
        })
        print(f"[stage2] step {step:>4d}  loss={float(loss):.3f}  "
              f"ce_gene={float(ce_gene):.3f}  mse={float(mse_loss):.3f}  "
              f"t_pred_std={t_pred_std_z:.3f}  L={ids.shape[1]:>5d}  "
              f"keep_frac={kept_frac:.2f}  gnorm={gnorm:.2e}")

        # Periodic validation — the only honest signal that the head is
        # generalizing rather than memorizing.
        if (val_specs and args.val_every > 0
                and (step + 1) % args.val_every == 0
                and time_head is not None):
            v = run_validation(
                model, time_head, decoder_capture,
                val_specs, cells_df,
                bos_id, eos_id, boq_id, eoq_id, pad_id, dummy_numeric_id,
                seq_length=args.seq_length, batch_size=args.batch_size,
                max_rows=args.val_max_rows, no_attn_bias=args.no_attn_bias,
                T_MEAN=T_MEAN, T_STD=T_STD,
                numeric_id_to_value=path_a_id_to_value,
            )
            print(f"[stage2] VAL@{step+1}  n={v['n']}  mse_z={v['mse_z']:.3f}  "
                  f"r={v['pearson_r']:.3f}  pred {v['pred_mean']:.2f}±{v['pred_std']:.2f}  "
                  f"targ {v['targ_mean']:.2f}±{v['targ_std']:.2f}")
            history[-1]["val"] = {**v, "step": step + 1}

        # Periodic checkpoint — cheap insurance for long runs.
        if args.save_every > 0 and (step + 1) % args.save_every == 0:
            ckpt_path = args.out_dir / f"stage2_heads_step{step+1}.pt"
            save_heads_checkpoint(ckpt_path, model, time_head, T_MEAN, T_STD)
            (args.out_dir / "summary_partial.json").write_text(json.dumps({
                "max_steps": args.max_steps, "step_done": step + 1,
                "history": history,
            }, indent=2))
            print(f"[stage2] checkpoint -> {ckpt_path.name}  ({step+1}/{args.max_steps})")

        step += 1

    if decoder_hook_handle is not None:
        decoder_hook_handle.remove()

    summary = {
        "max_steps":              args.max_steps,
        "batch_size":             args.batch_size,
        "k_context":              args.k_context,
        "seq_length":             args.seq_length,
        "n_specs":                len(specs),
        "per_sample_traj_count":  by_sample,
        "n_trainable":            n_trainable,
        "no_attn_bias":           args.no_attn_bias,
        "n_numeric_bins":         n_num_bins,
        "ce_num_random_baseline": float(np.log(n_num_bins)),
        "loss_weights": {"w_ce": args.w_ce, "w_num": args.w_num, "w_mse": args.w_mse},
        "time_target": {
            "pseudotime_col": args.pseudotime_col,
            "T_MEAN":         T_MEAN,
            "T_STD":          T_STD,
            "min":            float(t_arr.min()),
            "max":            float(t_arr.max()),
        },
        "history":                history,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[stage2] wrote {args.out_dir / 'summary.json'}")

    # Final checkpoint — what stage2_predict.py loads by default.
    save_heads_checkpoint(args.out_dir / "stage2_heads.pt", model, time_head, T_MEAN, T_STD)
    print(f"[stage2] wrote {args.out_dir / 'stage2_heads.pt'}")

    if history:
        first, last = history[0], history[-1]
        print(f"[stage2] DONE  "
              f"loss {first['loss']:.3f}->{last['loss']:.3f}  "
              f"ce_gene {first['ce_gene']:.3f}->{last['ce_gene']:.3f}  "
              f"mse {first['mse']:.3f}->{last['mse']:.3f}  "
              f"t_pred_std {first['t_pred_std_z']:.3f}->{last['t_pred_std_z']:.3f}")


if __name__ == "__main__":
    main()
