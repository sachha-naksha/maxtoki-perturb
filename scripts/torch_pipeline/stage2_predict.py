"""Stage 2 inference — load the finetuned heads and predict per-cell time-lapse
on a set of query cells.

Mirrors the 0-shot eval shape (e.g. ``configs/pdk4_evenly_seq8k.yaml``):
    - Context: K cells from `--context-sample` (or every donor if --per-donor-context),
      evenly-spaced across the full pseudotime range, sorted ascending.
    - Query:   every cell in `--query-sample` (or filterable via --query-age).
    - For each query: forward pass with peak-aware sparsity attention bias,
      read hidden state at <eoq>, push through TimeBetweenHead -> z-scored
      prediction -> de-z-score with T_MEAN/T_STD saved at finetune time.

Optional inhibition delta_t (``--gene PDK4 --direction inhibit``):
    For each query cell, run forward pass twice — once on the baseline trajectory
    and once on a trajectory where the query's RVE token list has the gene
    re-ordered (inhibit moves it to the end; delete removes it; overexpress
    moves it to the front). The keep_mask for the perturbed query is rebuilt
    from reg_mass[query_cell, gene_idx] using the per-sample threshold from the
    bundle's manifest. delta_t = t_perturbed - t_baseline (negative = predicted
    rejuvenation).

Outputs (--out-dir):
    scores.npz   per-row: cell_id, query_pseudotime, t_baseline, t_perturbed (if --gene), delta_t
    summary.json eval recipe + aggregate stats
    log to stdout

Usage (baseline-only, every 80yo cell as query):
    apptainer exec --nv ... stage2_predict.py \
        --finetune-dir out/finetune_skm_full \
        --query-age 80 --context-age 34 --K 3

With PDK4 inhibition (matches pdk4_evenly_seq8k.yaml):
    stage2_predict.py --finetune-dir out/finetune_skm_full \
        --query-age 80 --context-age 34 --K 3 \
        --gene PDK4 --direction inhibit
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Reuse trainer-side helpers — same module, no duplication.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage2_finetune import (  # type: ignore
    TimeBetweenHead,
    assemble_trajectory_for_finetune,
    attach_decoder_hidden_capture,
    build_model,
    flush_main_grad_to_grad,  # not used at inference but kept for parity
    freeze_backbone_keep_lm_head,
    init_single_rank_dist,
    load_bundle_and_remap,
    load_per_cell_pseudotime,
)
from perturbation import perturb_tokens, Direction  # type: ignore


# ---------------------------------------------------------------------------
# Trajectory specs in eval shape
# ---------------------------------------------------------------------------

def build_eval_specs(
    samples_str: np.ndarray,
    pseudotime: np.ndarray,
    context_sample: str,
    query_sample: str | None,
    query_age: int | None,
    ages_int: np.ndarray,
    K: int,
) -> tuple[list[dict], list[int]]:
    """Eval-time trajectory specs:
        - K evenly-spaced context cells from `context_sample`, sorted by pt asc.
        - Query: every cell with sample==query_sample OR age==query_age
          (filter via whichever is provided).
        - time_lapse_target = pt[query] - pt[context[-1]]  (pt[~100])
    Returns (specs, ctx_idxs)."""
    ctx_pool = np.where((samples_str == context_sample) & ~np.isnan(pseudotime))[0]
    if len(ctx_pool) == 0:
        raise ValueError(f"no cells with sample={context_sample!r} (and non-NaN pseudotime)")
    order = np.argsort(pseudotime[ctx_pool], kind="stable")
    sorted_idx = ctx_pool[order]
    if K >= len(sorted_idx):
        ctx_idxs = sorted_idx.astype(int).tolist()
    else:
        positions = np.linspace(0, len(sorted_idx) - 1, K).round().astype(int)
        ctx_idxs = sorted_idx[positions].astype(int).tolist()
    last_ctx_pt = float(pseudotime[ctx_idxs[-1]])

    if query_sample is not None:
        q_mask = (samples_str == query_sample)
    elif query_age is not None:
        q_mask = (ages_int == query_age)
    else:
        raise ValueError("must pass --query-sample or --query-age")
    q_mask &= ~np.isnan(pseudotime)
    query_idxs = np.where(q_mask)[0]

    specs = [
        {
            "query_idx":    int(q),
            "context_idxs": list(ctx_idxs),
            "time_lapse":   float(pseudotime[q] - last_ctx_pt),
        }
        for q in query_idxs
    ]
    return specs, ctx_idxs


# ---------------------------------------------------------------------------
# Perturbation: rewrite query cell's input_ids + rebuild keep_mask
# ---------------------------------------------------------------------------

def find_gene_token_and_idx(
    gene_symbol: str,
    genes_df: pd.DataFrame,
    full_dict: dict[str, int],
) -> tuple[int, int, str]:
    """Resolve gene symbol -> (full-dict token id, gene_idx in bundle, ensg)."""
    matches = genes_df[genes_df["symbol"].astype(str).str.upper() == gene_symbol.upper()]
    if matches.empty:
        raise ValueError(f"gene symbol {gene_symbol!r} not in bundle's genes.parquet")
    ensg = str(matches.iloc[0]["ensg"])
    gene_idx = int(matches.iloc[0]["gene_idx"])
    if ensg not in full_dict:
        raise ValueError(f"ensg {ensg!r} (symbol {gene_symbol!r}) not in BioNeMo full token dict")
    return int(full_dict[ensg]), gene_idx, ensg


def perturb_query_row(
    cells_df: pd.DataFrame,
    query_idx: int,
    gene_token: int,
    gene_idx_in_bundle: int,
    direction: Direction,
    bos_id: int, eos_id: int,
    reg_mass_csr: sp.csr_matrix,
    sample_threshold: float,
    min_keep: int = 32,
) -> pd.DataFrame:
    """Return a *copy* of cells_df with row ``query_idx`` rewritten:
        - input_ids / gene_positions: perturb_tokens(...) applied
        - keep_mask: rebuilt from reg_mass[query_cell, gene_idx] vs threshold
                     for each token's gene position.

    The bundle's reg_mass is in 'gene_idx' space (rna_sub.var index).
    gene_positions in cells.parquet are also in that space, so look-up is direct.
    """
    new_df = cells_df.copy()
    row = new_df.iloc[query_idx]
    ids   = list(row["input_ids"])
    gpos  = list(row["gene_positions"])
    if len(ids) != len(gpos):
        raise RuntimeError("input_ids / gene_positions length mismatch")

    # perturb_tokens operates on [BOS, gene_tokens, EOS]. Build a parallel
    # gene_positions array that we walk in lockstep so we can rebuild gpos.
    if ids[0] != bos_id or ids[-1] != eos_id:
        raise RuntimeError(f"row[{query_idx}] not BOS-...-EOS framed (saw {ids[0]}/{ids[-1]})")
    inner_ids   = ids[1:-1]
    inner_gpos  = gpos[1:-1]
    if any(g < 0 for g in inner_gpos):
        raise RuntimeError(f"row[{query_idx}] inner positions contain a non-gene token")

    # Build a gene_token -> gene_position lookup so the perturbation can re-emit
    # gene_positions consistent with the rewritten id list. (Two genes never
    # share a token id under MaxToki's per-ENSG vocab.)
    tok_to_gpos = {tok: gp for tok, gp in zip(inner_ids, inner_gpos)}
    if direction == "overexpress" and gene_token not in tok_to_gpos:
        # gene was absent; we add it, so we need its gene_idx-in-bundle to
        # know the keep_mask entry. Caller supplies via `gene_idx_in_bundle`.
        tok_to_gpos[gene_token] = gene_idx_in_bundle

    new_inner = perturb_tokens(
        token_ids=ids, gene_token=gene_token, direction=direction,
        bos_id=bos_id, eos_id=eos_id,
    )
    # perturb_tokens returns the full BOS-...-EOS list; strip and re-derive gpos.
    new_inner_ids = new_inner[1:-1]
    new_inner_gpos = [tok_to_gpos[t] for t in new_inner_ids]

    # Rebuild keep_mask for the perturbed query.
    mass_row = np.asarray(reg_mass_csr[query_idx].todense()).ravel()
    new_keep_inner = np.array(
        [bool(mass_row[g] >= sample_threshold) for g in new_inner_gpos],
        dtype=bool,
    )
    # min_keep floor (matches notebook recipe in build_attn_keep_mask)
    if new_keep_inner.sum() < min_keep and len(new_keep_inner) >= min_keep:
        order = np.argsort(-mass_row[new_inner_gpos])
        floor = np.zeros_like(new_keep_inner)
        floor[order[:min_keep]] = True
        new_keep_inner = new_keep_inner | floor

    new_df.at[new_df.index[query_idx], "input_ids"] = np.asarray(
        [bos_id] + list(new_inner_ids) + [eos_id], dtype=np.int64,
    )
    new_df.at[new_df.index[query_idx], "gene_positions"] = np.asarray(
        [-1] + list(new_inner_gpos) + [-1], dtype=np.int64,
    )
    new_df.at[new_df.index[query_idx], "keep_mask"] = (
        [True] + new_keep_inner.astype(bool).tolist() + [True]
    )
    return new_df


# ---------------------------------------------------------------------------
# Inference dataset (no labels, no loss_mask)
# ---------------------------------------------------------------------------

class EvalTrajectoryDataset(Dataset):
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

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, idx):
        s = self.specs[idx]
        traj = assemble_trajectory_for_finetune(
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
        traj["query_idx"]      = s["query_idx"]
        traj["context_idxs"]   = s["context_idxs"]
        return traj


def collate_eval(batch: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    L = max(len(r["input_ids"]) for r in batch)
    B = len(batch)
    input_ids   = torch.full((B, L), pad_id, dtype=torch.long)
    keep_mask   = torch.zeros((B, L), dtype=torch.bool)
    gene_mask   = torch.zeros((B, L), dtype=torch.float32)
    seq_lens    = torch.zeros(B, dtype=torch.long)
    numeric_pos = torch.zeros(B, dtype=torch.long)
    time_lapse  = torch.zeros(B, dtype=torch.float32)
    query_idx   = torch.zeros(B, dtype=torch.long)
    for i, r in enumerate(batch):
        ell = len(r["input_ids"])
        seq_lens[i] = ell
        input_ids[i, :ell] = torch.from_numpy(r["input_ids"])
        keep_mask[i, :ell] = torch.from_numpy(r["keep_mask"])
        # query gene-token mask: 1 on [query_start:query_end] (used by the
        # head's mean-pool branch).
        gene_mask[i, r["query_start"]:r["query_end"]] = 1.0
        numeric_pos[i] = r["numeric_position"]
        time_lapse[i]  = r["time_lapse_target"]
        query_idx[i]   = r["query_idx"]
    position_ids = torch.arange(L).unsqueeze(0).expand(B, -1).contiguous()
    return {
        "input_ids":      input_ids,
        "position_ids":   position_ids,
        "keep_mask":      keep_mask,
        "gene_loss_mask": gene_mask,
        "seq_lens":       seq_lens,
        "numeric_pos":    numeric_pos,
        "time_lapse":     time_lapse,
        "query_idx":      query_idx,
    }


# ---------------------------------------------------------------------------
# Forward pass: read t_pred per query
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_dataset(
    model, time_head, decoder_capture,
    cells_df, specs,
    bos_id, eos_id, boq_id, eoq_id, pad_id, dummy_numeric_id,
    seq_length, batch_size, no_attn_bias,
    T_MEAN, T_STD,
) -> dict[str, np.ndarray]:
    ds = EvalTrajectoryDataset(
        cells_df, specs, seq_length=seq_length,
        bos_id=bos_id, eos_id=eos_id, boq_id=boq_id, eoq_id=eoq_id,
        dummy_numeric_id=dummy_numeric_id,
    )
    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=0, drop_last=False,
        collate_fn=lambda b: collate_eval(b, pad_id=pad_id),
    )

    all_t_pred, all_query_idx, all_time_lapse = [], [], []
    n_done = 0
    for batch in dl:
        ids        = batch["input_ids"].cuda(non_blocking=True)
        pos        = batch["position_ids"].cuda(non_blocking=True)
        keep       = batch["keep_mask"].cuda(non_blocking=True)
        gene_mask  = batch["gene_loss_mask"].cuda(non_blocking=True)
        num_pos    = batch["numeric_pos"].cuda(non_blocking=True)
        time_lapse = batch["time_lapse"]

        if no_attn_bias:
            extra = None
        else:
            B, L = ids.shape
            key_bias = torch.zeros(B, 1, 1, L, dtype=torch.bfloat16, device=ids.device)
            key_bias[~keep[:, None, None, :]] = -1e4
            extra = {"attention_bias": key_bias}

        # We don't need labels/loss_mask at inference; pass dummies so the
        # backbone forward path still runs (MCoreGPTModel needs labels to
        # compute lm_outputs but we ignore that output here).
        labels = ids
        loss_mask = torch.zeros_like(ids, dtype=torch.float32)
        _ = model(
            input_ids=ids, position_ids=pos, attention_mask=None,
            labels=labels, loss_mask=loss_mask, extra_block_kwargs=extra,
        )
        h_sbh = decoder_capture["h"]              # (S, B, H)
        h_bsh = h_sbh.transpose(0, 1).contiguous()
        eoq_pos = (num_pos - 1).clamp_min(0)
        h_eoq = h_bsh[torch.arange(ids.size(0), device=h_bsh.device), eoq_pos]
        # Mean-pool query gene-token hiddens (matches training-time head input).
        qmask = gene_mask.to(h_bsh.dtype)
        denom = qmask.sum(dim=1, keepdim=True).clamp_min(1.0)
        h_qpool = (h_bsh * qmask.unsqueeze(-1)).sum(dim=1) / denom
        t_pred_z = time_head(h_eoq, h_qpool).float()
        t_pred_real = t_pred_z * T_STD + T_MEAN

        all_t_pred.append(t_pred_real.cpu().numpy())
        all_query_idx.append(batch["query_idx"].numpy())
        all_time_lapse.append(time_lapse.numpy())
        n_done += ids.size(0)
        if n_done % max(1, 50 // batch_size * batch_size) == 0:
            print(f"[predict]   {n_done}/{len(ds)} queries done")
    return {
        "t_pred":     np.concatenate(all_t_pred),
        "query_idx":  np.concatenate(all_query_idx),
        "time_lapse": np.concatenate(all_time_lapse),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-dir", type=Path,
                    default=Path("/projects/bhdw/asachan/tmp/maxtoki_finetune_skm"))
    ap.add_argument("--h5ad", type=Path,
                    default=Path("/projects/bhdw/asachan/tmp/atac_rna_pairing_skm_prep/rna_sub.h5ad"))
    ap.add_argument("--pseudotime-col", default="Pseudotime")
    ap.add_argument("--ckpt-dir", type=Path,
                    default=Path("/projects/bhdw/asachan/models/MaxToki/MaxToki-217M-bionemo"))
    ap.add_argument("--tokenizer-path", type=Path,
                    default=Path("/projects/bhdw/asachan/models/MaxToki/MaxToki-217M-bionemo/context/token_dictionary.json"))
    ap.add_argument("--finetune-dir", type=Path, required=True,
                    help="Directory containing stage2_heads.pt produced by stage2_finetune.py")
    ap.add_argument("--heads-file", type=str, default=None,
                    help="Override the heads filename in --finetune-dir "
                         "(e.g. stage2_heads_step1000.pt for the best-val checkpoint).")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/projects/bhdw/asachan/methods/maxtoki-perturb/out/finetune_skm_predict"))
    ap.add_argument("--seq-length", type=int, default=8192)
    ap.add_argument("--K", type=int, default=3, help="Number of context cells.")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--no-attn-bias", action="store_true")

    # Eval shape (mirror configs/pdk4_evenly_seq8k.yaml)
    ap.add_argument("--context-sample", default="YM2",
                    help="Sample to draw the K reference-trajectory context cells from.")
    ap.add_argument("--query-sample", default=None,
                    help="If set, every cell with sample==this is a query.")
    ap.add_argument("--query-age", type=int, default=80,
                    help="If --query-sample is unset, every cell with age==this is a query.")

    # Optional inhibition
    ap.add_argument("--gene", default=None,
                    help="Gene symbol to perturb on the query (e.g. PDK4). "
                         "Triggers a second forward pass and writes delta_t.")
    ap.add_argument("--direction", choices=["inhibit", "delete", "overexpress"],
                    default="inhibit")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap the number of query cells (smoke / dev).")

    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- backbone (same path as training)
    init_single_rank_dist(seed=0)
    print(f"[predict] loading backbone from {args.ckpt_dir}")
    model, tok, full_dict = build_model(args.ckpt_dir, args.tokenizer_path, args.seq_length)
    n_trainable = freeze_backbone_keep_lm_head(model)
    print(f"[predict] params: trainable surface = {n_trainable/1e6:.2f}M (now restored from heads ckpt)")

    # ---- restore finetuned heads
    heads_path = args.finetune_dir / (args.heads_file or "stage2_heads.pt")
    print(f"[predict] loading heads from {heads_path}")
    state = torch.load(heads_path, map_location="cpu", weights_only=False)
    T_MEAN = float(state["T_MEAN"])
    T_STD  = float(state["T_STD"])
    print(f"[predict] T_MEAN={T_MEAN:.3f}  T_STD={T_STD:.3f}")
    # LM head: copy params back
    lm_state = state["lm_head_state_dict"]
    n_loaded = 0
    for name, p in model.named_parameters():
        if name in lm_state:
            with torch.no_grad():
                p.copy_(lm_state[name].to(p.device, dtype=p.dtype))
            n_loaded += 1
    print(f"[predict] restored {n_loaded} LM-head tensors")
    # TimeBetweenHead
    time_head = None
    decoder_capture = decoder_hook_handle = None
    if "time_head_state_dict" in state:
        hidden_dim = int(model.config.hidden_size)
        time_head = TimeBetweenHead(hidden_dim=hidden_dim, dropout=0.0).cuda().to(torch.bfloat16)
        time_head.load_state_dict({k: v.to(torch.bfloat16) for k, v in state["time_head_state_dict"].items()})
        time_head.eval()
        decoder_capture, decoder_hook_handle = attach_decoder_hidden_capture(model)
        print(f"[predict] restored TimeBetweenHead (hidden={hidden_dim})")
    else:
        raise RuntimeError("stage2_heads.pt has no time_head_state_dict — was --no-time-head set during finetune?")
    model.eval()

    # ---- bundle + pseudotime
    print(f"[predict] loading bundle {args.bundle_dir}")
    cells_df, genes_df = load_bundle_and_remap(args.bundle_dir, full_dict)
    pseudotime = load_per_cell_pseudotime(args.h5ad, cells_df["cell_id"].astype(str).values,
                                          col=args.pseudotime_col)
    samples_str = cells_df["sample"].astype(str).values
    ages_int    = cells_df["age_categorical"].astype(int).values

    # ---- specials (must match the trained-time encoding)
    pad_id = int(full_dict["<pad>"])
    bos_id = int(full_dict["<bos>"])
    eos_id = int(full_dict["<eos>"])
    boq_id = int(full_dict["<boq>"])
    eoq_id = int(full_dict["<eoq>"])
    nums = {int(v): int(full_dict[v]) for v in full_dict if isinstance(v, str) and v.lstrip("-").isdigit()}
    dummy_numeric_id = nums.get(0, next(iter(nums.values())))

    # ---- specs
    specs, ctx_idxs = build_eval_specs(
        samples_str, pseudotime,
        context_sample=args.context_sample,
        query_sample=args.query_sample,
        query_age=args.query_age,
        ages_int=ages_int, K=args.K,
    )
    if args.limit:
        specs = specs[: args.limit]
    ctx_pts = [float(pseudotime[c]) for c in ctx_idxs]
    print(f"[predict] context: {args.context_sample} x {args.K} cells "
          f"@ pseudotime {[round(p, 1) for p in ctx_pts]}")
    print(f"[predict] queries: {len(specs)} (sample={args.query_sample!r}, age={args.query_age})")
    if not specs:
        raise RuntimeError("no query cells matched the filter")

    # ---- baseline forward
    print(f"[predict] === baseline forward pass ===")
    base = predict_dataset(
        model, time_head, decoder_capture,
        cells_df, specs,
        bos_id, eos_id, boq_id, eoq_id, pad_id, dummy_numeric_id,
        seq_length=args.seq_length, batch_size=args.batch_size,
        no_attn_bias=args.no_attn_bias,
        T_MEAN=T_MEAN, T_STD=T_STD,
    )
    print(f"[predict] baseline t_pred  mean={base['t_pred'].mean():.3f}  "
          f"std={base['t_pred'].std():.3f}  "
          f"min={base['t_pred'].min():.3f}  max={base['t_pred'].max():.3f}")
    print(f"[predict] target  t_lapse  mean={base['time_lapse'].mean():.3f}  "
          f"std={base['time_lapse'].std():.3f}")
    if base["t_pred"].std() > 0:
        r = float(np.corrcoef(base["t_pred"], base["time_lapse"])[0, 1])
        print(f"[predict] Pearson r(t_pred, t_target) = {r:.3f}")
    else:
        r = float("nan")
        print(f"[predict] Pearson r unavailable (t_pred has zero variance)")

    # ---- optional inhibition delta
    pert_t = None
    if args.gene is not None:
        gene_token, gene_idx, ensg = find_gene_token_and_idx(args.gene, genes_df, full_dict)
        manifest = json.loads((args.bundle_dir / "manifest.json").read_text())
        thresholds = manifest["attn_sparsity"]["thresholds"]
        # reg_mass for keep_mask rebuild
        reg_mass_csr = sp.load_npz(args.bundle_dir / "reg_mass.npz")
        print(f"[predict] === perturbed forward: {args.gene} ({ensg}, token={gene_token}) "
              f"direction={args.direction} ===")

        # Apply perturbation to every query row in a fresh dataframe.
        pert_df = cells_df.copy()
        for s in specs:
            qi = s["query_idx"]
            sample = samples_str[qi]
            pert_df = perturb_query_row(
                pert_df, query_idx=qi,
                gene_token=gene_token, gene_idx_in_bundle=gene_idx,
                direction=args.direction,
                bos_id=bos_id, eos_id=eos_id,
                reg_mass_csr=reg_mass_csr,
                sample_threshold=float(thresholds[sample]),
            )
        pert = predict_dataset(
            model, time_head, decoder_capture,
            pert_df, specs,
            bos_id, eos_id, boq_id, eoq_id, pad_id, dummy_numeric_id,
            seq_length=args.seq_length, batch_size=args.batch_size,
            no_attn_bias=args.no_attn_bias,
            T_MEAN=T_MEAN, T_STD=T_STD,
        )
        delta_t = pert["t_pred"] - base["t_pred"]
        pert_t = pert["t_pred"]
        print(f"[predict] perturbed t_pred  mean={pert_t.mean():.3f}  std={pert_t.std():.3f}")
        print(f"[predict] delta_t (pert-base)  mean={delta_t.mean():.3f}  "
              f"std={delta_t.std():.3f}  median={np.median(delta_t):.3f}")
        n_neg = int((delta_t < 0).sum())
        print(f"[predict] {n_neg}/{len(delta_t)} cells have delta_t<0 (predicted rejuvenation)")

    # ---- save
    cell_ids = cells_df["cell_id"].astype(str).values
    save_dict = dict(
        cell_id=cell_ids[base["query_idx"]],
        query_pseudotime=pseudotime[base["query_idx"]],
        query_idx=base["query_idx"],
        time_lapse_target=base["time_lapse"],
        t_baseline=base["t_pred"],
    )
    if pert_t is not None:
        save_dict["t_perturbed"] = pert_t
        save_dict["delta_t"] = pert_t - base["t_pred"]
    np.savez(args.out_dir / "scores.npz", **save_dict)
    print(f"[predict] wrote {args.out_dir / 'scores.npz'}")

    summary = {
        "finetune_dir":   str(args.finetune_dir),
        "ckpt_dir":       str(args.ckpt_dir),
        "context_sample": args.context_sample,
        "context_idxs":   ctx_idxs,
        "context_pseudotimes": ctx_pts,
        "query_sample":   args.query_sample,
        "query_age":      args.query_age,
        "K":              args.K,
        "seq_length":     args.seq_length,
        "n_queries":      len(specs),
        "T_MEAN":         T_MEAN,
        "T_STD":          T_STD,
        "no_attn_bias":   args.no_attn_bias,
        "baseline": {
            "t_pred_mean": float(base["t_pred"].mean()),
            "t_pred_std":  float(base["t_pred"].std()),
            "pearson_r_to_target": r,
        },
    }
    if args.gene is not None:
        delta_t = pert_t - base["t_pred"]
        summary["perturbation"] = {
            "gene":      args.gene,
            "ensg":      ensg,
            "direction": args.direction,
            "delta_t_mean":  float(delta_t.mean()),
            "delta_t_std":   float(delta_t.std()),
            "delta_t_median":float(np.median(delta_t)),
            "n_negative":    int((delta_t < 0).sum()),
        }
        # Per-donor breakdown — does the model give DIFFERENT answers for
        # OM6 vs OM9? That's the personalization claim.
        per_sample = {}
        for s in sorted(set(samples_str[base["query_idx"]])):
            mask = samples_str[base["query_idx"]] == s
            if mask.sum() == 0:
                continue
            per_sample[s] = {
                "n":             int(mask.sum()),
                "t_baseline_mean": float(base["t_pred"][mask].mean()),
                "t_baseline_std":  float(base["t_pred"][mask].std()),
                "delta_t_mean":  float(delta_t[mask].mean()),
                "delta_t_std":   float(delta_t[mask].std()),
                "n_negative":    int((delta_t[mask] < 0).sum()),
            }
        summary["perturbation"]["per_sample"] = per_sample
        print(f"[predict] per-donor delta_t breakdown:")
        for s, d in per_sample.items():
            print(f"           {s}: n={d['n']:>4d}  t_base={d['t_baseline_mean']:+.2f}±{d['t_baseline_std']:.2f}  "
                  f"delta_t={d['delta_t_mean']:+.4f}±{d['delta_t_std']:.4f}  "
                  f"n_neg={d['n_negative']}/{d['n']}")
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[predict] wrote {args.out_dir / 'summary.json'}")
    print(f"[predict] DONE")

    if decoder_hook_handle is not None:
        decoder_hook_handle.remove()


if __name__ == "__main__":
    main()
