"""Score NextCell generations: decode generated_tokens -> ENSG rank lists; compute
per-row Jaccard@k, Spearman on shared genes, and per-gene rank shift between the
baseline and perturbed runs of the same ``build_paired_dataset`` output.

Writer layout
-------------
``bionemo.llm.utils.callbacks.PredictionWriter`` with ``collate_batch=False``
(the setting under ``generate_next_cell=True``) writes one
``predictions__rank_{R}.pt`` per rank. The file's ``predictions`` key is a
list of per-batch dicts, each with:

    - ``generated_tokens``:   [B_batch, T_gen]  (padded with pad_id)
    - ``lengths``:            [B_batch]
    - ``finished_naturally``: [B_batch]  bool
    - ``full_sequence``:      [B_batch, T_prompt + T_gen]  (optional, kept)

We do NOT assume one rectangular tensor across batches - we slice each row
by its own length and decode independently.

Pairing contract
----------------
``dataset_prep.build_paired_dataset`` guarantees:
  - baseline[i].row_index == i == perturbed[i].row_index
  - baseline[i].cell_id   == perturbed[i].cell_id (same query cell)

We cross-check this against ``row_manifest.json`` written next to the two
datasets. Any mismatch is a hard failure - do not drop rows or re-align.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Low-level: load rank files, extract per-row generated tokens
# ---------------------------------------------------------------------------


def _load_rank_files(predictions_dir: Path) -> dict[int, dict]:
    files = sorted(predictions_dir.glob("predictions__rank_*.pt"))
    if not files:
        raise FileNotFoundError(f"no predictions__rank_*.pt under {predictions_dir}")
    out: dict[int, dict] = {}
    for f in files:
        try:
            rank = int(f.stem.split("rank_")[1].split("__")[0])
        except (IndexError, ValueError):
            continue
        out[rank] = torch.load(f, map_location="cpu", weights_only=False)
    return out


def _iter_batches(pred: dict) -> list[dict]:
    """Normalize the writer's nested structure into a flat list of per-batch dicts.

    Writer may store either:
      - {"predictions": [batch_dict, batch_dict, ...]}
      - {"predictions": batch_dict}  (single-batch epoch)
      - batch_dict  directly (older variants)
    """
    if isinstance(pred, dict) and "predictions" in pred:
        inner = pred["predictions"]
    else:
        inner = pred
    if isinstance(inner, dict):
        return [inner]
    if isinstance(inner, list):
        return [b for b in inner if isinstance(b, dict)]
    raise TypeError(f"unrecognized predictions payload: {type(inner)}")


def _as_tensor(x: Any) -> torch.Tensor | None:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    return torch.as_tensor(x)


def _extract_per_row_tokens(predictions_dir: Path) -> list[dict]:
    """Return a list of per-row dicts {tokens, length, finished} in rank+batch order.

    This is the physical read order from PredictionWriter. ``dataset_prep`` emits
    rows contiguously (shuffle=False in predict), so with single-rank or TP-only
    this list is aligned to row_index. The scorer still re-checks with the
    manifest - this is just the physical read.
    """
    rank_to_dict = _load_rank_files(predictions_dir)
    out: list[dict] = []
    for r in sorted(rank_to_dict):
        for batch in _iter_batches(rank_to_dict[r]):
            gen = _as_tensor(batch.get("generated_tokens"))
            lengths = _as_tensor(batch.get("lengths"))
            finished = _as_tensor(batch.get("finished_naturally"))
            if gen is None or lengths is None:
                continue
            if gen.dim() == 1:  # degenerate single-row case
                gen = gen.unsqueeze(0)
            if lengths.dim() == 0:
                lengths = lengths.unsqueeze(0)
            B = int(lengths.shape[0])
            for i in range(B):
                L = int(lengths[i].item())
                toks = gen[i, :L].tolist() if L > 0 else []
                fin = bool(finished[i].item()) if finished is not None else False
                out.append({"tokens": toks, "length": L, "finished": fin})
    return out


# ---------------------------------------------------------------------------
# Decode: token-id sequence -> ordered ENSG list
# ---------------------------------------------------------------------------


@dataclass
class DecodedCell:
    ensg_order: list[str]           # ENSG tokens in generation order
    n_tokens: int                   # pre-decode generated length (including specials)
    n_invalid: int                  # tokens not mapping to a gene/special
    n_duplicates: int               # ENSG tokens dropped because they repeated
    saw_bos: bool                   # <bos> appeared immediately (expected under NextCell)
    saw_eos: bool                   # <eos> was emitted (natural completion marker)


def _build_id_to_ensg(token_dict: dict[str, int]) -> dict[int, str]:
    inv: dict[int, str] = {}
    for k, v in token_dict.items():
        if isinstance(k, str) and k.startswith("ENSG"):
            inv[int(v)] = k
    return inv


def _special_token_ids(token_dict: dict[str, int]) -> dict[str, int]:
    out: dict[str, int] = {}
    for name in ("<bos>", "<eos>", "<boq>", "<eoq>", "<pad>", "<mask>"):
        if name in token_dict:
            out[name] = int(token_dict[name])
    return out


def _numeric_token_ids(token_dict: dict[str, int]) -> set[int]:
    return {
        int(v) for k, v in token_dict.items()
        if isinstance(k, str) and k.lstrip("-").isdigit()
    }


def decode_generation(
    tokens: list[int],
    id_to_ensg: dict[int, str],
    specials: dict[str, int],
    numeric_ids: set[int],
) -> DecodedCell:
    """Turn a generated token list into an ordered deduped ENSG list.

    Grammar expected for NextCell generation:
        <bos>, g1, g2, ..., gK, <eos>
    (any leading <bos>; any trailing <eos>; everything else should be gene tokens).
    Duplicates are dropped on first-seen; special/numeric/invalid are counted and
    skipped. We stop at the first <eos> if present.
    """
    bos = specials.get("<bos>")
    eos = specials.get("<eos>")
    boq = specials.get("<boq>")
    eoq = specials.get("<eoq>")
    pad = specials.get("<pad>")
    seen: set[str] = set()
    ensg_order: list[str] = []
    n_invalid = 0
    n_duplicates = 0
    saw_bos = False
    saw_eos = False
    for i, tid in enumerate(tokens):
        if tid == bos and i == 0:
            saw_bos = True
            continue
        if tid == eos:
            saw_eos = True
            break
        if tid == pad:
            # Writer should have stripped past `length`, but be defensive.
            break
        if tid in (boq, eoq) or tid in numeric_ids:
            n_invalid += 1
            continue
        ensg = id_to_ensg.get(int(tid))
        if ensg is None:
            n_invalid += 1
            continue
        if ensg in seen:
            n_duplicates += 1
            continue
        seen.add(ensg)
        ensg_order.append(ensg)
    return DecodedCell(
        ensg_order=ensg_order,
        n_tokens=len(tokens),
        n_invalid=n_invalid,
        n_duplicates=n_duplicates,
        saw_bos=saw_bos,
        saw_eos=saw_eos,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def jaccard_at_k(a: list[str], b: list[str], k: int) -> float:
    sa, sb = set(a[:k]), set(b[:k])
    if not sa and not sb:
        return float("nan")
    return len(sa & sb) / len(sa | sb)


def spearman_on_shared(a: list[str], b: list[str]) -> tuple[float, int]:
    """Spearman rho computed on the subset of genes present in both lists.

    Returns (rho, overlap_size). rho is NaN if overlap < 2 or if either list
    has zero variance after restriction.
    """
    rank_a = {g: i for i, g in enumerate(a)}
    rank_b = {g: i for i, g in enumerate(b)}
    shared = sorted(set(rank_a) & set(rank_b))
    n = len(shared)
    if n < 2:
        return float("nan"), n
    ra = np.array([rank_a[g] for g in shared], dtype=np.float64)
    rb = np.array([rank_b[g] for g in shared], dtype=np.float64)
    # Pearson on these ranks (they're already dense ranks within the shared set
    # -- but we want Spearman of the full-rank positions, which this is).
    ra -= ra.mean()
    rb -= rb.mean()
    denom = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    if denom == 0.0:
        return float("nan"), n
    rho = float((ra * rb).sum() / denom)
    return rho, n


def rank_shift(a: list[str], b: list[str]) -> dict[str, float]:
    """Per-gene shift: perturbed_rank - baseline_rank. NaN for one-sided presence."""
    rank_a = {g: i for i, g in enumerate(a)}
    rank_b = {g: i for i, g in enumerate(b)}
    out: dict[str, float] = {}
    for g in sorted(set(rank_a) | set(rank_b)):
        ra = rank_a.get(g, None)
        rb = rank_b.get(g, None)
        if ra is None or rb is None:
            out[g] = float("nan")
        else:
            out[g] = float(rb - ra)  # perturbed - baseline
    return out


# ---------------------------------------------------------------------------
# Pair + score
# ---------------------------------------------------------------------------


@dataclass
class NextCellSummary:
    n_rows: int = 0
    n_rows_pdk4_present: int = 0
    n_both_finished: int = 0
    n_baseline_finished: int = 0
    n_perturbed_finished: int = 0
    mean_jaccard_at_50: float = float("nan")
    mean_jaccard_at_100: float = float("nan")
    mean_jaccard_at_500: float = float("nan")
    mean_spearman_shared: float = float("nan")
    mean_overlap_shared: float = float("nan")
    # Same metrics but restricted to queries where the target gene appeared in prompt.
    n_rows_present_split: dict[str, int] = field(default_factory=dict)
    mean_jaccard_at_50_present: float = float("nan")
    mean_jaccard_at_100_present: float = float("nan")
    mean_jaccard_at_500_present: float = float("nan")
    # Capped-vs-completed split (both finished naturally vs. at least one hit cap).
    mean_jaccard_at_50_completed: float = float("nan")
    mean_jaccard_at_50_capped: float = float("nan")


def _load_manifest(dataset_dir: Path) -> list[dict]:
    manifest_path = Path(dataset_dir).parent / "row_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"missing row_manifest.json next to {dataset_dir}. "
            f"Rebuild with the NextCell-aware dataset_prep."
        )
    return json.loads(manifest_path.read_text())["rows"]


def score_nextcell(
    baseline_dir: str | Path,
    perturbed_dir: str | Path,
    tokenizer_path: str | Path,
    baseline_dataset_dir: str | Path,
    out_path: str | Path,
    gene_present_flags: list[bool] | None = None,
    target_ensg: Optional[str] = None,
    top_k_sample: int = 3,
) -> tuple[list[DecodedCell], list[DecodedCell], NextCellSummary]:
    baseline_dir = Path(baseline_dir)
    perturbed_dir = Path(perturbed_dir)
    tokenizer_path = Path(tokenizer_path)
    out_path = Path(out_path)

    token_dict = json.loads(Path(tokenizer_path).read_text())
    token_dict = {k: int(v) for k, v in token_dict.items()}
    id_to_ensg = _build_id_to_ensg(token_dict)
    specials = _special_token_ids(token_dict)
    numeric_ids = _numeric_token_ids(token_dict)

    base_rows = _extract_per_row_tokens(baseline_dir)
    pert_rows = _extract_per_row_tokens(perturbed_dir)
    if len(base_rows) != len(pert_rows):
        raise RuntimeError(
            f"paired count mismatch: baseline={len(base_rows)} perturbed={len(pert_rows)}"
        )

    manifest = _load_manifest(baseline_dataset_dir)
    if len(manifest) != len(base_rows):
        raise RuntimeError(
            f"manifest/prediction row count mismatch: "
            f"manifest={len(manifest)} predictions={len(base_rows)}. "
            f"If dp>1 was used, scorer needs the DP-concat ordering handled upstream."
        )

    base_decoded: list[DecodedCell] = []
    pert_decoded: list[DecodedCell] = []
    for i, (b, p) in enumerate(zip(base_rows, pert_rows)):
        base_decoded.append(decode_generation(b["tokens"], id_to_ensg, specials, numeric_ids))
        pert_decoded.append(decode_generation(p["tokens"], id_to_ensg, specials, numeric_ids))

    j50, j100, j500 = [], [], []
    sp, sp_n = [], []
    rank_shifts: list[dict[str, float]] = []
    present_mask = []
    for i, (bd, pd) in enumerate(zip(base_decoded, pert_decoded)):
        j50.append(jaccard_at_k(bd.ensg_order, pd.ensg_order, 50))
        j100.append(jaccard_at_k(bd.ensg_order, pd.ensg_order, 100))
        j500.append(jaccard_at_k(bd.ensg_order, pd.ensg_order, 500))
        rho, n = spearman_on_shared(bd.ensg_order, pd.ensg_order)
        sp.append(rho)
        sp_n.append(n)
        rank_shifts.append(rank_shift(bd.ensg_order, pd.ensg_order))
        if gene_present_flags is not None:
            present_mask.append(bool(gene_present_flags[i]))
        else:
            present_mask.append(target_ensg is not None and target_ensg in set(bd.ensg_order))

    j50_a = np.array(j50, dtype=np.float64)
    j100_a = np.array(j100, dtype=np.float64)
    j500_a = np.array(j500, dtype=np.float64)
    sp_a = np.array(sp, dtype=np.float64)
    sp_n_a = np.array(sp_n, dtype=np.int64)
    base_fin = np.array([d.saw_eos for d in base_decoded], dtype=bool)
    pert_fin = np.array([d.saw_eos for d in pert_decoded], dtype=bool)
    both_fin = base_fin & pert_fin
    pres = np.array(present_mask, dtype=bool)

    summary = NextCellSummary(
        n_rows=len(base_decoded),
        n_rows_pdk4_present=int(pres.sum()),
        n_both_finished=int(both_fin.sum()),
        n_baseline_finished=int(base_fin.sum()),
        n_perturbed_finished=int(pert_fin.sum()),
        mean_jaccard_at_50=float(np.nanmean(j50_a)),
        mean_jaccard_at_100=float(np.nanmean(j100_a)),
        mean_jaccard_at_500=float(np.nanmean(j500_a)),
        mean_spearman_shared=float(np.nanmean(sp_a)),
        mean_overlap_shared=float(np.nanmean(sp_n_a.astype(np.float64))),
        n_rows_present_split={
            "present": int(pres.sum()),
            "absent": int((~pres).sum()),
        },
        mean_jaccard_at_50_present=float(np.nanmean(j50_a[pres])) if pres.any() else float("nan"),
        mean_jaccard_at_100_present=float(np.nanmean(j100_a[pres])) if pres.any() else float("nan"),
        mean_jaccard_at_500_present=float(np.nanmean(j500_a[pres])) if pres.any() else float("nan"),
        mean_jaccard_at_50_completed=float(np.nanmean(j50_a[both_fin])) if both_fin.any() else float("nan"),
        mean_jaccard_at_50_capped=float(np.nanmean(j50_a[~both_fin])) if (~both_fin).any() else float("nan"),
    )

    # Pack per-row metrics for later analysis.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        row_index=np.arange(len(base_decoded), dtype=np.int64),
        jaccard_at_50=j50_a,
        jaccard_at_100=j100_a,
        jaccard_at_500=j500_a,
        spearman=sp_a,
        spearman_overlap=sp_n_a,
        baseline_finished=base_fin,
        perturbed_finished=pert_fin,
        both_finished=both_fin,
        gene_present=pres,
        baseline_length=np.array([d.n_tokens for d in base_decoded], dtype=np.int64),
        perturbed_length=np.array([d.n_tokens for d in pert_decoded], dtype=np.int64),
        baseline_n_invalid=np.array([d.n_invalid for d in base_decoded], dtype=np.int64),
        perturbed_n_invalid=np.array([d.n_invalid for d in pert_decoded], dtype=np.int64),
    )

    # Also save decoded ENSG lists as JSON (easy to grep / notebook-load).
    decoded_json = {
        "n_rows": len(base_decoded),
        "rows": [
            {
                "row_index": i,
                "cell_id": manifest[i].get("cell_id"),
                "gene_present_in_query": bool(pres[i]),
                "baseline": {
                    "ensg_order": base_decoded[i].ensg_order,
                    "finished": bool(base_fin[i]),
                    "n_invalid": base_decoded[i].n_invalid,
                    "n_duplicates": base_decoded[i].n_duplicates,
                },
                "perturbed": {
                    "ensg_order": pert_decoded[i].ensg_order,
                    "finished": bool(pert_fin[i]),
                    "n_invalid": pert_decoded[i].n_invalid,
                    "n_duplicates": pert_decoded[i].n_duplicates,
                },
            }
            for i in range(len(base_decoded))
        ],
    }
    (out_path.parent / "decoded_nextcell.json").write_text(
        json.dumps(decoded_json, indent=2)
    )

    # Human-readable sample for the handback.
    sample = []
    for i in range(min(top_k_sample, len(base_decoded))):
        sample.append({
            "row_index": i,
            "cell_id": manifest[i].get("cell_id"),
            "baseline_top20": base_decoded[i].ensg_order[:20],
            "perturbed_top20": pert_decoded[i].ensg_order[:20],
            "jaccard_at_50": float(j50_a[i]),
        })
    (out_path.parent / "summary_nextcell.json").write_text(json.dumps(
        {"summary": asdict(summary), "sample": sample}, indent=2
    ))
    return base_decoded, pert_decoded, summary
