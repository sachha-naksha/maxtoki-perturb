"""Sanity gate for NextCell generations: do the generated cells look like real cells?

Compares each generated ranking (baseline / perturbed) from a NextCell run against
real rank-value cell tokenizations:
  - the 3 context cells exactly as the model saw them (recovered from baseline.dataset)
  - the real OM target cell of that row (re-tokenized from the h5ad)
  - a random sample of real OM and YM2 cells
and reports top-k overlaps next to real-vs-real reference overlaps.

CPU only. Run inside maxtoki-dev.sif from the repo root:
  python3 delta/scripts/sanity_nextcell.py out/nextcell_pdk4_inhibit_smoke_delta_22764696
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "torch_pipeline"))
from tokenizer import CellTokenizer  # noqa: E402

K_LIST = (50, 100, 500)
N_REF = 30
RNG = np.random.default_rng(0)


def load_h5ad(path: Path):
    import anndata as ad
    a = ad.read_h5ad(path)
    src = a.raw if a.raw is not None else a
    var = src.var
    for col in ("ensembl_id", "feature_id"):
        if col in var.columns:
            ens = var[col].astype(str).tolist()
            break
    else:
        ens = var.index.astype(str).tolist()
    return a, src.X, ens


def cell_tokens(tok: CellTokenizer, X, ens, i: int, n_counts: float) -> list[int]:
    row = X[i]
    row = row.toarray().ravel() if hasattr(row, "toarray") else np.asarray(row).ravel()
    ids = tok.tokenize_expression(ens, row, n_counts=n_counts)
    return ids[1:-1]  # strip <bos>/<eos>


def split_context_cells(input_ids: list[int], bos: int, eos: int) -> list[list[int]]:
    cells, cur, inside = [], [], False
    for t in input_ids:
        if t == bos:
            cur, inside = [], True
        elif t == eos and inside:
            cells.append(cur)
            inside = False
        elif inside:
            cur.append(t)
    return cells


def overlap(a, b, k):
    return len(set(a[:k]) & set(b[:k])) / k


def main(run_dir: Path):
    tok = CellTokenizer()
    id2ens = {v: k for k, v in tok.token_dict.items() if k.startswith("ENSG")}
    sym = json.load(open(REPO / "src/maxtoki_mlx/resources/gene_name_id.json"))
    ens2sym = {v: k for k, v in sym.items()}
    S = lambda L, n=15: [ens2sym.get(g, g) for g in L[:n]]

    spec = json.load(open(run_dir / "spec.resolved.json"))
    h5ad = REPO / spec["data"]["h5ad"]
    count_col = spec["data"].get("count_col")
    adata, X, ens = load_h5ad(h5ad)
    obs = adata.obs
    ncounts = obs[count_col].to_numpy(dtype=float) if count_col else None
    idx_of = {cid: i for i, cid in enumerate(obs.index.astype(str))}

    from datasets import load_from_disk
    ds = load_from_disk(str(run_dir / "baseline.dataset"))
    dec = {r["row_index"]: r for r in json.load(open(run_dir / "decoded_nextcell.json"))["rows"]}

    # Context cells as the model saw them (same for every row under pool strategy).
    ctx_tok = split_context_cells(ds[0]["input_ids"], tok.bos_id, tok.eos_id)
    ctx = [[id2ens[t] for t in c if t in id2ens] for c in ctx_tok]
    ctx_ids = ds[0]["context_cell_ids"]
    print(f"context cells: {ctx_ids}  lengths={[len(c) for c in ctx]}")

    # Random real reference cells.
    age = obs["age"].astype(str)
    om_idx = np.where(age == "80")[0]
    ym_idx = np.where(age == "34")[0]
    ref_om = [cell_tokens(tok, X, ens, i, ncounts[i] if ncounts is not None else None) for i in RNG.choice(om_idx, N_REF, replace=False)]
    ref_ym = [cell_tokens(tok, X, ens, i, ncounts[i] if ncounts is not None else None) for i in RNG.choice(ym_idx, N_REF, replace=False)]
    ref_om = [[id2ens[t] for t in c] for c in ref_om]
    ref_ym = [[id2ens[t] for t in c] for c in ref_ym]
    om_top500_union = set().union(*(set(c[:500]) for c in ref_om))

    print("\n== reference: real-vs-real top-k overlap (what 'looks like a cell' means) ==")
    for k in K_LIST:
        oo = np.mean([overlap(ref_om[i], ref_om[j], k) for i in range(N_REF) for j in range(i + 1, N_REF)])
        oy = np.mean([overlap(ref_om[i], ref_ym[j], k) for i in range(N_REF) for j in range(N_REF)])
        cc = overlap(ctx[-1], ctx[0], k)
        print(f"  top{k:<4} OM-vs-OM {oo:.2f}   OM-vs-YM2 {oy:.2f}   ctx3-vs-ctx1 {cc:.2f}")

    print("\n== generated vs real ==")
    hdr = f"{'row':>3} {'cond':<9}" + "".join(f" {'ctx3@'+str(k):>8}" for k in K_LIST) + "".join(f" {'tgt@'+str(k):>8}" for k in K_LIST) + f" {'OM@100':>7} {'YM@100':>7} {'inOM500':>8}"
    print(hdr)
    agg = {}
    for r in ds:
        ri = r["row_index"]
        tgt_i = idx_of[r["cell_id"]]
        tgt = [id2ens[t] for t in cell_tokens(tok, X, ens, tgt_i, ncounts[tgt_i] if ncounts is not None else None)]
        for cond in ("baseline", "perturbed"):
            g = dec[ri][cond]["ensg_order"]
            vals = [overlap(g, ctx[-1], k) for k in K_LIST] + [overlap(g, tgt, k) for k in K_LIST]
            vals += [np.mean([overlap(g, c, 100) for c in ref_om]), np.mean([overlap(g, c, 100) for c in ref_ym])]
            vals += [len(set(g[:500]) & om_top500_union) / 500]
            agg.setdefault(cond, []).append(vals)
            print(f"{ri:>3} {cond:<9}" + "".join(f" {v:>8.2f}" for v in vals[:6]) + f" {vals[6]:>7.2f} {vals[7]:>7.2f} {vals[8]:>8.2f}")
    print("mean:")
    for cond, rows in agg.items():
        m = np.mean(rows, axis=0)
        print(f"    {cond:<9}" + "".join(f" {v:>8.2f}" for v in m[:6]) + f" {m[6]:>7.2f} {m[7]:>7.2f} {m[8]:>8.2f}")

    print("\n== top-15 by symbol ==")
    print("  ctx_3 (young, pt=100):   ", S(ctx[-1]))
    print("  a real OM cell:          ", S(ref_om[0]))
    print("  generated row0 baseline: ", S(dec[0]["baseline"]["ensg_order"]))
    print("  generated row0 perturbed:", S(dec[0]["perturbed"]["ensg_order"]))
    # Where do the generated top-50 genes sit in the context cell it was conditioned on?
    g0 = dec[0]["baseline"]["ensg_order"][:50]
    pos = {e: i for i, e in enumerate(ctx[-1])}
    ranks = [pos.get(e) for e in g0]
    print(f"\n  generated-row0 top-50: {sum(r is None for r in ranks)}/50 absent from ctx_3; "
          f"median rank in ctx_3 of the rest = {np.median([r for r in ranks if r is not None]) if any(r is not None for r in ranks) else 'n/a'}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
