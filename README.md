# maxtoki-mlx

Apple Silicon MLX port of [MaxToki](https://www.biorxiv.org/content/10.64898/2026.03.30.715396v1), NVIDIA + Gladstone + Yamanaka Lab's temporal model for predicting cell aging trajectories.

The upstream release is CUDA-only and requires NVIDIA BioNeMo. This port runs on any M-series Mac using MLX.

## Install

```bash
pip install maxtoki-mlx
```

## Quick start

```python
from maxtoki_mlx import load_model, tokenize_cell, in_silico_perturb
import scanpy as sc

model = load_model("217m")  # or "1b"
adata = sc.read_h5ad("my_cells.h5ad")
tokens = tokenize_cell(adata[0])

# Delete a gene and see how cell state shifts
result = in_silico_perturb(model, tokens, gene="ENSG00000109906", direction="delete")
print(f"Age delta: {result['delta_years']:+.1f} years")
```

## Status

- [x] 217M port (float32), numerical parity with HF transformers (max abs diff 3e-5)
- [x] 1B port with GQA + llama3 RoPE scaling, numerical parity confirmed
- [x] Rank-value cell tokenizer
- [x] In-silico gene perturbation
- [ ] Temporal head (blocked on BioNeMo distcp conversion)
- [ ] CELLxGENE data loader

## Model details

MaxToki is a decoder-only Llama trained on 175M cells. Two variants published:

| | 217M | 1B |
|---|---|---|
| Layers | 11 | 20 |
| Hidden | 1232 | 2304 |
| Heads | 8 | 16 |
| KV heads | 8 | 8 (GQA) |
| Head dim | 154 | 144 |
| Vocab | 20275 | 20275 |
| RoPE | standard | llama3 scaled |

Input format: rank-value encoding of gene expression. Each cell becomes `[<bos>, gene_rank_1, gene_rank_2, ..., <eos>]` where genes are sorted descending by median-normalized expression.

## Credits

- MaxToki: J Gomez Ortega et al., Theodoris Lab @ Gladstone / NVIDIA BioNeMo, bioRxiv 10.64898/2026.03.30.715396
- Geneformer (tokenizer architecture): Theodoris et al.
- MLX: Apple ML team

The biology credit goes entirely to the MaxToki authors. This project makes their pretrained weights accessible on Apple Silicon.

## License

Apache 2.0 (matches upstream MaxToki license).
