# maxtoki-perturb

In-silico dynamic ANY GENE perturbation modules for [MaxToki](https://www.biorxiv.org/content/10.64898/2026.03.30.715396v1), NVIDIA + Christina's temporal model for predicting cell aging trajectories.

The upstream release is CUDA-only and requires NVIDIA BioNeMo.

#### Run from Apptainer/Docker using the .def (->.sif) image provided in upstream

## Status

- [.] 217M (float32), zero-shot perturbation prediction results with viz
- [.] 217M (float32), rank re-ordered prediction using FIREFate CP
- [.] 217M (float32), attention matrix augmentation for multi-modality using FIREFate dynamic TF activity

## Model details

MaxToki is a decoder-only Llama trained on 175M cells. Two variants published on HuggingFace:

| | 217M | 1B |
|---|---|---|
| Layers | 11 | 20 |
| Hidden | 1232 | 2304 |
| Heads | 8 | 16 |
| KV heads | 8 | 8 (GQA) |
| Head dim | 154 | 144 |
| Vocab | 20275 | 20275 |
| RoPE | standard | llama3 scaled |

Input format: rank-value encoding of gene_name, where its expression median-normalized across entire 175M corpus gives its rank. Each cell becomes `[<bos>, gene_rank_1, gene_rank_2, ..., <eos>]`

## Credits

- MaxToki: J Gomez Ortega et al., Theodoris Lab @ Gladstone / NVIDIA BioNeMo, bioRxiv 10.64898/2026.03.30.715396
- Geneformer (tokenizer architecture): Theodoris et al.
- MLX-port: Apple ML team & @srijitiyer

## License

Apache 2.0 (matches upstream MaxToki license).
