# maxtoki-perturb-pipeline (Delta release)

Self-contained zero-shot gene-perturbation -> temporal-MSE pipeline for
the BioNeMo MaxToki model. CUDA / x86 only.

## Layout

```
maxtoki-perturb-pipeline/
├── scripts/torch_pipeline/         # the pipeline (Python module)
│   ├── tokenizer.py
│   ├── perturbation.py
│   ├── spec.py
│   ├── dataset_prep.py
│   ├── predict_runner.py
│   ├── score.py
│   ├── run_inhibit_temporal_mse.py
│   └── configs/
│       ├── example_self_baseline.yaml
│       ├── example_pseudotime_trajectory.yaml
│       ├── example_fixed_reference.yaml
│       └── example_young_context_old_query.yaml
├── src/maxtoki_mlx/resources/      # rank-value tokenizer JSONs
│   ├── token_dictionary.json       # backbone-only (no temporal tokens)
│   ├── gene_median.json
│   └── gene_name_id.json           # HGNC -> Ensembl
├── docs/                           # Sphinx / Read-the-Docs source
└── slurm/run_one.sbatch            # sample SLURM script (edit accounts)
```

## On Delta

```bash
# 1) extract
tar -xzf maxtoki-perturb-pipeline-*.tar.gz
cd maxtoki-perturb-pipeline

# 2) make sure the BioNeMo container has light extras
apptainer exec --nv $BIONEMO_IMG pip install --user pyyaml anndata scipy

# 3) edit the spec
$EDITOR scripts/torch_pipeline/configs/example_young_context_old_query.yaml

# 4) submit
sbatch slurm/run_one.sbatch
```

CRITICAL: pass the **full** BioNeMo token dictionary (the one shipped in your
distcp under ``context/token_dictionary.json``) via ``--tokenizer-path``. The
``token_dictionary.json`` shipped here is the trimmed maxtoki-mlx version and
**does not** contain ``<boq>`` / ``<eoq>`` / numeric temporal tokens; the
pipeline will error out at startup if it sees the trimmed dict.

## Build the docs

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs/source docs/build
xdg-open docs/build/index.html
```

## Run on RTD

The ``.readthedocs.yaml`` at the project root configures Read the Docs to
build automatically. Just connect this repo to your RTD account.

## Branch / version

This bundle was cut from branch
``claude/review-gene-perturbation-pipeline-9Aj8J`` of
``sachha-naksha/maxtoki-mlx``.
