Running on Delta
================

NCSA Delta provides A100 (40 / 80 GB) and H200 nodes plus the BioNeMo
container. The pipeline is set up to run inside that container; everything
below assumes you've ``ssh``'d into a Delta login node.

1. Stage the bundle
-------------------

Copy the released tarball to your Delta scratch:

.. code-block:: bash

   scp dist/maxtoki-perturb-pipeline-*.tar.gz <login>@login.delta.ncsa.illinois.edu:/scratch/<group>/<user>/
   ssh <login>@login.delta.ncsa.illinois.edu
   cd /scratch/<group>/<user>
   tar -xzf maxtoki-perturb-pipeline-*.tar.gz
   cd maxtoki-perturb-pipeline

Or pull directly from git:

.. code-block:: bash

   git clone <your-fork>/maxtoki-mlx.git
   cd maxtoki-mlx
   git checkout claude/review-gene-perturbation-pipeline-9Aj8J

2. Stage the BioNeMo distcp checkpoint
--------------------------------------

Either copy from somewhere local, or pull from the public release. Ensure
both the model dir and a token-dictionary JSON with ``<boq>``, ``<eoq>``, and
the numeric timestep tokens are available.

3. Stage your h5ad
------------------

The 4111-cell h5ad. Required columns covered in :doc:`quickstart`.

4. SLURM script
---------------

Single-GPU, single-node example for a 1B-parameter run with a 16k context.
Adjust ``--account`` and partitions for your allocation.

.. code-block:: bash

   #!/bin/bash
   #SBATCH --job-name=maxtoki-zlx1
   #SBATCH --partition=gpuA100x4
   #SBATCH --gres=gpu:1
   #SBATCH --cpus-per-task=8
   #SBATCH --mem=120G
   #SBATCH --time=02:00:00
   #SBATCH --account=<your-account>
   #SBATCH --output=logs/zlx1_%j.out

   set -euo pipefail
   cd /scratch/<group>/<user>/maxtoki-mlx

   # Use the BioNeMo container Delta provides (path is illustrative -- ask Delta support
   # for the exact module / image you have access to).
   module load apptainer
   IMG=/sw/external/NGC/bionemo:latest.sif

   apptainer exec --nv \
       --bind /scratch/<group>/<user>:/work \
       --bind /weights:/weights \
       "$IMG" \
       python -m scripts.torch_pipeline.run_inhibit_temporal_mse \
           --spec scripts/torch_pipeline/configs/example_young_context_old_query.yaml \
           --ckpt-dir /weights/maxtoki-1b-bionemo \
           --tokenizer-path /weights/maxtoki-1b-bionemo/context/token_dictionary.json \
           --variant 1b \
           --seq-length 16384 \
           --out-dir /work/out/zlx1 \
           --gene-symbol ZLX1 --direction inhibit

5. Multi-GPU (tensor / context parallel)
----------------------------------------

For 1B + 16k on H200, a single 80 GB H200 is generally enough at
``micro_batch_size=1``. If you want to push batch size or seq_length further,
switch on tensor parallel and / or context parallel:

.. code-block:: bash

   apptainer exec --nv ... \
       python -m scripts.torch_pipeline.run_inhibit_temporal_mse \
           ... \
           --devices 2 \
           --tensor-parallel-size 2 \
           --micro-batch-size 1

These map directly onto the upstream
``bionemo.maxtoki.predict.predict`` Megatron parallel knobs.

6. Sweeping multiple genes
--------------------------

The simplest way is one SLURM job per gene; reuse the same ``--out-dir``
prefix and one subdir per gene:

.. code-block:: bash

   for SYM in ZLX1 KLF4 SOX2 NANOG; do
     sbatch --export=GENE=$SYM run_one.sbatch
   done

In ``run_one.sbatch``, the only difference from the script above is
``--gene-symbol "$GENE"`` and ``--out-dir /work/out/$GENE``.
