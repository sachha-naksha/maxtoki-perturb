Installation
============

The torch_pipeline runs inside the BioNeMo container. Outside the container,
only the dataset-prep and scoring stages will work (no ``bionemo`` /
``nemo`` / ``megatron`` available).

Inside the BioNeMo container (recommended on Delta)
---------------------------------------------------

.. code-block:: bash

   # The BioNeMo container ships these already; just confirm.
   python -c "import bionemo, nemo, megatron, datasets, torch; print('ok')"

   # Light extras the torch_pipeline needs
   pip install --user pyyaml anndata scipy

Outside the container (prep + score only)
-----------------------------------------

.. code-block:: bash

   pip install numpy scipy torch anndata datasets pyyaml

   # The model loader (predict_runner.run_headless_predict) will raise
   # ImportError because bionemo / nemo / megatron aren't installed -- this
   # is fine if you only want to build datasets and score existing
   # predictions__rank_*.pt files.

Token dictionary
----------------

The ``maxtoki_mlx`` package ships a stripped token dictionary (gene + special
tokens, no ``<boq>`` / ``<eoq>`` / numeric tokens). For the temporal head,
**always** point ``--tokenizer-path`` at the full ``token_dictionary_v1.json``
shipped with the BioNeMo distcp checkpoint, e.g.:

.. code-block:: text

   /weights/maxtoki-1b-bionemo/context/token_dictionary.json

Building the docs
-----------------

.. code-block:: bash

   pip install -r docs/requirements.txt
   sphinx-build -b html docs/source docs/build

   # Or with the bundled Makefile
   cd docs && make html
