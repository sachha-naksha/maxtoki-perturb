# Progress — NextCell on Aging SKM (DeltaAI)

Companion to `AGENT_NEXTCELL.md`. The DeltaAI agent appends to this file as it works
through §7.1 – §7.8 of the brief. Keep entries chronological. Do not commit heavy
artifacts (`.sif`, `.safetensors`, `.distcp`, generated-token dumps, h5ad); logs under
`deltaai/logs/` stay untracked.

---

## §7.1 DeltaAI baseline
<!-- repo commit, upstream commit, container + prefix paths, checkpoint & tokenizer SHAs,
     H5AD donor counts, pseudotime column, selected context cell IDs, PDK4 ENSG + token id -->

## §7.2 Spec + generation config
<!-- files touched; round-trip test result -->

## §7.3 Prompt construction
<!-- grammar gate output; CPU dataloader smoke on 2-row toy dataset -->

## §7.4 Predict wiring + 1-query GPU inspection
<!-- generated_tokens / lengths / finished_naturally sample; greedy reproducibility check -->

## §7.5 NextCell scorer
<!-- unit-test coverage; metric sanity on hand-crafted paired rank lists -->

## §7.6 CPU tests + 20-query GPU smoke
<!-- pytest summary; smoke job id, wall, tokens/s, peak VRAM, EOS rate (baseline / perturbed) -->

## §7.7 Launch scripts + full-run sizing
<!-- paths of smoke/full configs + sbatch scripts; projected full-run wall-clock -->

## §7.8 Handback
<!-- paste the Handback template from AGENT_NEXTCELL.md §8 here, filled in -->
