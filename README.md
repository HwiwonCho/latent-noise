# Latent Noise LLM Experiments (Archive)

This repository is an **archive-first** snapshot of exploratory experiments on injecting stochastic noise
into intermediate transformer layers (e.g., Layer 20) and logging entropy signals during long-form generation.

## What this is
- A reproducible *starting point* (not a polished paper)
- A place to keep: code, configs, raw logs, and failure cases
- A baseline vs. adaptive-noise comparison archive

## What this is NOT (yet)
- A claim of consciousness / sentience
- A finalized method with thorough ablations

## Contents
- `notebooks/` : original Colab notebook snapshot (as-is)
- `experiments/`: raw generation logs (baseline + adaptive)
- `src/` : minimal extracted utilities (hook + runner skeleton)
- `configs/`: example config stubs
- `analysis/`: notes & TODOs for next steps
- `failures/`: where semantic-collapse logs should go

## Next steps (recommended)
1. Make a small **config-driven runner** that can reproduce logs deterministically (seeded).
2. Add 2–3 **ablations** (layer sweep, sigma sweep, seed sweep).
3. Add 1–2 **metrics** beyond entropy (repetition rate, topic drift, self-referential/meta rate).
