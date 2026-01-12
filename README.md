# latent-noise: Adaptive Broadening for Topic Fixation

An exploratory research archive investigating whether controlled stochastic
perturbations in intermediate layers of Large Language Models (LLMs) can help
mitigate **topic fixation** and **context stagnation** during long-form
generation.

This repository is maintained as an **archive-first project**, recording
observations, empirical trends, and failure cases rather than presenting a
finalized methodology or strong performance claims.

---

## Motivation

During long-form generation (e.g., creative writing or extended dialogue),
LLMs often exhibit **topic fixation**: once a narrative direction or hypothesis
is established, the model tends to over-commit to the existing context,
repeatedly reinforcing similar patterns instead of exploring alternative yet
plausible continuations. This behavior frequently results in a loss of
*narrative agility*.

From a **Bayesian inference perspective**, such behavior can be interpreted as
a form of *premature posterior concentration*. The model’s implicit posterior
over latent states becomes overly peaked around a locally dominant hypothesis,
making it difficult to escape even when contextual shifts or corrective
instructions are introduced.

This project explores whether topic fixation can be addressed by regulating
the **confidence of latent state updates**, rather than by increasing
surface-level randomness at the output.

---

## Core Idea: Adaptive Broadening

Instead of modifying output sampling parameters (e.g., temperature or top-p),
this work explores **Adaptive Broadening**: selectively broadening the latent
state distribution when generation exhibits signs of stagnation.

The guiding hypothesis is that intermediate transformer layers encode a latent
space where multiple plausible continuations coexist. Overconfident updates at
this level may cause premature commitment to a single trajectory. Introducing
small, controlled stochastic perturbations may help maintain exploration while
preserving overall semantic coherence.

In this framing, noise is **not treated as a creativity mechanism**, but as a
tool for moderating confidence during latent-state updates.

---

## Scope of Exploration

This repository documents exploratory investigations including:

- Latent perturbations via forward hooks at selected transformer layers
  (e.g., Layer 20)
- Entropy- or similarity-informed heuristics for identifying stagnation
- Qualitative comparisons between baseline generation and
  latent-noise–controlled generation

Reported effects are **preliminary and qualitative**, intended to inform
further controlled study rather than establish definitive conclusions.

---

## What This Is Not

This project does **not** claim:

- Emergent consciousness, agency, or intentionality in LLMs
- A universal solution to creativity, hallucination, or alignment
- A complete Bayesian model of LLM cognition

All interpretations are descriptive and exploratory.

---

## Repository Structure

analysis/ # Research notes and qualitative interpretations
experiments/ # Raw generation logs and comparison data
notebooks/ # Google Colab notebooks (original experimental context)
src/ # Minimal implementations (hooks, controllers)
failures/ # Negative results and non-conclusive trials


---

## Notes on Notebooks

Notebooks were created in **Google Colab**. Due to widget metadata differences,
GitHub’s notebook preview may display an “Invalid Notebook” warning. The
notebooks run correctly in Colab and standard Jupyter environments.

---

## Use of AI-Assisted Tools

Portions of the experimental code were developed with the assistance of
AI-based programming tools (e.g., large language models) used as coding aids
for boilerplate generation, refactoring, and debugging.

All research decisions, including problem formulation, hypothesis design,
experimental setup, and interpretation of results, were made by the author.
The use of AI-assisted tools does not affect authorship or responsibility for
the content of this repository.

---

## Status

Exploratory. Non-conclusive. Actively evolving.

---

## License

MIT License. This project is provided as-is, without warranty.

