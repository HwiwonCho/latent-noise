## Notes / TODO (Archive)

### Immediate (archive-first)
- Keep original notebook intact: notebooks/colab_notebook_original.ipynb
- Keep raw logs intact: experiments/*.txt

### Next (research-mode)
- Convert notebook -> config-driven scripts
- Standardize logging format (JSONL recommended)
- Add ablations:
  - layer sweep (e.g., 10..30)
  - sigma cap sweep (0.1, 0.2, 0.3, 0.5)
  - seed sweep (0..4)
- Add metrics:
  - repetition rate (ngram)
  - meta/self-referential rate
  - topic drift (simple embedding cosine over windows)

### Safety / framing
- Avoid over-claiming ("conscious AI").
- Frame as: latent perturbation changes narrative policy / state consistency.
