"""Runner skeleton (archive).

Goal:
- Load a HF causal LM
- Attach AdaptiveEntropyNoiseHook
- Step token-by-token (or small chunks), logging:
  - entropy (final logits)
  - sigma (current)
  - generated text

This file is intentionally a starting point; adapt to your exact notebook logic.
"""

from dataclasses import dataclass
from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .hooks import AdaptiveEntropyNoiseHook, set_seed

@dataclass
class RunConfig:
    model_id: str
    layer_idx: int = 20
    k: float = 0.8
    min_sigma: float = 0.0
    max_sigma: float = 0.3
    seed: int = 0
    temperature: float = 0.6
    max_new_tokens: int = 512

def main(cfg: RunConfig, prompt: str, out_path: str):
    set_seed(cfg.seed)

    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_id, device_map="auto", torch_dtype="auto")
    model.eval()

    hook = AdaptiveEntropyNoiseHook(cfg.layer_idx, cfg.k, cfg.min_sigma, cfg.max_sigma).attach(model)

    inputs = tok(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"model_id={cfg.model_id}\nlayer={cfg.layer_idx} k={cfg.k} min={cfg.min_sigma} max={cfg.max_sigma}\n\n")
        for t in range(cfg.max_new_tokens):
            out = model(input_ids=input_ids)
            logits = out.logits[:, -1, :] / max(cfg.temperature, 1e-6)
            entropy, sigma = hook.update_sigma_from_logits(logits, is_warmup=False)

            next_id = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=-1)

            text = tok.decode(next_id[0], skip_special_tokens=False)
            f.write(f"[t={t}] H={entropy:.3f} sigma={sigma:.3f} token={repr(text)}\n")

    hook.detach()

if __name__ == "__main__":
    # Example usage (edit as needed)
    cfg = RunConfig(model_id="meta-llama/Meta-Llama-3-8B-Instruct")
    prompt = "Write a scene on an orbital elevator repair bay..."
    main(cfg, prompt, out_path="outputs_log.txt")
