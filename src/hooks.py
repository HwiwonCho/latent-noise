import torch
import torch.nn.functional as F

class AdaptiveEntropyNoiseHook:
    """Injects Gaussian noise into a target transformer layer via forward-hook.

    sigma is adapted from *output logits entropy*:
        sigma = clamp(k / (entropy + eps), min_sigma, max_sigma)

    Notes:
    - This is intentionally minimal for archiving/repro.
    - For tuple outputs (some HF modules), we perturb output[0] (hidden_states).
    """

    def __init__(self, target_layer_idx: int, k: float = 0.0, min_sigma: float = 0.0, max_sigma: float = 0.0):
        self.target_layer_idx = int(target_layer_idx)
        self.k = float(k)
        self.min_sigma = float(min_sigma)
        self.max_sigma = float(max_sigma)
        self.current_sigma = 0.0
        self.handle = None

    def _hook_fn(self, module, inputs, output):
        if self.current_sigma <= 1e-9:
            return output

        if isinstance(output, tuple):
            hidden_states = output[0]
            noise = torch.randn_like(hidden_states) * self.current_sigma
            return (hidden_states + noise,) + output[1:]
        else:
            noise = torch.randn_like(output) * self.current_sigma
            return output + noise

    @torch.no_grad()
    def update_sigma_from_logits(self, logits: torch.Tensor, is_warmup: bool = False):
        # logits: (B, V) for the next-token distribution
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1).mean().item()

        if is_warmup:
            self.current_sigma = 0.0
            return entropy, self.current_sigma

        new_sigma = self.k / (entropy + 1e-6)
        self.current_sigma = max(self.min_sigma, min(float(new_sigma), self.max_sigma))
        return entropy, self.current_sigma

    def attach(self, model):
        layer = model.model.layers[self.target_layer_idx]
        self.handle = layer.register_forward_hook(self._hook_fn)
        return self

    def detach(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def set_seed(seed: int = 0):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
