import torch
import torch.nn as nn
from typing import List, Tuple, Optional

class IHA(nn.Module):
    """Instance‑Aware Hyperbolic Activation (IHA)

    Implements y = sinh(κ * (x - μ)) / (κ * (σ + eps))
    where μ and σ are per‑instance statistics (computed over all non‑batch dims),
    and κ is a learnable scalar (shared) or vector (per‑channel).

    Args:
        kappa_init (float): Initial value of κ.
        per_channel (bool): Learn one κ per channel if True, else a single κ.
        eps (float): Numerical stability term added to σ.
        inplace (bool): Kept for API compatibility with timm; has no effect.
    """
    def __init__(self,
                 kappa_init: float = 0.1,
                 per_channel: bool = False,
                 eps: float = 1e-5,
                 inplace: bool = False):
        super().__init__()
        self.per_channel = per_channel
        self.eps = eps
        self.inplace = inplace  # kept to align with nn.ReLU signature

        if per_channel:
            # κ will be lazily initialized based on number of channels in first forward
            self.kappa = None  # type: Optional[nn.Parameter]
            self.register_buffer("_initialized", torch.tensor(0, dtype=torch.uint8))
            self.kappa_init = kappa_init
        else:
            self.kappa = nn.Parameter(torch.tensor(kappa_init))

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def _lazy_init_kappa(self, x: torch.Tensor):
        """Allocate per‑channel κ the first time we see input with channel dim."""
        c = x.size(1)
        kappa_vec = torch.full((c,), self.kappa_init, device=x.device, dtype=x.dtype)
        self.kappa = nn.Parameter(kappa_vec)
        self._initialized.fill_(1)

    # ---------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.per_channel and (self.kappa is None or self._initialized.item() == 0):
            self._lazy_init_kappa(x)

        # Compute per‑instance mean & std over all non‑batch dims
        dims: Tuple[int, ...] = tuple(range(1, x.ndim))
        mu = x.mean(dim=dims, keepdim=True)
        sigma = x.var(dim=dims, unbiased=False, keepdim=True).add(self.eps).sqrt()

        # Broadcast κ
        if self.per_channel:
            shape: List[int] = [1, -1] + [1] * (x.ndim - 2)
            kappa = self.kappa.view(*shape)
        else:
            kappa = self.kappa

        # Avoid numerical issues when κ ~ 0 by using a safe divisor
        safe_kappa = torch.where(kappa.abs() < 1e-4,
                                 torch.ones_like(kappa) * 1e-4,
                                 kappa)
        return torch.sinh(safe_kappa * (x - mu)) / (safe_kappa * (sigma + self.eps))

    # ---------------------------------------------------------------------
    # Representation helpers
    # ---------------------------------------------------------------------
    def extra_repr(self) -> str:
        return f"per_channel={self.per_channel}, eps={self.eps}"

# -------------------------------------------------------------------------
# Convenience factory to plug into timm ------------------------------------------------
# -------------------------------------------------------------------------

def iha_act_layer(**kwargs):
    """Factory so that timm can consume IHA via act_layer=iha_act_layer."""
    def _create(inplace: bool = False):
        return IHA(inplace=inplace, **kwargs)
    return _create

# Example usage with timm --------------------------------------------------
# import timm
# model = timm.create_model(
#     'swin_tiny_patch4_window7_224',
#     pretrained=True,
#     act_layer=iha_act_layer(kappa_init=0.1, per_channel=True)
# )
