import torch
import torch.nn as nn
from typing import Tuple, List

class BMCN(nn.Module):
    """Batchless Mutual‑Covariance Normalization (BMCN)

    Normalises each *individual* sample (instance) by its own statistics, then
    maintains global running estimates (\mu, \sigma^2) for evaluation, similar to
    BatchNorm but without using batch statistics for the forward pass. This
    makes it robust to tiny batch sizes (even B=1) while still offering the
    inference‑time consistency of BatchNorm.

    Args:
        num_features (int): Number of channels / features to normalise.
        eps (float): Epsilon for numerical stability.
        momentum (float): Momentum for running statistics.
        affine (bool): If True, learns scale (\gamma) and shift (\beta).
        channel_last (bool): Set True for (N, L, C) tensors (e.g. ViT tokens).
    """

    def __init__(self,
                 num_features: int,
                 eps: float = 1e-5,
                 momentum: float = 0.1,
                 affine: bool = True,
                 channel_last: bool = False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.channel_last = channel_last

        # Running (global) statistics for inference
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.running_mean: torch.Tensor
        self.running_var: torch.Tensor

        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.channel_last:
            # (N, L, C): normalise per‑token & per‑sample over L dimension
            dims: Tuple[int, ...] = (1,)  # length / seq dim
            mean = x.mean(dim=dims, keepdim=True)
            var = x.var(dim=dims, unbiased=False, keepdim=True)
            if self.training and self.momentum is not None:
                # Update running stats using mean over batch dimension N & seq L
                self._update_running_stats(mean, var, dim_batch=(0, 1))
        else:
            # (N, C, H, W…) channel‑first convolutional features
            dims = tuple(range(2, x.ndim))  # spatial dims
            mean = x.mean(dim=dims, keepdim=True)
            var = x.var(dim=dims, unbiased=False, keepdim=True)
            if self.training and self.momentum is not None:
                # Update running stats: average over N & spatial dims
                spatial_dims = tuple(range(2, x.ndim))
                self._update_running_stats(mean, var, dim_batch=(0, *spatial_dims))

        if not self.training:
            # Use running stats for inference; reshape for broadcasting
            shape: List[int] = [1, -1] + [1] * (x.ndim - 2) if not self.channel_last else [1, 1, -1]
            mean = self.running_mean.view(*shape)
            var = self.running_var.view(*shape)

        x_hat = (x - mean) / torch.sqrt(var + self.eps)

        if self.affine:
            if self.channel_last:
                w = self.weight.view(1, 1, -1)
                b = self.bias.view(1, 1, -1)
            else:
                w = self.weight.view(1, -1, *([1] * (x.ndim - 2)))
                b = self.bias.view(1, -1, *([1] * (x.ndim - 2)))
            x_hat = x_hat * w + b
        return x_hat

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _update_running_stats(self, mean: torch.Tensor, var: torch.Tensor, dim_batch: Tuple[int, ...]):
        """Momentum update of running mean/var using *detached* tensors."""
        # Collapse batch & spatial dims before updating global stats
        mean_detached = mean.detach().mean(dim=dim_batch)
        var_detached = var.detach().mean(dim=dim_batch)
        
        # Ensure the shape matches running stats
        if self.channel_last:
            # For channel_last, mean/var should be shape (C,)
            # The mean/var from forward pass will be (N, 1, C) -> need to extract (C,)
            if mean_detached.dim() > 1:
                mean_detached = mean_detached.squeeze()
                var_detached = var_detached.squeeze()
            # Take last dimension if still multi-dimensional
            if mean_detached.numel() == self.num_features:
                pass  # Already correct shape
            elif mean_detached.numel() > self.num_features:
                mean_detached = mean_detached[-self.num_features:]
                var_detached = var_detached[-self.num_features:]
        else:
            # For channel_first, mean/var should be shape (C,)
            if mean_detached.dim() > 1:
                mean_detached = mean_detached.squeeze()
                var_detached = var_detached.squeeze()
            # Ensure we have the right number of features
            if mean_detached.numel() > self.num_features:
                mean_detached = mean_detached[:self.num_features]
                var_detached = var_detached[:self.num_features]
        
        # Final safety check
        if mean_detached.numel() != self.num_features:
            # Fallback: just use the current running stats (no update)
            return
            
        self.running_mean.mul_(1.0 - self.momentum).add_(self.momentum * mean_detached)
        self.running_var.mul_(1.0 - self.momentum).add_(self.momentum * var_detached)

    # ------------------------------------------------------------------
    # Representation helpers
    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        return (f"num_features={self.num_features}, eps={self.eps}, momentum={self.momentum}, "
                f"affine={self.affine}, channel_last={self.channel_last}")

# ----------------------------------------------------------------------------
# Factory for timm "norm_layer=" argument -----------------------------------
# ----------------------------------------------------------------------------

def bmcn_norm_layer(eps: float = 1e-5, momentum: float = 0.1, affine: bool = True, channel_last: bool = False):
    """Factory so timm can build BMCN via norm_layer=bmcn_norm_layer."""
    def _create(num_features):
        return BMCN(num_features, eps=eps, momentum=momentum, affine=affine, channel_last=channel_last)
    return _create

# Example (CNNs) -------------------------------------------------------------
# import timm, functools
# model = timm.create_model(
#     'resnet18',
#     pretrained=True,
#     norm_layer=bmcn_norm_layer(eps=1e-5, momentum=0.1, affine=True)
# )
# Example (ViTs) -------------------------------------------------------------
# model = timm.create_model(
#     'vit_small_patch16_224',
#     pretrained=True,
#     # Swaps LayerNorm → BMCN (channel_last=True)
#     norm_layer=bmcn_norm_layer(channel_last=True)
# )


    # ------------------------------------------------------------------
    # Representation helpers
    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        return (f"num_features={self.num_features}, eps={self.eps}, momentum={self.momentum}, "
                f"affine={self.affine}, channel_last={self.channel_last}")

# ----------------------------------------------------------------------------
# Factory for timm "norm_layer=" argument -----------------------------------
# ----------------------------------------------------------------------------

def bmcn_norm_layer(eps: float = 1e-5, momentum: float = 0.1, affine: bool = True, channel_last: bool = False):
    """Factory so timm can build BMCN via norm_layer=bmcn_norm_layer."""
    def _create(num_features):
        return BMCN(num_features, eps=eps, momentum=momentum, affine=affine, channel_last=channel_last)
    return _create

# Example (CNNs) -------------------------------------------------------------
# import timm, functools
# model = timm.create_model(
#     'resnet18',
#     pretrained=True,
#     norm_layer=bmcn_norm_layer(eps=1e-5, momentum=0.1, affine=True)
# )
# Example (ViTs) -------------------------------------------------------------
# model = timm.create_model(
#     'vit_small_patch16_224',
#     pretrained=True,
#     # Swaps LayerNorm → BMCN (channel_last=True)
#     norm_layer=bmcn_norm_layer(channel_last=True)
# )

