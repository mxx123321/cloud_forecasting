"""S2T-Net variants with optional spherical residual adapters."""

from __future__ import annotations

import importlib.util
import math
import numpy as np
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F


_MODULE_PATH = Path(__file__).resolve().parents[2] / "models/GIS_final/module1.py"
_SPEC = importlib.util.spec_from_file_location("s2tnet_original_module", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load original S2T-Net module from {_MODULE_PATH}")
_ORIGINAL_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_ORIGINAL_MODULE)
Symmetric_FDED_UNet = _ORIGINAL_MODULE.Symmetric_FDED_UNet


class SFNOInputAdapter(nn.Module):
    """Regional SHT residual adapter operating on the input time channels."""

    def __init__(
        self,
        channels: int,
        height: int,
        width: int,
        lmax: int,
        regular_to_native: np.ndarray,
        native_to_regular: np.ndarray,
    ) -> None:
        super().__init__()
        from torch_harmonics import InverseRealSHT, RealSHT

        self.height = height
        self.width = width
        self.register_buffer("regular_to_native", torch.as_tensor(regular_to_native, dtype=torch.long))
        self.register_buffer("native_to_regular", torch.as_tensor(native_to_regular, dtype=torch.long))
        self.sht = RealSHT(height, width, lmax=lmax, mmax=lmax, grid="equiangular")
        self.isht = InverseRealSHT(height, width, lmax=lmax, mmax=lmax, grid="equiangular")
        self.spectral_weight = nn.Parameter(
            torch.ones(channels, lmax, lmax, dtype=torch.cfloat)
            + 0.02 * torch.randn(channels, lmax, lmax, dtype=torch.cfloat)
        )
        self.mix = nn.Conv2d(channels, channels, 1, bias=False)
        self.norm = nn.GroupNorm(math.gcd(4, channels), channels)
        self.gamma = nn.Parameter(torch.tensor(0.1))

    @staticmethod
    def _gather(x: torch.Tensor, index: torch.Tensor, height: int, width: int) -> torch.Tensor:
        return x.flatten(-2).index_select(-1, index).reshape(*x.shape[:-2], height, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        regular = self._gather(x, self.regular_to_native, self.height, self.width)
        spectrum = self.sht(regular)
        update = self.isht(spectrum * self.spectral_weight.unsqueeze(0))
        update = self.mix(update)
        native_update = self._gather(update, self.native_to_regular, *x.shape[-2:])
        return x + self.gamma * F.gelu(self.norm(native_update))


class HEALPixInputAdapter(nn.Module):
    """Residual adapter using native HEALPix discretization and adjacency."""

    def __init__(
        self,
        channels: int,
        raster_to_hpx: np.ndarray,
        hpx_neighbors: np.ndarray,
    ) -> None:
        super().__init__()
        raster_to_hpx_tensor = torch.as_tensor(raster_to_hpx, dtype=torch.long)
        self.register_buffer("raster_to_hpx", raster_to_hpx_tensor)
        self.register_buffer("neighbors", torch.as_tensor(hpx_neighbors, dtype=torch.long))
        npix = hpx_neighbors.shape[0]
        counts = torch.bincount(raster_to_hpx_tensor, minlength=npix).float().clamp_min(1.0)
        self.register_buffer("counts", counts)
        self.center = nn.Conv1d(channels, channels, 1, bias=False)
        self.neighbor = nn.Conv1d(channels, channels, 1, bias=False)
        self.norm = nn.GroupNorm(math.gcd(4, channels), channels)
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        flat = x.flatten(-2)
        index = self.raster_to_hpx.view(1, 1, -1).expand(batch, channels, -1)
        hpx = x.new_zeros(batch, channels, self.counts.numel())
        hpx.scatter_add_(2, index, flat)
        hpx = hpx / self.counts.view(1, 1, -1)
        neighbor_mean = hpx[:, :, self.neighbors].mean(dim=-1)
        update = self.center(hpx) + self.neighbor(neighbor_mean)
        raster_update = update.index_select(-1, self.raster_to_hpx).reshape_as(x)
        return x + self.gamma * F.gelu(self.norm(raster_update))


class S2TNetVariant(nn.Module):
    """Original Symmetric FDED U-Net with an optional input adapter."""

    def __init__(
        self,
        in_steps: int,
        out_steps: int,
        resolution: int,
        adapter: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.adapter = adapter if adapter is not None else nn.Identity()
        self.backbone = Symmetric_FDED_UNet(
            in_channels=in_steps,
            out_channels=out_steps,
            base_dim=32,
            img_size=resolution,
            patch_size=2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.adapter(x))
