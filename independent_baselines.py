"""Compact spherical baselines for the recovered-label diagnostic experiment."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class ResidualPlanarBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(width, width, 3, padding=1, groups=width)
        self.pointwise = nn.Conv2d(width, width, 1)
        self.norm = nn.GroupNorm(4, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = self.pointwise(self.depthwise(x))
        return x + F.gelu(self.norm(update))


class PlanarDirect(nn.Module):
    """Native-raster control without SHT or HEALPix."""

    def __init__(self, in_steps: int, out_steps: int, width: int, layers: int) -> None:
        super().__init__()
        self.input_projection = nn.Conv2d(in_steps, width, 1)
        self.blocks = nn.Sequential(*[ResidualPlanarBlock(width) for _ in range(layers)])
        self.output_projection = nn.Conv2d(width, out_steps, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_projection(self.blocks(self.input_projection(x)))


class SphericalSpectralBlock(nn.Module):
    """A genuine SHT -> learned spectral filter -> inverse SHT block."""

    def __init__(
        self,
        width: int,
        nlat: int,
        nlon: int,
        lmax: int,
        mmax: int,
    ) -> None:
        super().__init__()
        try:
            from torch_harmonics import InverseRealSHT, RealSHT
        except ImportError as exc:
            raise RuntimeError("SFNO requires torch-harmonics") from exc

        self.sht = RealSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid="equiangular")
        self.isht = InverseRealSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid="equiangular")
        scale = 1.0 / max(1, width) ** 0.5
        self.spectral_weight = nn.Parameter(
            scale * torch.randn(width, lmax, mmax, dtype=torch.cfloat)
        )
        self.local = nn.Conv2d(width, width, 1)
        self.norm = nn.GroupNorm(4, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spectrum = self.sht(x)
        spectral_update = self.isht(spectrum * self.spectral_weight.unsqueeze(0))
        update = spectral_update + self.local(x)
        return x + F.gelu(self.norm(update))


class SFNOLocalDirect(nn.Module):
    """Regional task-adapted SFNO using real spherical harmonic transforms."""

    def __init__(
        self,
        in_steps: int,
        out_steps: int,
        width: int,
        layers: int,
        nlat: int,
        nlon: int,
        lmax: int,
        mmax: int,
        regular_to_native: np.ndarray,
        native_to_regular: np.ndarray,
    ) -> None:
        super().__init__()
        self.nlat = nlat
        self.nlon = nlon
        self.register_buffer(
            "regular_to_native", torch.as_tensor(regular_to_native, dtype=torch.long)
        )
        self.register_buffer(
            "native_to_regular", torch.as_tensor(native_to_regular, dtype=torch.long)
        )
        self.input_projection = nn.Conv2d(in_steps, width, 1)
        self.blocks = nn.ModuleList(
            [
                SphericalSpectralBlock(width, nlat, nlon, lmax, mmax)
                for _ in range(layers)
            ]
        )
        self.output_projection = nn.Conv2d(width, out_steps, 1)

    @staticmethod
    def _gather_grid(x: torch.Tensor, index: torch.Tensor, h: int, w: int) -> torch.Tensor:
        flat = x.flatten(-2)
        gathered = flat.index_select(-1, index)
        return gathered.reshape(*x.shape[:-2], h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        regular = self._gather_grid(
            x, self.regular_to_native, self.nlat, self.nlon
        )
        features = self.input_projection(regular)
        for block in self.blocks:
            features = block(features)
        regular_logits = self.output_projection(features)
        return self._gather_grid(
            regular_logits,
            self.native_to_regular,
            x.shape[-2],
            x.shape[-1],
        )


class HEALPixGraphBlock(nn.Module):
    """Topology-aware message passing over true HEALPix neighbors."""

    def __init__(self, width: int, neighbors: np.ndarray) -> None:
        super().__init__()
        self.register_buffer("neighbors", torch.as_tensor(neighbors, dtype=torch.long))
        self.center = nn.Conv1d(width, width, 1)
        self.neighbor = nn.Conv1d(width, width, 1)
        self.norm = nn.GroupNorm(4, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # neighbors has shape [Npix, 8]; gathering retains cross-face adjacency.
        neighbor_values = x[:, :, self.neighbors]
        neighbor_mean = neighbor_values.mean(dim=-1)
        update = self.center(x) + self.neighbor(neighbor_mean)
        return x + F.gelu(self.norm(update))


class DLWPHPXRegionalAdapted(nn.Module):
    """Regional DLWP-HPX adaptation with true HEALPix discretization/topology."""

    def __init__(
        self,
        in_steps: int,
        out_steps: int,
        width: int,
        layers: int,
        raster_to_hpx: np.ndarray,
        hpx_neighbors: np.ndarray,
    ) -> None:
        super().__init__()
        raster_to_hpx_tensor = torch.as_tensor(raster_to_hpx, dtype=torch.long)
        self.register_buffer("raster_to_hpx", raster_to_hpx_tensor)
        npix = hpx_neighbors.shape[0]
        counts = torch.bincount(raster_to_hpx_tensor, minlength=npix).float().clamp_min(1.0)
        self.register_buffer("hpx_counts", counts)
        self.input_projection = nn.Conv1d(in_steps, width, 1)
        self.blocks = nn.ModuleList(
            [HEALPixGraphBlock(width, hpx_neighbors) for _ in range(layers)]
        )
        self.output_projection = nn.Conv1d(width, out_steps, 1)

    def raster_to_healpix(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        flat = x.reshape(batch, channels, height * width)
        index = self.raster_to_hpx.view(1, 1, -1).expand(batch, channels, -1)
        result = x.new_zeros(batch, channels, self.hpx_counts.numel())
        result.scatter_add_(2, index, flat)
        return result / self.hpx_counts.view(1, 1, -1)

    def healpix_to_raster(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        return x.index_select(-1, self.raster_to_hpx).reshape(
            x.shape[0], x.shape[1], height, width
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        features = self.input_projection(self.raster_to_healpix(x))
        for block in self.blocks:
            features = block(features)
        logits = self.output_projection(features)
        return self.healpix_to_raster(logits, height, width)

