#!/usr/bin/env python3
"""Run a transparent recovered-label SFNO/DLWP-HPX diagnostic benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from independent_baselines import DLWPHPXRegionalAdapted, PlanarDirect, SFNOLocalDirect


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPORT_ROOT = (
    ROOT / "output_predictions" / "Ours_GIS_SCIENCE_final" / "Beijing"
)
DEFAULT_GEOGRID = Path(
    "/data6/mxx_code/multimodal_cloud/outputs/era5_dstp_validation/geogrids/beijing.npz"
)
MODEL_NAMES = ("planar_control", "sfno_local", "dlwp_hpx_regional")


@dataclass(frozen=True)
class Sample:
    name: str
    paths: tuple[Path, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT_ROOT)
    parser.add_argument("--geogrid", type=Path, default=DEFAULT_GEOGRID)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--train-month", default="202401")
    parser.add_argument("--val-month", default="202404")
    parser.add_argument("--test-month", default="202407")
    parser.add_argument("--input-steps", type=int, default=24)
    parser.add_argument("--output-steps", type=int, default=24)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--nside", type=int, default=64)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--lmax", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-train", type=int, default=192)
    parser.add_argument("--max-val", type=int, default=96)
    parser.add_argument("--max-test", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--sanity-only", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_samples(root: Path, month: str, total_steps: int, limit: int) -> list[Sample]:
    month_root = root / month / "Input_32_Output_48"
    if not month_root.is_dir():
        raise FileNotFoundError(month_root)
    samples = []
    for folder in sorted(path for path in month_root.iterdir() if path.is_dir()):
        paths = tuple(sorted(folder.glob("*_step??_true.png")))
        if len(paths) >= total_steps:
            samples.append(Sample(folder.name, paths[:total_steps]))
    if limit > 0 and len(samples) > limit:
        indices = np.linspace(0, len(samples) - 1, limit, dtype=int)
        samples = [samples[index] for index in indices]
    if not samples:
        raise RuntimeError(f"No usable samples in {month_root}")
    return samples


class RecoveredCloudDataset(Dataset):
    def __init__(
        self,
        samples: list[Sample],
        input_steps: int,
        output_steps: int,
        resolution: int,
    ) -> None:
        self.samples = samples
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.resolution = resolution

    def __len__(self) -> int:
        return len(self.samples)

    def _load_mask(self, path: Path) -> tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(path).convert("RGBA").resize(
            (self.resolution, self.resolution), Image.Resampling.NEAREST
        )
        values = np.asarray(image)
        valid = values[:, :, 3] > 0
        cloud = (values[:, :, :3].mean(axis=2) < 128) & valid
        return torch.from_numpy(cloud.astype(np.float32)), torch.from_numpy(valid.copy())

    def __getitem__(self, index: int):
        sample = self.samples[index]
        frames, valid_masks = zip(*(self._load_mask(path) for path in sample.paths))
        frames_tensor = torch.stack(frames)
        valid_tensor = torch.stack(valid_masks)
        split = self.input_steps
        x = frames_tensor[:split]
        y = frames_tensor[split : split + self.output_steps]
        valid = valid_tensor[split : split + self.output_steps]
        return x, y, valid, sample.name


def unit_vectors(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    lat = np.deg2rad(latitude)
    lon = np.deg2rad(longitude)
    return np.stack(
        (np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)),
        axis=-1,
    )


def resized_geogrid(path: Path, resolution: int) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    rows = np.linspace(0, data["latitude"].shape[0] - 1, resolution).round().astype(int)
    cols = np.linspace(0, data["latitude"].shape[1] - 1, resolution).round().astype(int)
    return data["latitude"][np.ix_(rows, cols)], data["longitude"][np.ix_(rows, cols)]


def build_spherical_regrid(
    latitude: np.ndarray, longitude: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    native_vectors = unit_vectors(latitude, longitude).reshape(-1, 3)
    regular_lat = np.linspace(latitude.max(), latitude.min(), latitude.shape[0])
    regular_lon = np.linspace(longitude.min(), longitude.max(), longitude.shape[1])
    regular_latitude, regular_longitude = np.meshgrid(regular_lat, regular_lon, indexing="ij")
    regular_vectors = unit_vectors(regular_latitude, regular_longitude).reshape(-1, 3)
    regular_to_native = cKDTree(native_vectors).query(regular_vectors)[1]
    native_to_regular = cKDTree(regular_vectors).query(native_vectors)[1]
    return regular_to_native.astype(np.int64), native_to_regular.astype(np.int64)


def build_healpix_mapping(
    latitude: np.ndarray, longitude: np.ndarray, nside: int
) -> tuple[np.ndarray, np.ndarray]:
    theta = np.deg2rad(90.0 - latitude.reshape(-1))
    phi = np.mod(np.deg2rad(longitude.reshape(-1)), 2 * np.pi)
    raster_to_hpx = hp.ang2pix(nside, theta, phi, nest=True).astype(np.int64)
    npix = hp.nside2npix(nside)
    neighbors = hp.get_all_neighbours(nside, np.arange(npix), nest=True).T.astype(np.int64)
    centers = np.arange(npix)[:, None]
    neighbors = np.where(neighbors < 0, centers, neighbors)
    return raster_to_hpx, neighbors


def make_model(name: str, args: argparse.Namespace, mappings: dict) -> nn.Module:
    common = dict(
        in_steps=args.input_steps,
        out_steps=args.output_steps,
        width=args.width,
        layers=args.layers,
    )
    if name == "planar_control":
        return PlanarDirect(**common)
    if name == "sfno_local":
        return SFNOLocalDirect(
            **common,
            nlat=args.resolution,
            nlon=args.resolution,
            lmax=args.lmax,
            mmax=args.lmax,
            regular_to_native=mappings["regular_to_native"],
            native_to_regular=mappings["native_to_regular"],
        )
    if name == "dlwp_hpx_regional":
        return DLWPHPXRegionalAdapted(
            **common,
            raster_to_hpx=mappings["raster_to_hpx"],
            hpx_neighbors=mappings["hpx_neighbors"],
        )
    raise ValueError(name)


def masked_loss(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    loss = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (loss * valid).sum() / valid.sum().clamp_min(1)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    model.train(optimizer is not None)
    total_loss = 0.0
    total_items = 0
    for x, y, valid, _ in loader:
        x, y, valid = x.to(device), y.to(device), valid.to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None):
            logits = model(x)
            loss = masked_loss(logits, y, valid)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * x.shape[0]
        total_items += x.shape[0]
    return total_loss / total_items


def evaluate_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_steps: int,
) -> list[dict]:
    desired = [
        (0, "15min"),
        (1, "30min"),
        (7, "2h"),
        (15, "4h"),
        (23, "6h"),
        (47, "12h"),
    ]
    horizons = [(index, label) for index, label in desired if index < output_steps]
    counts = {index: dict(tp=0, tn=0, fp=0, fn=0) for index, _ in horizons}
    model.eval()
    with torch.no_grad():
        for x, y, valid, _ in loader:
            prediction = torch.sigmoid(model(x.to(device))) >= 0.5
            target = y.to(device).bool()
            valid = valid.to(device).bool()
            for index, _ in horizons:
                pred_i, target_i, valid_i = prediction[:, index], target[:, index], valid[:, index]
                counts[index]["tp"] += int((pred_i & target_i & valid_i).sum())
                counts[index]["tn"] += int((~pred_i & ~target_i & valid_i).sum())
                counts[index]["fp"] += int((pred_i & ~target_i & valid_i).sum())
                counts[index]["fn"] += int((~pred_i & target_i & valid_i).sum())
    rows = []
    for index, label in horizons:
        c = counts[index]
        total = sum(c.values())
        accuracy = (c["tp"] + c["tn"]) / max(1, total)
        f1 = 2 * c["tp"] / max(1, 2 * c["tp"] + c["fp"] + c["fn"])
        iou = c["tp"] / max(1, c["tp"] + c["fp"] + c["fn"])
        rows.append(dict(step=index, horizon=label, accuracy=accuracy, f1=f1, iou=iou, **c))
    return rows


def benchmark_forward(model: nn.Module, example: torch.Tensor, device: torch.device) -> dict:
    model.eval()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for _ in range(3):
            model(example)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(10):
            output = model(example)
        torch.cuda.synchronize(device)
    return {
        "output_shape": list(output.shape),
        "latency_ms_per_batch": (time.perf_counter() - start) * 1000 / 10,
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
    }


def roundtrip_report(
    sample: torch.Tensor,
    mappings: dict,
    resolution: int,
    output_dir: Path,
) -> dict:
    image = sample[0].numpy()
    r2n = mappings["regular_to_native"]
    n2r = mappings["native_to_regular"]
    spherical = image.reshape(-1)[r2n].reshape(resolution, resolution)
    spherical_back = spherical.reshape(-1)[n2r].reshape(resolution, resolution)

    pix = mappings["raster_to_hpx"]
    npix = mappings["hpx_neighbors"].shape[0]
    sums = np.bincount(pix, weights=image.reshape(-1), minlength=npix)
    counts = np.bincount(pix, minlength=npix).clip(min=1)
    healpix_values = sums / counts
    healpix_back = healpix_values[pix].reshape(resolution, resolution)

    def scores(reconstructed: np.ndarray) -> dict:
        pred = reconstructed >= 0.5
        true = image >= 0.5
        return {
            "mae": float(np.mean(np.abs(reconstructed - image))),
            "iou": float(np.logical_and(pred, true).sum() / max(1, np.logical_or(pred, true).sum())),
            "valid_pixel_ratio": 1.0,
        }

    fig, axes = plt.subplots(1, 5, figsize=(13, 2.8))
    for axis, values, title in zip(
        axes,
        (image, spherical, spherical_back, healpix_back, np.abs(healpix_back - image)),
        ("Native", "Regular spherical", "Spherical round-trip", "HEALPix round-trip", "HPX abs. error"),
    ):
        axis.imshow(values, cmap="gray", vmin=0, vmax=1)
        axis.set_title(title, fontsize=9)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "roundtrip_regridding.png", dpi=220)
    plt.close(fig)
    return {"spherical": scores(spherical_back), "healpix": scores(healpix_back)}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("This diagnostic requires a working CUDA environment")
    device = torch.device("cuda:0")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    total_steps = args.input_steps + args.output_steps
    if total_steps > 48:
        raise ValueError("Recovered exports contain only 48 consecutive labels per sample")

    train_samples = list_samples(args.export_root, args.train_month, total_steps, args.max_train)
    val_samples = list_samples(args.export_root, args.val_month, total_steps, args.max_val)
    test_samples = list_samples(args.export_root, args.test_month, total_steps, args.max_test)
    datasets = {
        "train": RecoveredCloudDataset(train_samples, args.input_steps, args.output_steps, args.resolution),
        "val": RecoveredCloudDataset(val_samples, args.input_steps, args.output_steps, args.resolution),
        "test": RecoveredCloudDataset(test_samples, args.input_steps, args.output_steps, args.resolution),
    }
    loaders = {
        "train": DataLoader(datasets["train"], args.batch_size, shuffle=True, num_workers=0),
        "val": DataLoader(datasets["val"], args.batch_size, shuffle=False, num_workers=0),
        "test": DataLoader(datasets["test"], args.batch_size, shuffle=False, num_workers=0),
    }

    latitude, longitude = resized_geogrid(args.geogrid, args.resolution)
    regular_to_native, native_to_regular = build_spherical_regrid(latitude, longitude)
    raster_to_hpx, hpx_neighbors = build_healpix_mapping(latitude, longitude, args.nside)
    mappings = dict(
        regular_to_native=regular_to_native,
        native_to_regular=native_to_regular,
        raster_to_hpx=raster_to_hpx,
        hpx_neighbors=hpx_neighbors,
    )
    example_x, _, _, _ = next(iter(loaders["train"]))
    roundtrip = roundtrip_report(example_x[0], mappings, args.resolution, args.output_dir)

    run_config = vars(args).copy()
    run_config.update(
        {
            "export_root": str(args.export_root),
            "geogrid": str(args.geogrid),
            "output_dir": str(args.output_dir),
            "sample_counts": {name: len(dataset) for name, dataset in datasets.items()},
            "experiment_status": "ADAPTED_DIAGNOSTIC_ONLY",
            "data_limitation": "Binary median-filtered true PNG exports; original 4-class 512x512 CLM tensors unavailable.",
            "tensor_flow": f"[B,{args.input_steps},{args.resolution},{args.resolution}] -> [B,{args.output_steps},{args.resolution},{args.resolution}]",
            "roundtrip": roundtrip,
            "healpix_npix": int(hp.nside2npix(args.nside)),
            "healpix_faces": 12,
            "active_healpix_pixels": int(np.unique(raster_to_hpx).size),
        }
    )

    all_metrics = []
    model_summaries = {}
    for model_name in args.models:
        print(f"\n=== {model_name} ===", flush=True)
        model = make_model(model_name, args, mappings).to(device)
        parameters = sum(parameter.numel() for parameter in model.parameters())
        smoke = benchmark_forward(model, example_x[:1].to(device), device)
        if args.sanity_only:
            model_summaries[model_name] = {"parameters": parameters, **smoke}
            del model
            torch.cuda.empty_cache()
            continue

        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        best_val = float("inf")
        best_epoch = 0
        checkpoint = args.output_dir / f"{model_name}_best.pt"
        history = []
        for epoch in range(1, args.epochs + 1):
            train_loss = run_epoch(model, loaders["train"], device, optimizer)
            val_loss = run_epoch(model, loaders["val"], device, None)
            history.append(dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss))
            print(f"epoch={epoch} train={train_loss:.5f} val={val_loss:.5f}", flush=True)
            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                torch.save(model.state_dict(), checkpoint)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        metrics = evaluate_metrics(model, loaders["test"], device, args.output_steps)
        for row in metrics:
            all_metrics.append({"model": model_name, **row})
        write_csv(args.output_dir / f"{model_name}_history.csv", history)
        model_summaries[model_name] = {
            "parameters": parameters,
            "best_epoch": best_epoch,
            "best_validation_loss": best_val,
            **smoke,
        }
        del model
        torch.cuda.empty_cache()

    run_config["models"] = model_summaries
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )
    if all_metrics:
        write_csv(args.output_dir / "test_metrics.csv", all_metrics)
    print(json.dumps(model_summaries, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
