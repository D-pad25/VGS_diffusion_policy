#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize Diffusion Policy ablation results (num_inference_steps × JPEG quality).

Inputs
------
CSV produced by eval_ablation script, with columns like:
num_inference_steps, jpeg_quality, mae_all7, rmse_all7, infer_ms_mean, psnr_base_mean, psnr_wrist_mean, ...

Outputs
-------
<out_dir>/
  rmse_vs_steps_by_jpeg.png
  mae_vs_steps_by_jpeg.png
  latency_vs_steps_by_jpeg.png
  psnr_vs_rmse_scatter.png
  eval_ablation_plots.pdf   # all figures in one PDF
"""

from __future__ import annotations
import os
import math
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


@dataclass
class Args:
    csv: str
    out_dir: str = "./viz_ablation_out"
    # Which error metrics to emphasize in line charts:
    metric_rmse: str = "rmse_all7"
    metric_mae: str = "mae_all7"
    # Optional filter (e.g., to only show compress_before_resize=True rows)
    compress_before_resize_only: Optional[bool] = None  # None means no filter
    # Whether to show grid & tight layout
    grid: bool = True


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_clean_df(csv_path: str, compress_filter: Optional[bool]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Normalize column names just in case
    df.columns = [c.strip() for c in df.columns]

    # Coerce dtypes
    num_cols = [
        "num_inference_steps", "jpeg_quality",
        "mae_all7", "rmse_all7", "mae_joints6", "rmse_joints6",
        "mae_grip", "rmse_grip",
        "mae_vs_qdelta_all7", "rmse_vs_qdelta_all7",
        "infer_ms_mean", "infer_ms_median",
        "action_chunk_len_mean",
        "psnr_base_mean", "psnr_wrist_mean",
    ]
    df = _coerce_numeric(df, num_cols)

    # Normalize boolean-ish column if present
    if "compress_before_resize" in df.columns:
        # Accept True/False/TRUE/FALSE/1/0
        df["compress_before_resize"] = df["compress_before_resize"].astype(str).str.lower().map(
            {"true": True, "false": False, "1": True, "0": False}
        ).fillna(df.get("compress_before_resize"))

    # Optional filter by compress flag
    if compress_filter is not None and "compress_before_resize" in df.columns:
        df = df[df["compress_before_resize"] == compress_filter]

    # Drop rows with missing essentials
    df = df.dropna(subset=["num_inference_steps", "jpeg_quality"])
    df["num_inference_steps"] = df["num_inference_steps"].astype(int)
    df["jpeg_quality"] = df["jpeg_quality"].astype(int)

    # Sort for nice line plots
    df = df.sort_values(["jpeg_quality", "num_inference_steps"]).reset_index(drop=True)
    return df


def line_plot_by_jpeg(df: pd.DataFrame, y: str, out_path: str, grid: bool, ylabel: Optional[str] = None, title: Optional[str] = None):
    plt.figure(figsize=(8, 5))
    for jq, sub in df.groupby("jpeg_quality"):
        sub = sub.sort_values("num_inference_steps")
        plt.plot(sub["num_inference_steps"], sub[y], marker="o", label=f"JPEG Q={jq}")
    plt.xlabel("Num inference steps")
    plt.ylabel(ylabel or y)
    if grid:
        plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(title="Compression")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def latency_plot(df: pd.DataFrame, out_path: str, grid: bool):
    plt.figure(figsize=(8, 5))
    for jq, sub in df.groupby("jpeg_quality"):
        sub = sub.sort_values("num_inference_steps")
        plt.plot(sub["num_inference_steps"], sub["infer_ms_mean"], marker="o", label=f"JPEG Q={jq}")
    plt.xlabel("Num inference steps")
    plt.ylabel("Inference time (ms, mean per chunk)")
    if grid:
        plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(title="Compression")
    plt.title("Latency vs Inference Steps")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def psnr_vs_rmse_scatter(df: pd.DataFrame, out_path: str, grid: bool, metric: str = "rmse_all7"):
    # Filter to rows that actually have PSNR (i.e., compressed cases)
    m = (~df["psnr_base_mean"].isna()) & (~df[metric].isna())
    sub = df[m].copy()
    if sub.empty:
        # Nothing to plot (e.g., only Q=0). Still write an empty plot with a note.
        plt.figure(figsize=(8, 5))
        plt.text(0.5, 0.5, "No PSNR available (no compressed rows)", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        return

    plt.figure(figsize=(8, 5))
    # One point per (nis, jq)
    # Color/marker defaults only (we are not setting specific colors per instruction)
    for (nis), sub2 in sub.groupby("num_inference_steps"):
        plt.scatter(sub2["psnr_base_mean"], sub2[metric], label=f"Steps={nis}", s=40)

    plt.xlabel("PSNR (Base camera, dB)")
    plt.ylabel(metric)
    if grid:
        plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(title="Num inference steps")
    plt.title("PSNR vs Error (compressed only)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def print_top_configs(df: pd.DataFrame, metric: str = "rmse_all7", k: int = 3):
    keep_cols = [
        "num_inference_steps", "jpeg_quality", "compress_before_resize",
        metric, "mae_all7", "rmse_all7", "infer_ms_mean", "infer_ms_median",
        "psnr_base_mean", "psnr_wrist_mean"
    ]
    available = [c for c in keep_cols if c in df.columns]
    ranked = df.sort_values([metric, "infer_ms_mean"], ascending=[True, True])
    top = ranked.head(k)[available]
    print("\nTop configurations (by {} then latency):".format(metric))
    print(top.to_string(index=False))


def main(args: Args):
    _ensure_dir(args.out_dir)
    df = load_clean_df(args.csv, args.compress_before_resize_only)

    # Print a quick leaderboard
    print_top_configs(df, metric=args.metric_rmse, k=3)

    # Build figures
    paths = {}
    paths["rmse"] = os.path.join(args.out_dir, "rmse_vs_steps_by_jpeg.png")
    line_plot_by_jpeg(
        df, y=args.metric_rmse, out_path=paths["rmse"], grid=args.grid,
        ylabel="RMSE (all 7 actions)", title="RMSE vs Inference Steps (by JPEG quality)"
    )

    paths["mae"] = os.path.join(args.out_dir, "mae_vs_steps_by_jpeg.png")
    line_plot_by_jpeg(
        df, y=args.metric_mae, out_path=paths["mae"], grid=args.grid,
        ylabel="MAE (all 7 actions)", title="MAE vs Inference Steps (by JPEG quality)"
    )

    paths["lat"] = os.path.join(args.out_dir, "latency_vs_steps_by_jpeg.png")
    latency_plot(df, out_path=paths["lat"], grid=args.grid)

    paths["psnr"] = os.path.join(args.out_dir, "psnr_vs_rmse_scatter.png")
    psnr_vs_rmse_scatter(df, out_path=paths["psnr"], grid=args.grid, metric=args.metric_rmse)

    # Save a combined PDF
    pdf_path = os.path.join(args.out_dir, "eval_ablation_plots.pdf")
    with PdfPages(pdf_path) as pdf:
        for p in [paths["rmse"], paths["mae"], paths["lat"], paths["psnr"]]:
            img = plt.imread(p)
            plt.figure(figsize=(8, 5))
            plt.imshow(img)
            plt.axis("off")
            pdf.savefig(bbox_inches="tight")
            plt.close()

    print("\nSaved figures:")
    for k, v in paths.items():
        print(f"  - {k}: {v}")
    print(f"  - all: {pdf_path}")


if __name__ == "__main__":
    try:
        import tyro  # keeps CLI consistent with your other scripts
        args = tyro.cli(Args)
    except Exception:
        # Fallback to simple argparse if tyro isn't available
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("--csv", required=True, help="Path to metrics_by_combo.csv")
        p.add_argument("--out_dir", default="./viz_ablation_out")
        p.add_argument("--metric_rmse", default="rmse_all7")
        p.add_argument("--metric_mae", default="mae_all7")
        p.add_argument("--compress_before_resize_only", type=str, default=None,
                       help="True/False to filter rows; omit for no filter")
        p.add_argument("--grid", action="store_true")
        ns = p.parse_args()
        cbr = None
        if ns.compress_before_resize_only is not None:
            cbr = ns.compress_before_resize_only.strip().lower() in ("1", "true", "yes")
        args = Args(
            csv=ns.csv,
            out_dir=ns.out_dir,
            metric_rmse=ns.metric_rmse,
            metric_mae=ns.metric_mae,
            compress_before_resize_only=cbr,
            grid=ns.grid
        )
    main(args)