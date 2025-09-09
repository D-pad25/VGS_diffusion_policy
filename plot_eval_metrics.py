#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot evaluation metrics from eval_diffusion_episode.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_metrics(csv_path: str, out_dir: str = None):
    df = pd.read_csv(csv_path)

    if out_dir is None:
        out_dir = os.path.dirname(csv_path)
    os.makedirs(out_dir, exist_ok=True)

    # --- Per-step errors ---
    plt.figure(figsize=(12,6))
    plt.plot(df["step"], df["mae_all7"], label="MAE all7")
    plt.plot(df["step"], df["rmse_all7"], label="RMSE all7")
    plt.xlabel("Step")
    plt.ylabel("Error (rad)")
    plt.title("Per-step error (all 7 DOF)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "error_all7.png"))
    plt.close()

    # --- Joints vs Gripper ---
    plt.figure(figsize=(12,6))
    plt.plot(df["step"], df["mae_joints6"], label="MAE joints6")
    plt.plot(df["step"], df["mae_grip"], label="MAE gripper")
    plt.xlabel("Step")
    plt.ylabel("Error (rad)")
    plt.title("Per-step error (joints vs gripper)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "error_joints_vs_gripper.png"))
    plt.close()

    # --- Proxy vs qdelta ---
    if "mae_vs_qdelta_all7" in df.columns:
        plt.figure(figsize=(12,6))
        plt.plot(df["step"], df["mae_vs_qdelta_all7"], label="MAE vs qdelta")
        plt.plot(df["step"], df["rmse_vs_qdelta_all7"], label="RMSE vs qdelta")
        plt.xlabel("Step")
        plt.ylabel("Error (rad)")
        plt.title("Action vs observed joint delta")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "error_vs_qdelta.png"))
        plt.close()

    print(f"[PLOT] Saved plots to {out_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Path to metrics.csv from evaluation")
    parser.add_argument("--out_dir", default=None, help="Directory to save plots")
    args = parser.parse_args()

    plot_metrics(args.csv_path, args.out_dir)
