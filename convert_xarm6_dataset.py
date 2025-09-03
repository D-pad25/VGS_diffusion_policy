# tools/convert_xarm6_dataset.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xArm6 → ReplayBuffer converter with no required CLI args.

Just run:
    python tools/convert_xarm6_dataset.py

Auto-detects:
  - dataset root (tries $XARM6_DATASET_PATH, then real_xarm_image.yaml, then CWD)
  - shape_meta (tries real_xarm_image.yaml, else uses a safe default)
  - camera res from shape_meta (falls back to 224x224)

Writes a Hydra-compatible cache zip:
  <dataset_root>/<md5>.zarr.zip
so training with RealXArm6ImageDataset(use_cache=True) will "Cache hit".

Optional flags if you need them later:
  --mode plain --out_path /path/to/output.zarr.zip
  --overwrite
  --verify_read
"""
from __future__ import annotations
import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import argparse

import numpy as np
import zarr

# repo imports
from diffusion_policy.real_world.real_xarm6_data_conversion import real_data_to_replay_buffer
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs

# optional (only used if present)
try:
    from omegaconf import OmegaConf
except Exception:
    OmegaConf = None  # will gracefully fall back to defaults

register_codecs()


# ------------------------ Defaults ------------------------


# ------------------------ Helpers ------------------------
def _repo_root() -> Path:
    # tools/convert_xarm6_dataset.py → repo root = parents[1]
    return Path(__file__).resolve().parents[1]

def _find_task_yaml_path() -> Optional[Path]:
    """
    Find diffusion_policy/config/task/real_xarm_image.yaml by walking up from this file,
    then try CWD as a fallback. Returns the first match or None.
    """
    here = Path(__file__).resolve().parent
    candidates = []
    for d in [here, *here.parents]:
        candidates.append(d / "diffusion_policy" / "config" / "task" / "real_xarm_image.yaml")
    candidates.append(Path.cwd() / "diffusion_policy" / "config" / "task" / "real_xarm_image.yaml")
    for p in candidates:
        if p.exists():
            return p
    return None

def _try_load_task_yaml() -> Optional[Dict[str, Any]]:
    """
    Load dataset_path (and optionally shape_meta) from real_xarm_image.yaml
    WITHOUT resolving interpolations (to avoid ${...} errors).
    """
    try:
        cfg_path = _find_task_yaml_path()
        if cfg_path is None:
            return None

        if OmegaConf is None:
            # Minimal fallback: parse dataset_path by scanning the file
            # (keeps things working even if OmegaConf isn't available).
            text = cfg_path.read_text(encoding="utf-8")
            for line in text.splitlines():
                if line.strip().startswith("dataset_path:"):
                    ds = line.split("dataset_path:", 1)[1].strip()
                    return {"dataset_path": ds, "shape_meta": None}
            return None

        cfg = OmegaConf.load(cfg_path.as_posix())
        dataset_path = cfg.get("dataset_path")
        # DO NOT resolve shape_meta (may contain ${...} that require full Hydra composition)
        shape_meta = None
        if "shape_meta" in cfg:
            shape_meta = OmegaConf.to_container(cfg.get("shape_meta"), resolve=False)

        # Basic sanity: dataset_path must exist
        if not dataset_path:
            return None

        print(f"[converter] Loaded YAML: {cfg_path}")
        print(f"[converter] YAML dataset_path: {dataset_path}")
        return {
            "dataset_path": str(dataset_path),
            "shape_meta": shape_meta
        }
    except Exception:
        # Last-resort: treat as not found
        return None


def _detect_dataset_root(task_cfg) -> Path:
    """
    Priority:
      1) $XARM6_DATASET_PATH
      2) real_xarm_image.yaml: task.dataset_path
    """
    # 1) ENV var
    env_path = os.environ.get("XARM6_DATASET_PATH", "").strip()
    if env_path:
        root = Path(env_path).expanduser().resolve()
        if root.exists():
            print(f"[converter] Using dataset root from $XARM6_DATASET_PATH: {root}")
            return root
    
    # 2) Task YAML
    if task_cfg and task_cfg.get("dataset_path"):
        root = Path(task_cfg["dataset_path"]).expanduser().resolve()
        if root.exists():
            print(f"[converter] Using dataset root from real_xarm_image.yaml: {root}")
            return root
        
    raise FileNotFoundError(
        "Could not find dataset root. Please set dataset_path in real_xarm_image.yaml "
        "or XARM6_DATASET_PATH, or run from the dataset folder."
    )


def _detect_shape_meta_and_res(task_cfg) -> Tuple[Dict[str, Any], Tuple[int, int]]:
    """
    Try to read shape_meta from real_xarm_image.yaml; else use DEFAULT_SHAPE_META.
    Infer camera_res (W,H) from the first rgb entry; else (224,224).
    """
    shape_meta = None
    if task_cfg and task_cfg.get("shape_meta"):
        shape_meta = task_cfg["shape_meta"]

    if shape_meta is None:
        raise FileNotFoundError(
        "Could not find shape_meta in real_xarm_image.yaml"
        )

    # infer (W,H) from any rgb obs
    cam_res = (224, 224)
    try:
        for k, v in shape_meta.get("obs", {}).items():
            if v.get("type") == "rgb" and "shape" in v:
                c, h, w = v["shape"]
                cam_res = (int(w), int(h))
                break
    except Exception:
        pass

    return shape_meta, cam_res


def _md5_cache_name(shape_meta, camera_res: Tuple[int, int]) -> str:
    fp = {"shape_meta": shape_meta, "camera_res": list(camera_res)}
    shape_meta_json = json.dumps(fp, sort_keys=True)
    return hashlib.md5(shape_meta_json.encode("utf-8")).hexdigest() + ".zarr.zip"


def _print_group(prefix: str, group):
    for k, arr in group.items():
        if isinstance(arr, zarr.Array):
            print(f"{prefix}/{k:15s} shape={arr.shape} dtype={arr.dtype} chunks={arr.chunks}")
        elif isinstance(arr, zarr.Group):
            _print_group(f"{prefix}/{k}", arr)


def _integrity_checks(store: zarr.storage.BaseStore):
    g = zarr.open(store, mode="r")
    data = g["data"]; meta = g["meta"]

    T = data["robot_state"].shape[0]
    ep_ends = meta["episode_ends"][:]
    ep_lens = meta["episode_lengths"][:]

    print("\n=== Stored arrays ===")
    _print_group("data", data)
    _print_group("meta", meta)

    print(f"\nTotal steps (T): {T}, Episodes (E): {len(ep_ends)}")
    assert ep_ends[-1] == T, "Final episode_end != total steps"
    assert np.all(ep_ends == np.cumsum(ep_lens)), "episode_ends != cumsum(episode_lengths)"
    assert data["robot_state"].shape == (T, 7)
    assert data["action"].shape == (T, 7)
    print("✅ Episode metadata consistent, and low-dim shapes OK.")


# ------------------------ Main ------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_path", type=str, default=None,
                        help="Custom output .zarr.zip path")
    parser.add_argument("--overwrite", action="store_true",
                    help="Force overwrite if output already exists")
    parser.add_argument("--verify_read", action="store_true",
                    help="Verify images after writing (slower)")
    args = parser.parse_args()

    
    # Nice defaults for thread noise
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    task_cfg = _try_load_task_yaml()
    dataset_root = _detect_dataset_root(task_cfg)
    shape_meta, camera_res = _detect_shape_meta_and_res(task_cfg)
    shape_meta = OmegaConf.to_container(shape_meta)
    image_keys = [k for k, v in shape_meta["obs"].items() if v.get("type") == "rgb"]
    # default: write Hydra-compatible cache under dataset_root
    if args.out_path is None:
        out_zip = dataset_root / _md5_cache_name(shape_meta, camera_res)
    else:
        p = Path(args.out_path).resolve()
        if args.out_path.endswith(".zip"):
            out_zip = p
        else:
            out_zip = p / _md5_cache_name(shape_meta, camera_res)

    print(f"[converter] Target cache zip: {out_zip}")
    print(f"[converter] Detected camera_res: {camera_res} (W,H)")

    if out_zip.exists() and not args.overwrite:
        print(f"[converter] Output already exists. Skipping convert. (Use --force by deleting the file)")
        with zarr.ZipStore(str(out_zip), mode="r") as zs:
            _integrity_checks(zs)
        return
    
    print("▶️ Converting raw episodes → ReplayBuffer (in-memory)…")
    mem_store = zarr.MemoryStore()
    rb: ReplayBuffer = real_data_to_replay_buffer(
        dataset_path=str(dataset_root),
        out_store=mem_store,
        out_resolutions=camera_res,                 # (W,H)
        lowdim_keys=['robot_state','action','timestamp','episode_ends','episode_lengths'],
        image_keys=image_keys,
        # sensible defaults; your function already has progress bars
        n_encoding_threads=os.cpu_count(),
        max_inflight_tasks=os.cpu_count()*5,
        verify_read=args.verify_read
    )
    print("✅ Conversion complete. Writing zip…")

    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zarr.ZipStore(str(out_zip), mode="w") as zs:
        rb.save_to_store(store=zs)
    print(f"💾 Saved: {out_zip}")

    with zarr.ZipStore(str(out_zip), mode="r") as zs:
        _integrity_checks(zs)

    print("\n✨ Done. Training with RealXArm6ImageDataset(use_cache=True) will now ‘Cache hit’.")
    print("   (Make sure shape_meta & camera_res match what you used here.)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Friendly hint for the only thing you might still need to tell it
        print(f"\n[converter] Error: {e}")
        print("Tip: set the dataset root via environment var, e.g.:")
        print("  XARM6_DATASET_PATH=/path/to/raw_episodes python tools/convert_xarm6_dataset.py")
        raise
