#!/usr/bin/env python3
import argparse
from pathlib import Path
import pickle
import numpy as np
import zarr
import cv2
from tqdm import tqdm
from typing import Dict, Any, List, Tuple
from resize_pkl import resize_with_pad

# ------------------- IO helpers -------------------

def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)

def find_episode_dirs(root: Path) -> List[Path]:
    eps = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and any(x.suffix.lower() == ".pkl" for x in p.glob("*.pkl")):
            eps.append(p)
    return eps

def list_episode_pkls(ep_dir: Path) -> List[Path]:
    return sorted([p for p in ep_dir.glob("*.pkl") if p.is_file()])

def is_sequence_episode(obj: Dict[str, Any]) -> bool:
    # Heuristic: if any expected key is T-major, treat as episode-level PKL
    for k in ["base_rgb", "wrist_rgb", "joint_positions", "gripper_position", "control"]:
        if k in obj and isinstance(obj[k], np.ndarray):
            arr = obj[k]
            # images often (T,H,W,C) for epi-level, or (H,W,C) for step-level
            if arr.ndim >= 2 and arr.shape[0] > 8:  # likely T
                return True
    return False

# ------------------- Image helpers -------------------

def to_hwc_uint8(img: np.ndarray) -> np.ndarray:
    if img.ndim != 3:
        raise ValueError(f"Expected 3D image, got {img.shape}")
    # If CHW -> HWC
    if img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
        img = np.moveaxis(img, 0, -1)
    if img.dtype != np.uint8:
        img = img.astype(np.uint8, copy=False)
    return img

def resize_hwc(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    return cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)

# ------------------- Zarr appender -------------------

class ZarrAppender:
    def __init__(self, out_path: Path):
        if out_path.exists():
            if out_path.is_dir():
                import shutil
                shutil.rmtree(out_path)
            else:
                out_path.unlink()
        self.store = zarr.DirectoryStore(str(out_path))
        self.root = zarr.group(store=self.store, overwrite=True)
        self.arrs = {}
        self.t = 0
        self.episode_ends = []

    def _ensure(self, name: str, first_val: np.ndarray, dtype=None, chunk_t: int = 64):
        if name in self.arrs:
            return
        arr = np.asarray(first_val)
        shape = (0,) + tuple(arr.shape)
        maxshape = (None,) + tuple(arr.shape)
        if dtype is None:
            dtype = arr.dtype
        chunks = (chunk_t,) + tuple(arr.shape)
        self.arrs[name] = self.root.require_dataset(
            name=name, shape=shape, maxshape=maxshape, chunks=chunks, dtype=dtype
        )

    def append(self, name: str, value: np.ndarray):
        v = np.asarray(value)
        if name not in self.arrs:
            self._ensure(name, v)
        ds = self.arrs[name]
        ds.resize((ds.shape[0] + 1,) + ds.shape[1:])
        ds[-1] = v

    def end_episode(self):
        self.episode_ends.append(self.t)

    def step_inc(self, n: int = 1):
        self.t += n

    def finalize(self):
        self.root.create_dataset("episode_ends", data=np.asarray(self.episode_ends, dtype=np.int64),
                                 dtype=np.int64, overwrite=True)

# ------------------- Core conversion -------------------

def extract_robot_state(joint_positions: np.ndarray, gripper_position: np.ndarray) -> np.ndarray:
    """
    Build 7-D robot_state = first 6 joints + gripper_position.
    - joint_positions: shape (7,) or (6,)
    - gripper_position: scalar or shape (1,)
    """
    jp = np.asarray(joint_positions).astype(np.float32)
    if jp.ndim == 0:
        jp = jp.reshape(1)
    # take first 6 joints (pad with zeros if fewer provided)
    if jp.shape[0] >= 6:
        jp6 = jp[:6]
    else:
        jp6 = np.zeros(6, dtype=np.float32)
        jp6[: jp.shape[0]] = jp

    gp = gripper_position
    if np.isscalar(gp):
        gp = np.array([gp], dtype=np.float32)
    else:
        gp = np.asarray(gp).astype(np.float32).reshape(-1)
        gp = gp[:1] if gp.size > 0 else np.zeros(1, dtype=np.float32)

    return np.concatenate([jp6, gp], axis=0).astype(np.float32)  # (7,)

def append_step_from_obj(
    obj: Dict[str, Any],
    t: int,
    app: ZarrAppender,
    out_h: int,
    out_w: int,
    include_action: bool
):
    # RGBs
    for k in ["base_rgb", "wrist_rgb"]:
        if k not in obj:
            raise KeyError(f"Missing key '{k}'")
        img = obj[k][t] if (t is not None) else obj[k]
        img = resize_with_pad(img, out_h, out_w)
        app.append(k, img)

    # robot_state = [joint_positions[:6], gripper_position]
    if "joint_positions" not in obj or "gripper_position" not in obj:
        raise KeyError("Need 'joint_positions' and 'gripper_position' in each step.")
    jp = obj["joint_positions"][t] if (t is not None) else obj["joint_positions"]
    gp = obj["gripper_position"][t] if (t is not None and isinstance(obj["gripper_position"], np.ndarray) and obj["gripper_position"].ndim>=1) else obj["gripper_position"]
    rs = extract_robot_state(jp, gp)
    app.append("robot_state", rs)

    # Optional: action
    if include_action and "control" in obj:
        act = obj["control"][t] if (t is not None) else obj["control"]
        act = np.asarray(act).astype(np.float32).reshape(-1)
        app.append("action", act)

def convert_episode(ep_pkls: List[Path], app: ZarrAppender, out_h: int, out_w: int, include_action: bool):
    pkls = [load_pickle(p) for p in ep_pkls]
    if not pkls:
        return

    # Episode-level PKL
    if len(pkls) == 1 and is_sequence_episode(pkls[0]):
        epi = pkls[0]
        # infer T from a present key
        T = None
        for key in ["base_rgb", "wrist_rgb", "joint_positions", "gripper_position", "control"]:
            if key in epi and isinstance(epi[key], np.ndarray):
                arr = epi[key]
                if arr.ndim >= 2:
                    T = arr.shape[0]
                    break
        if T is None:
            raise ValueError("Could not infer T from episode-level PKL.")
        for t in range(T):
            append_step_from_obj(epi, t, app, out_h, out_w, include_action)
            app.step_inc()
        app.end_episode()
        return

    # Step-level PKLs
    for obj in pkls:
        append_step_from_obj(obj, None, app, out_h, out_w, include_action)
        app.step_inc()
    app.end_episode()

# ------------------- CLI -------------------

def main():
    ap = argparse.ArgumentParser(description="Minimal converter to ReplayBuffer Zarr with base_rgb, wrist_rgb, robot_state(7).")
    ap.add_argument("--input_root", type=Path, required=True, help="Folder of episode subfolders containing .pkl files.")
    ap.add_argument("--out_path", type=Path, required=True, help="Output Zarr directory path.")
    ap.add_argument("--resize", nargs=2, type=int, default=[224, 224], metavar=("H", "W"), help="Target image size.")
    ap.add_argument("--include-action", type=str, default="false", help="Whether to write 'action' from 'control' (true/false).")
    args = ap.parse_args()

    input_root: Path = args.input_root
    out_path: Path = args.out_path
    out_h, out_w = map(int, args.resize)
    include_action = str(args.include_action).lower() in ("1", "true", "yes", "y")

    if not input_root.is_dir():
        raise SystemExit(f"❌ input_root is not a directory: {input_root}")

    ep_dirs = find_episode_dirs(input_root)
    if not ep_dirs:
        raise SystemExit(f"❌ No episode subfolders with .pkl files found under: {input_root}")

    app = ZarrAppender(out_path)

    total_steps = 0
    total_eps = 0
    for ep in tqdm(ep_dirs, desc="Converting episodes"):
        pkls = list_episode_pkls(ep)
        if not pkls:
            continue
        convert_episode(pkls, app, out_h, out_w, include_action)
        total_eps += 1
        total_steps = app.t

    app.finalize()
    print("\n✅ Done.")
    print(f" Episodes:    {total_eps}")
    print(f" Total steps: {total_steps}")
    print(f" Out Zarr:    {out_path}")
    keys = ['base_rgb', 'wrist_rgb', 'robot_state', 'episode_ends'] + (['action'] if include_action else [])
    print(f" Keys:        {keys}")

if __name__ == "__main__":
    main()
