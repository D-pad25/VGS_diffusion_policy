"""
RealXArm6ImageDataset
---------------------
Drop-in dataset class adapted for Daniel's xArm6 setup with two RealSense cameras
(wrist & base) and 7-DoF action/state (6 joints + gripper). Keeps images at 224x224.

Assumptions
- Raw data is converted via `real_data_to_replay_buffer` into a ReplayBuffer-like zarr store with keys:
    - 'base_rgb'    : (T, H, W, 3) uint8
    - 'wrist_rgb'   : (T, H, W, 3) uint8
    - 'robot_state' : (T, 7) float32   # [j1..j6, gripper]
    - 'action'      : (T, 7) float32   # same convention as robot_state
    - 'episode_ends': (E,) int64       # cumulative T indices
- Units: keep whatever your logger produced (deg or rad), but be consistent for
  both state and action. The normalizer will scale appropriately.
- Image size: 224x224 (C,H,W) after dataset transforms.
"""

from typing import Dict, List, Tuple, Optional
import os
import json
import shutil
import hashlib
import copy

import numpy as np
import torch
import zarr
import cv2

from filelock import FileLock
from threadpoolctl import threadpool_limits
from omegaconf import OmegaConf

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import (
    LinearNormalizer, SingleFieldLinearNormalizer
)
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask
)
from diffusion_policy.real_world.real_xarm6_data_conversion_low_mem import real_data_to_replay_buffer
from diffusion_policy.common.normalize_util import get_image_range_normalizer


# ---------- Lightweight on-disk RB view ----------

class ZarrReplayBufferView:
    """Minimal interface used by SequenceSampler without materializing arrays."""
    def __init__(self, store: zarr.storage.BaseStore):
        self._store = store
        self._root = zarr.open_group(store=store, mode='r')  # expects data/, meta/
        self.data = self._root['data']
        self.meta = self._root['meta']

    def __getitem__(self, key):
        if key in self.data:
            return self.data[key]
        elif key in self.meta:
            return self.meta[key]
        else:
            raise KeyError(key)

    def keys(self):
        # SequenceSampler should only iterate data keys, not meta
        return list(self.data.keys())

    def values(self):
        return [self[k] for k in self.keys()]

    def items(self):
        return [(k, self[k]) for k in self.keys()]

    @property
    def episode_ends(self):
        return self.meta['episode_ends']

    @property
    def n_episodes(self) -> int:
        return int(self.meta['episode_ends'].shape[0])

    def close(self):
        if isinstance(self._store, zarr.ZipStore):
            self._store.close()




# ---------- Safe last-dim slicer (no full array read) ----------

def _zarr_copy_lastdim(zarr_arr: zarr.Array, keep_idxs: List[int]) -> zarr.Array:
    assert zarr_arr.ndim >= 1
    old_shape = zarr_arr.shape
    new_shape = old_shape[:-1] + (len(keep_idxs),)

    parent = zarr_arr._parent
    tmp_name = zarr_arr.name + "_tmp"
    tmp = parent.require_dataset(
        name=tmp_name,
        shape=new_shape,
        chunks=zarr_arr.chunks[:-1] + (min(len(keep_idxs), zarr_arr.chunks[-1]),),
        dtype=zarr_arr.dtype,
        compressor=zarr_arr.compressor
    )
    for idx in np.ndindex(old_shape[:-1]):
        tmp[idx + (slice(None),)] = zarr_arr.oindex[idx + (keep_idxs,)]

    # Rename with fallback for older zarr
    try:
        # new-ish zarr
        del parent[zarr_arr.name]
        parent[tmp.name].rename(zarr_arr.name)
    except Exception:
        # older zarr
        parent.move(tmp.name, zarr_arr.name, allow_overwrite=True)
    return parent[zarr_arr.name]


# ---------- Streaming mean/std (Welford) with epsilon clamp ----------

def _stream_mean_std(zarr_arr: zarr.Array, axis=0, chunk_rows: int = 8192, eps: float = 1e-8):
    n = 0
    mean = None
    M2 = None
    T = zarr_arr.shape[0]
    for i in range(0, T, chunk_rows):
        x = zarr_arr[i:i+chunk_rows]
        n_batch = x.shape[0]
        if n_batch == 0:
            continue
        batch_mean = x.mean(axis=axis)
        batch_var = x.var(axis=axis, ddof=0)
        if mean is None:
            mean = batch_mean
            M2 = batch_var * n_batch
            n = n_batch
        else:
            delta = batch_mean - mean
            new_n = n + n_batch
            mean = mean + delta * (n_batch / new_n)
            M2 = M2 + batch_var * n_batch + (delta ** 2) * (n * n_batch / new_n)
            n = new_n
    std = np.sqrt(M2 / max(n - 1, 1))
    return mean, np.maximum(std, eps)


# ---------- Build/Load RB directly on disk ----------

def _get_replay_buffer(
    dataset_path: str,
    shape_meta: dict,
    store: zarr.storage.BaseStore,
    camera_res: Tuple[int, int] = (224, 224),
) -> ReplayBuffer:
    # parse obs keys & output resolutions
    rgb_keys, lowdim_keys, out_resolutions, lowdim_shapes = [], [], {}, {}
    obs_shape_meta = shape_meta["obs"]
    for key, attr in obs_shape_meta.items():
        typ = attr.get("type", "low_dim")
        shp = tuple(attr.get("shape"))
        if typ == "rgb":
            rgb_keys.append(key)
            out_resolutions[key] = (camera_res[0], camera_res[1])
        else:
            lowdim_keys.append(key)
            lowdim_shapes[key] = shp

    action_shape = tuple(shape_meta["action"]["shape"])
    assert action_shape in [(2,), (6,), (7,)], f"Unexpected action shape: {action_shape}"

    cv2.setNumThreads(1)
    with threadpool_limits(1):
        rb = real_data_to_replay_buffer(
            dataset_path=dataset_path,
            out_store=store,                         # build straight into the disk store
            out_resolutions=out_resolutions,
            lowdim_keys=lowdim_keys + ["action"],
            image_keys=rgb_keys,
            n_decoding_threads=2,
            n_encoding_threads=2,
            max_inflight_tasks=4,
            verify_read=False
        )

    # slice actions / state last-dim safely using store arrays
    root = zarr.open_group(store=store, mode="a")
    data = root["data"]
    if action_shape == (2,):
        _zarr_copy_lastdim(data["action"], [0, 1])
    elif action_shape == (6,):
        _zarr_copy_lastdim(data["action"], [0, 1, 2, 3, 4, 5])

    if "robot_state" in data and data["robot_state"].shape[-1] > 7:
        _zarr_copy_lastdim(data["robot_state"], list(range(7)))

    return rb


# ---------- Dataset ----------

class RealXArm6ImageDataset(BaseImageDataset):
    """
    Real-world xArm6 dataset for Diffusion Policy, images at 224x224 and 7-DoF actions by default.
    Avoids large RAM spikes by streaming to/from on-disk zarr stores.
    """
    def __init__(self,
                 shape_meta: dict,
                 dataset_path: str,
                 out_dir: Optional[str] = None,
                 horizon: int = 16,
                 pad_before: int = 0,
                 pad_after: int = 0,
                 n_obs_steps: Optional[int] = None,
                 n_latency_steps: int = 0,
                 use_cache: bool = True,
                 seed: int = 42,
                 val_ratio: float = 0.1,
                 max_train_episodes: Optional[int] = None,
                 delta_action: bool = False,
                 delta_exclude_gripper: bool = True,
                 camera_res: Tuple[int, int] = (224, 224),
                 work_store_kind: str = "dir",     # "dir" or "zip" for the cache file
                 work_dir_name: str = "_zarr_cache"  # (unused) for potential unzip dir
                 ) -> None:
        assert os.path.isdir(dataset_path), f"dataset_path not found: {dataset_path}"

        target_dir = out_dir if out_dir is not None else dataset_path
        fp = {"shape_meta": OmegaConf.to_container(shape_meta), "camera_res": camera_res}
        shape_meta_hash = hashlib.md5(json.dumps(fp, sort_keys=True).encode("utf-8")).hexdigest()
        cache_name = f"{shape_meta_hash}.zarr" + (".zip" if work_store_kind == "zip" else "")
        cache_path = os.path.join(target_dir, cache_name)
        cache_lock_path = cache_path + ".lock"
        print(f"[RealXArm6ImageDataset] cache_path={cache_path}")

        self._store_ref = None
        replay_buffer = None

        print(f"[RealXArm6ImageDataset] Acquiring cache lock: {cache_lock_path}")
        with FileLock(cache_lock_path):
            if not os.path.exists(cache_path):
                print("[RealXArm6ImageDataset] Cache miss → streaming conversion to DISK store...")
                if work_store_kind == "zip":
                    # build to directory first, then zip atomically
                    build_dir = os.path.join(target_dir, f"{shape_meta_hash}.zarr.dir")
                    os.makedirs(build_dir, exist_ok=True)
                    store = zarr.DirectoryStore(build_dir)
                    rb = _get_replay_buffer(
                        dataset_path=dataset_path,
                        shape_meta=shape_meta,
                        store=store,
                        camera_res=camera_res
                    )
                    # zip it
                    with zarr.ZipStore(cache_path, mode="w", compression=0) as z:
                        rb.save_to_store(z)
                    # optionally remove build_dir
                    shutil.rmtree(build_dir, ignore_errors=True)
                    self._store_ref = zarr.ZipStore(cache_path, mode="r")
                    replay_buffer = ZarrReplayBufferView(self._store_ref)
                else:
                    # write straight to a directory store (best RAM profile)
                    os.makedirs(target_dir, exist_ok=True)
                    store = zarr.DirectoryStore(os.path.join(target_dir, f"{shape_meta_hash}.zarr"))
                    _ = _get_replay_buffer(
                        dataset_path=dataset_path,
                        shape_meta=shape_meta,
                        store=store,
                        camera_res=camera_res
                    )
                    self._store_ref = store
                    replay_buffer = ZarrReplayBufferView(self._store_ref)
                print("[RealXArm6ImageDataset] Cache built.")
            else:
                print("[RealXArm6ImageDataset] Cache hit → opening on-disk store...")
                if work_store_kind == "zip":
                    self._store_ref = zarr.ZipStore(cache_path, mode="r")
                else:
                    self._store_ref = zarr.DirectoryStore(cache_path)
                replay_buffer = ZarrReplayBufferView(self._store_ref)
                print("[RealXArm6ImageDataset] Cache opened.")

        assert replay_buffer is not None

        # optional delta_action (chunked per episode)
        if delta_action:
            actions_arr = replay_buffer["action"]  # zarr.Array
            n_dim = actions_arr.shape[1]
            ep_ends = replay_buffer.episode_ends[:]
            if (delta_exclude_gripper and n_dim == 7):
                delta_idxs = list(range(6))
                keep_grip = True
            else:
                delta_idxs, keep_grip = list(range(n_dim)), False

            start = 0
            for end in ep_ends:
                seg = actions_arr[start:end]           # zarr reads this slice only
                d = np.diff(seg, axis=0, prepend=seg[0:1])
                if keep_grip:
                    d[:, 6] = seg[:, 6]
                actions_arr[start:end, delta_idxs] = d[:, delta_idxs]
                start = int(end)

        # parse keys
        rgb_keys, lowdim_keys = [], []
        for k, attr in shape_meta["obs"].items():
            (rgb_keys if attr.get("type", "low_dim") == "rgb" else lowdim_keys).append(k)

        key_first_k = {}
        if n_obs_steps is not None:
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

        # train/val split
        val_mask = get_val_mask(replay_buffer.n_episodes, val_ratio, seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)

        sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon + n_latency_steps,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k
        )

        # expose
        self.replay_buffer = replay_buffer
        self._store_keepalive = self._store_ref
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.n_obs_steps = n_obs_steps
        self.val_mask = val_mask
        self.horizon = horizon
        self.n_latency_steps = n_latency_steps
        self.pad_before = pad_before
        self.pad_after = pad_after

    def __del__(self):
        # Close ZipStore to release file handle on some platforms
        try:
            if hasattr(self, "_store_ref") and isinstance(self._store_ref, zarr.ZipStore):
                self._store_ref.close()
        except Exception:
            pass

    # -------------------- Public API --------------------
    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon + self.n_latency_steps,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.val_mask
        )
        # keep the same val_mask (no flipping)
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # action
        normalizer["action"] = SingleFieldLinearNormalizer.create_fit(
            self.replay_buffer["action"]
        )

        # low-dim obs
        for key in self.lowdim_keys:
            normalizer[key] = SingleFieldLinearNormalizer.create_fit(
                self.replay_buffer[key]
            )

        # images → [0,1]
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()

        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        # NOTE: materializes into RAM; prefer iterating in chunks if large.
        return torch.from_numpy(self.replay_buffer["action"][:])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Limit threads for dataloader workers
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)

        # Only keep first n_obs_steps of obs (if set)
        T_slice = slice(self.n_obs_steps)

        obs_dict = {}
        for key in self.rgb_keys:
            # (T,H,W,C) uint8 → (T,C,H,W) float32 in [0,1]
            obs_dict[key] = np.moveaxis(data[key][T_slice], -1, 1).astype(np.float32) / 255.0
            del data[key]

        for key in self.lowdim_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            del data[key]

        action = data["action"].astype(np.float32)
        if self.n_latency_steps > 0:
            action = action[self.n_latency_steps:]

        torch_data = {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "action": torch.from_numpy(action)
        }
        return torch_data


# -------------------- Convenience: example shape_meta --------------------

EXAMPLE_SHAPE_META = {
    "obs": {
        "base_rgb":   {"shape": [3, 224, 224], "type": "rgb"},
        "wrist_rgb":  {"shape": [3, 224, 224], "type": "rgb"},
        "robot_state": {"shape": [7], "type": "low_dim"}  # [j1..j6, gripper]
    },
    "action": {"shape": [7]}  # [j1..j6, gripper]
}


'''
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Path with raw episodes for real_data_to_replay_buffer")
    parser.add_argument("--use_cache", action="store_true", help="Cache ReplayBuffer to zipped zarr")
    parser.add_argument("--delta_action", action="store_true", help="Use delta actions (joints-only by default)")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--n_obs_steps", type=int, default=2)
    args = parser.parse_args()

    ds = RealXArm6ImageDataset(
        shape_meta=EXAMPLE_SHAPE_META,
        dataset_path=args.dataset_path,
        use_cache=args.use_cache,
        delta_action=args.delta_action,
        val_ratio=args.val_ratio,
        horizon=args.horizon,
        n_obs_steps=args.n_obs_steps,
    )
    print("Dataset ready!")
    print(f"- #episodes: {ds.replay_buffer.n_episodes}")
    print(f"- #seq (train): {len(ds)}")
    print(f"- rgb keys: {ds.rgb_keys}")
    print(f"- low-dim keys: {ds.lowdim_keys}")
'''
