"""
RealXArm6LowdimDataset
----------------------
Low-dim–only dataset for Daniel's xArm6 setup.
Loads ONLY 'robot_state' (+ 'action') from converted episodes; completely
skips image decode to minimize CPU RAM and dataloader overhead.

Assumptions
- Converted episodes produced by:
    diffusion_policy.real_world.real_xarm6_data_conversion.real_data_to_replay_buffer
- Keys present in the converted store:
    - 'robot_state': (T, 7) float32  # [j1..j6, gripper]
    - 'action'     : (T, 7) float32
    - 'episode_ends': (E,) int64  (cumulative T)
- Units: whatever your logger used (deg/rad). Normalizer will scale properly.
"""

from typing import Dict, Optional, List
import os, json, hashlib, shutil, copy
import numpy as np
import torch, zarr
from filelock import FileLock
from threadpoolctl import threadpool_limits
from omegaconf import OmegaConf

from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.normalizer import (
    LinearNormalizer, SingleFieldLinearNormalizer
)
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask
)
from diffusion_policy.real_world.real_xarm6_data_conversion import real_data_to_replay_buffer


def _build_lowdim_replay_buffer(
    dataset_path: str,
    lowdim_keys: List[str],
    action_shape: tuple,
    store: zarr.storage.BaseStore,
) -> ReplayBuffer:
    """Create a ReplayBuffer with ONLY low-dim keys + action (no images)."""
    # No image keys, no resize, just ingest low-dim fields and action
    with threadpool_limits(1):
        rb = real_data_to_replay_buffer(
            dataset_path=dataset_path,
            out_store=store,
            out_resolutions={},                 # no images → no resizing
            lowdim_keys=lowdim_keys + ["action"],
            image_keys=[],                      # critical: skip image decode
        )

    # Sanity slice actions if larger than expected (keep first N)
    if action_shape in [(2,), (6,), (7,)]:
        want = action_shape[-1]
        if rb["action"].shape[-1] > want:
            _zarr_slice_last_dim_(rb["action"], list(range(want)))
    else:
        raise AssertionError(f"Unexpected action shape: {action_shape}")

    # Example: if robot_state ended up >7 dims, keep first 7
    if "robot_state" in rb and rb["robot_state"].shape[-1] > 7:
        _zarr_slice_last_dim_(rb["robot_state"], list(range(7)))

    return rb


def _zarr_slice_last_dim_(zarr_arr, idxs: List[int]):
    arr = zarr_arr[:]
    arr = arr[..., idxs]
    zarr_arr.resize(zarr_arr.shape[:-1] + (len(idxs),))
    zarr_arr[:] = arr
    return zarr_arr


class RealXArm6LowdimDataset(BaseImageDataset):
    """
    Low-dim dataset for Diffusion Policy on xArm6 (no images).
    Features:
      - zipped zarr caching (separate fingerprint from image dataset)
      - optional delta-action (with gripper include/exclude)
      - episode-level train/val split
      - latency support
    """
    def __init__(
        self,
        shape_meta: dict,
        dataset_path: str,
        out_dir: Optional[str] = None,          # where to store cache (defaults to dataset_path)
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
    ) -> None:
        assert os.path.isdir(dataset_path), f"dataset_path not found: {dataset_path}"

        # Parse low-dim keys from shape_meta (ignore any 'rgb' entries)
        obs_sm = shape_meta["obs"]
        lowdim_keys, lowdim_shapes = [], {}
        for k, a in obs_sm.items():
            if a.get("type", "low_dim") == "low_dim":
                lowdim_keys.append(k)
                lowdim_shapes[k] = tuple(a.get("shape"))
        action_shape = tuple(shape_meta["action"]["shape"])

        # Build or load cache
        rb: Optional[ReplayBuffer] = None
        if use_cache:
            fp = {
                "kind": "lowdim_only",
                "shape_meta": OmegaConf.to_container(shape_meta),
            }
            fp_json = json.dumps(fp, sort_keys=True)
            fp_hash = hashlib.md5(fp_json.encode("utf-8")).hexdigest()
            target_dir = out_dir if out_dir is not None else dataset_path
            cache_path = os.path.join(target_dir, f"{fp_hash}.zarr.zip")
            lock_path = cache_path + ".lock"
            with FileLock(lock_path):
                if not os.path.exists(cache_path):
                    try:
                        rb = _build_lowdim_replay_buffer(
                            dataset_path=dataset_path,
                            lowdim_keys=lowdim_keys,
                            action_shape=action_shape,
                            store=zarr.MemoryStore(),
                        )
                        with zarr.ZipStore(cache_path, mode="w") as zs:
                            rb.save_to_store(store=zs)
                    except Exception:
                        if os.path.exists(cache_path):
                            shutil.rmtree(cache_path, ignore_errors=True)
                        raise
                else:
                    with zarr.ZipStore(cache_path, mode="r") as zs:
                        rb = ReplayBuffer.copy_from_store(src_store=zs, store=zarr.MemoryStore())
        else:
            rb = _build_lowdim_replay_buffer(
                dataset_path=dataset_path,
                lowdim_keys=lowdim_keys,
                action_shape=action_shape,
                store=zarr.MemoryStore(),
            )
        assert rb is not None, "ReplayBuffer creation failed."

        # Optional: convert to delta actions (default excludes gripper index 6)
        if delta_action:
            acts = rb["action"][:]
            n_dim = acts.shape[1]
            episode_ends = rb.episode_ends[:]
            acts_diff = np.zeros_like(acts)
            if delta_exclude_gripper and n_dim == 7:
                delta_idxs = list(range(6))
            else:
                delta_idxs = list(range(n_dim))
            for epi, end in enumerate(episode_ends):
                start = 0 if epi == 0 else episode_ends[epi - 1]
                seg = acts[start:end]
                d = np.diff(seg, axis=0, prepend=seg[0:1])
                acts_diff[start:end, delta_idxs] = d[:, delta_idxs]
                if delta_exclude_gripper and n_dim == 7:
                    acts_diff[start:end, 6] = seg[:, 6]
            rb["action"][:] = acts_diff

        # Build sampler (episode-level split)
        key_first_k = {}
        if n_obs_steps is not None:
            for k in lowdim_keys:
                key_first_k[k] = n_obs_steps

        val_mask = get_val_mask(n_episodes=rb.n_episodes, val_ratio=val_ratio, seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)

        sampler = SequenceSampler(
            replay_buffer=rb,
            sequence_length=horizon + n_latency_steps,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k,
        )

        # Expose
        self.replay_buffer = rb
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.lowdim_keys = lowdim_keys
        self.n_obs_steps = n_obs_steps
        self.val_mask = val_mask
        self.horizon = horizon
        self.n_latency_steps = n_latency_steps
        self.pad_before = pad_before
        self.pad_after = pad_after

    # ---------- Public API ----------
    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon + self.n_latency_steps,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.val_mask,
        )
        val_set.val_mask = ~self.val_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        norm = LinearNormalizer()
        norm["action"] = SingleFieldLinearNormalizer.create_fit(self.replay_buffer["action"])
        for k in self.lowdim_keys:
            norm[k] = SingleFieldLinearNormalizer.create_fit(self.replay_buffer[k])
        return norm

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)

        # Only keep first n_obs_steps of obs (if set)
        T = slice(self.n_obs_steps)

        obs = {}
        for k in self.lowdim_keys:
            obs[k] = data[k][T].astype(np.float32)
            del data[k]

        act = data["action"].astype(np.float32)
        if self.n_latency_steps > 0:
            act = act[self.n_latency_steps:]

        return {
            "obs": dict_apply(obs, torch.from_numpy),
            "action": torch.from_numpy(act),
        }
