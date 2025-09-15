"""
RealXArm6LowdimDataset
----------------------
Same behavior as RealXArm6ImageDataset, but loads ONLY low-dim fields (e.g., 'robot_state')
and 'action'. No image decoding; caching still uses a zipped zarr with a fingerprint.

Assumptions (same as image variant, minus images):
- Raw episodes are converted via real_data_to_replay_buffer into a ReplayBuffer with keys:
    - 'robot_state' : (T, 7) float32
    - 'action'      : (T, 7) float32
    - 'episode_ends': (E,) int64
- Units: whatever your logger uses (deg/rad); normalizer will scale appropriately.
"""

from typing import Dict, List, Optional
import os, json, hashlib, shutil, copy

import numpy as np
import torch, zarr
from filelock import FileLock
from threadpoolctl import threadpool_limits
from omegaconf import OmegaConf

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.model.common.normalizer import (
    LinearNormalizer, SingleFieldLinearNormalizer
)
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask
)
from diffusion_policy.real_world.real_xarm6_data_conversion import real_data_to_replay_buffer


def _zarr_resize_index_last_dim(zarr_arr, idxs: List[int]):
    """In-place slice on last dimension of a zarr array and resize storage."""
    data = zarr_arr[:]
    data = data[..., idxs]
    zarr_arr.resize(zarr_arr.shape[:-1] + (len(idxs),))
    zarr_arr[:] = data
    return zarr_arr


def _get_replay_buffer_lowdim(
    dataset_path: str,
    shape_meta: dict,
    store: zarr.storage.BaseStore,
) -> ReplayBuffer:
    """Build a ReplayBuffer from real data (low-dim only, no images)."""
    # parse obs keys from shape_meta
    lowdim_keys: List[str] = []
    obs_shape_meta = shape_meta["obs"]
    for key, attr in obs_shape_meta.items():
        if attr.get("type", "low_dim") == "low_dim":
            lowdim_keys.append(key)

    action_shape = tuple(shape_meta["action"]["shape"])
    assert action_shape in [(2,), (6,), (7,)], f"Unexpected action shape: {action_shape}"

    # build from raw episodes (no images)
    with threadpool_limits(1):
        rb = real_data_to_replay_buffer(
            dataset_path=dataset_path,
            out_store=store,
            out_resolutions={},              # no image resize
            lowdim_keys=lowdim_keys + ["action"],
            image_keys=[],                   # critical: skip image decode
        )

    # optional slicing if larger than expected
    if action_shape == (2,):
        _zarr_resize_index_last_dim(rb["action"], [0, 1])
    elif action_shape == (6,):
        _zarr_resize_index_last_dim(rb["action"], [0, 1, 2, 3, 4, 5])

    for key in lowdim_keys:
        shp = tuple(obs_shape_meta[key].get("shape"))
        if rb[key].shape[-1] > shp[-1]:
            _zarr_resize_index_last_dim(rb[key], list(range(shp[-1])))

    return rb


class RealXArm6LowdimDataset(BaseLowdimDataset):
    """
    Low-dim only dataset for Diffusion Policy (mirrors RealXArm6ImageDataset behavior).
    Features:
      - zipped zarr cache keyed by shape_meta (same style)
      - optional delta-action (with option to exclude gripper)
      - episode-level train/val split + sampler
      - latency support
    """
    def __init__(
        self,
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
        zarr_cache_provided: bool = False,
        zarr_cache_path_provided: Optional[str] = None,
    ) -> None:
        assert os.path.isdir(dataset_path) or (
            zarr_cache_provided and zarr_cache_path_provided is not None and os.path.isfile(zarr_cache_path_provided)
        ), f"dataset_path not found: {dataset_path}"
        
        # fingerprint (match style of image dataset; add a 'kind' so caches don't collide)
        if zarr_cache_provided:
            assert zarr_cache_path_provided is not None, "Must provide zarr_cache_path_provided when zarr_cache_provided=True"
            cache_zarr_path = zarr_cache_path_provided
        else:
            fp = {
                "kind": "lowdim_only",
                "shape_meta": OmegaConf.to_container(shape_meta)
            }
            fp_json = json.dumps(fp, sort_keys=True)
            fp_hash = hashlib.md5(fp_json.encode("utf-8")).hexdigest()

            target_dir = out_dir if out_dir is not None else dataset_path
            cache_zarr_path = os.path.join(target_dir, f"{fp_hash}.zarr.zip")
        cache_lock_path = cache_zarr_path + ".lock"

        # Build or load cache (identical flow)
        print(f"[RealXArm6ImageDataset] Acquiring cache lock: {cache_lock_path}")
        replay_buffer: Optional[ReplayBuffer] = None
        if use_cache:
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    try:
                        print("[RealXArm6ImageDataset] Cache miss → building ReplayBuffer in memory...")
                        # replay_buffer = _get_replay_buffer_lowdim(
                        #     dataset_path=dataset_path,
                        #     shape_meta=shape_meta,
                        #     store=zarr.MemoryStore(),
                        # )
                        # print("[RealXArm6ImageDataset] Saving cache to disk...")
                        # with zarr.ZipStore(cache_zarr_path, mode="w") as zip_store:
                        #     replay_buffer.save_to_store(store=zip_store)
                        print("[RealXArm6ImageDataset] Cache saved.")
                    except Exception as e:
                        if os.path.exists(cache_zarr_path):
                            shutil.rmtree(cache_zarr_path, ignore_errors=True)
                        raise e
                else:
                    with zarr.ZipStore(cache_zarr_path, mode="r") as zip_store:
                        print("[RealXArm6ImageDataset] Cache hit → loading ReplayBuffer from disk...")
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=zarr.MemoryStore()
                        )
                        print("[RealXArm6ImageDataset] Cache loaded.")
        else:
            replay_buffer = _get_replay_buffer_lowdim(
                dataset_path=dataset_path,
                shape_meta=shape_meta,
                store=zarr.MemoryStore(),
            )

        assert replay_buffer is not None

        # Optional: convert to delta actions (exclude gripper by default)
        if delta_action:
            actions = replay_buffer["action"][:]
            n_dim = actions.shape[1]
            episode_ends = replay_buffer.episode_ends[:]
            actions_diff = np.zeros_like(actions)
            if delta_exclude_gripper and n_dim == 7:
                delta_idxs = list(range(6))
            else:
                delta_idxs = list(range(n_dim))

            for epi, end in enumerate(episode_ends):
                start = 0 if epi == 0 else episode_ends[epi - 1]
                seg = actions[start:end]
                d = np.diff(seg, axis=0, prepend=seg[0:1])
                actions_diff[start:end, delta_idxs] = d[:, delta_idxs]
                if delta_exclude_gripper and n_dim == 7:
                    actions_diff[start:end, 6] = seg[:, 6]

            replay_buffer["action"][:] = actions_diff

        # parse keys
        lowdim_keys: List[str] = []
        for key, attr in shape_meta["obs"].items():
            if attr.get("type", "low_dim") == "low_dim":
                lowdim_keys.append(key)

        key_first_k = {}
        if n_obs_steps is not None:
            for key in lowdim_keys:
                key_first_k[key] = n_obs_steps

        # Split and sampler (same as image dataset)
        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)

        sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon + n_latency_steps,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k,
        )

        # Expose
        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.lowdim_keys = lowdim_keys
        self.n_obs_steps = n_obs_steps
        self.val_mask = val_mask
        self.horizon = horizon
        self.n_latency_steps = n_latency_steps
        self.pad_before = pad_before
        self.pad_after = pad_after

    # -------------------- Public API --------------------
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
        normalizer = LinearNormalizer()
        normalizer["action"] = SingleFieldLinearNormalizer.create_fit(
            self.replay_buffer["action"]
        )
        for key in self.lowdim_keys:
            normalizer[key] = SingleFieldLinearNormalizer.create_fit(
                self.replay_buffer[key]
            )
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)

        T_slice = slice(self.n_obs_steps)
        obs_dict = {}
        for key in self.lowdim_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            del data[key]

        action = data["action"].astype(np.float32)
        if self.n_latency_steps > 0:
            action = action[self.n_latency_steps:]

        return {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "action": torch.from_numpy(action),
        }
