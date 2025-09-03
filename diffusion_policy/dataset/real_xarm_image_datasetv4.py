# diffusion_policy/dataset/real_xarm_image_dataset.py
import os, pickle, json, hashlib, shutil
import numpy as np
import cv2
from filelock import FileLock
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask, downsample_mask
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, Jpeg2k

register_codecs()  # enable custom compression codecs (JPEG2000)

class RealXArmImageDataset(BaseImageDataset):
    def __init__(self, 
                 shape_meta: dict,
                 dataset_path: str,
                 horizon: int = 1,
                 pad_before: int = 0,
                 pad_after: int = 0,
                 n_obs_steps: int = None,
                 n_latency_steps: int = 0,
                 use_cache: bool = False,
                 seed: int = 42,
                 val_ratio: float = 0.0,
                 max_train_episodes: int = None,
                 delta_action: bool = False):
        assert os.path.isdir(dataset_path), f"Dataset path {dataset_path} not found"
        # If caching is enabled, attempt to load or create a cached zarr dataset
        replay_buffer = None
        if use_cache:
            # Unique hash based on shape_meta (to rebuild cache if config changes)
            shape_meta_json = json.dumps(shape_meta, sort_keys=True)
            shape_meta_hash = hashlib.md5(shape_meta_json.encode('utf-8')).hexdigest()
            cache_path = os.path.join(dataset_path, f"{shape_meta_hash}.zarr.zip")
            cache_lock = cache_path + '.lock'
            print("Cache file:", cache_path)
            with FileLock(cache_lock):
                if not os.path.exists(cache_path):
                    try:
                        print("No cache found. Building dataset from raw files...")
                        replay_buffer = self._build_replay_buffer(dataset_path, shape_meta)
                        print("Saving dataset to cache...")
                        # Save replay_buffer to a compressed .zarr.zip file
                        with ReplayBuffer.create_empty_zarr() as tmp_rb:
                            # Actually, save via zarr utility for compression
                            with open(cache_path, 'wb') as f:
                                # Use Zarr ZipStore to save compressed
                                import zarr
                                with zarr.ZipStore(f.name, mode='w') as store:
                                    replay_buffer.save_to_store(store=store)
                        print("Dataset cache saved.")
                    except Exception as e:
                        # Cleanup partial cache if failure
                        if os.path.exists(cache_path):
                            shutil.rmtree(cache_path, ignore_errors=True)
                        raise e
                else:
                    print("Loading dataset from cache...")
                    import zarr
                    # Load the cached replay buffer entirely into memory
                    with zarr.ZipStore(cache_path, mode='r') as store:
                        replay_buffer = ReplayBuffer.copy_from_store(src_store=store, store=zarr.MemoryStore())
                    print("Cache loaded.")
        # If not using cache, or cache load failed, build directly
        if replay_buffer is None:
            replay_buffer = self._build_replay_buffer(dataset_path, shape_meta)

        # If requested, convert actions to deltas (not typical for joint angles, so usually False)
        if delta_action:
            raise ValueError("delta_action not supported for multi-DoF joint actions in this dataset")

        # Determine which keys are image vs low-dim from shape_meta
        obs_meta = shape_meta['obs']
        self.rgb_keys = [k for k,v in obs_meta.items() if v.get('type','low_dim') == 'rgb']
        self.lowdim_keys = [k for k,v in obs_meta.items() if v.get('type','low_dim') == 'low_dim']

        # Create train/val split mask for episodes
        total_eps = replay_buffer.n_episodes
        val_mask = get_val_mask(n_episodes=total_eps, val_ratio=val_ratio, seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(train_mask, max_n=max_train_episodes, seed=seed)  # limit episodes if specified

        # Create a sequence sampler for training data sequences
        sequence_length = horizon + n_latency_steps
        sampler = SequenceSampler(replay_buffer=replay_buffer,
                                  sequence_length=sequence_length,
                                  pad_before=pad_before,
                                  pad_after=pad_after,
                                  episode_mask=train_mask)
        
        # Store attributes for use in other methods
        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.n_obs_steps = n_obs_steps
        self.n_latency_steps = n_latency_steps
        self.val_mask = val_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def _build_replay_buffer(self, dataset_path: str, shape_meta: dict) -> ReplayBuffer:
        """Load all episodes from the dataset directory into a ReplayBuffer."""
        # Create an empty in-memory replay buffer
        replay_buffer = ReplayBuffer.create_empty_numpy()  # using numpy backend for simplicity
        # Optional: prepare compressors and chunk sizes for image data
        compressors = {
            # Use JPEG2000 compression for images to save memory 
            'wrist_rgb': Jpeg2k(level=50),
            'base_rgb': Jpeg2k(level=50)
        }
        chunks = {
            # Chunk images by time step (one frame per chunk) for efficient compression
            'wrist_rgb': (1, shape_meta['obs']['wrist_rgb']['shape'][1], shape_meta['obs']['wrist_rgb']['shape'][2], 3),
            'base_rgb':  (1, shape_meta['obs']['base_rgb']['shape'][1],  shape_meta['obs']['base_rgb']['shape'][2],  3)
        }
        # Iterate over episodes in sorted order
        episode_dirs = sorted([d for d in os.listdir(dataset_path) if d.startswith("episode")])
        for ep in episode_dirs:
            ep_path = os.path.join(dataset_path, ep)
            if not os.path.isdir(ep_path):
                continue
            # Lists to collect data for this episode
            obs_joint_list = []
            obs_wrist_list = []
            obs_base_list = []
            action_list = []
            # Iterate over step files in order
            step_files = sorted([f for f in os.listdir(ep_path) if f.endswith(".pkl")], key=lambda x: int(x.replace("step", "").replace(".pkl","")))
            for step_file in step_files:
                step_path = os.path.join(ep_path, step_file)
                with open(step_path, 'rb') as f:
                    data = pickle.load(f)
                # Extract relevant fields
                joint_pos = np.array(data["joint_positions"], dtype=np.float32)   # shape (7,)
                # (If "joint_positions" already includes gripper, it's our full 7-D state)
                obs_joint_list.append(joint_pos)
                # Resize images to 224x224
                wrist_img = cv2.resize(data["wrist_rgb"], (224, 224), interpolation=cv2.INTER_AREA)
                base_img  = cv2.resize(data["base_rgb"],  (224, 224), interpolation=cv2.INTER_AREA)
                obs_wrist_list.append(wrist_img.astype(np.uint8))
                obs_base_list.append(base_img.astype(np.uint8))
                # Action (7-D) as float32
                action = np.array(data["control"], dtype=np.float32)  # shape (7,)
                action_list.append(action)
            # Stack episode data into numpy arrays
            obs_joint_arr = np.stack(obs_joint_list, axis=0)         # shape (T, 7)
            obs_wrist_arr = np.stack(obs_wrist_list, axis=0)         # shape (T, 224, 224, 3)
            obs_base_arr  = np.stack(obs_base_list, axis=0)          # shape (T, 224, 224, 3)
            action_arr    = np.stack(action_list, axis=0)            # shape (T, 7)
            # Add this episode to the replay buffer
            episode_data = {
                'joint_positions': obs_joint_arr,
                'wrist_rgb': obs_wrist_arr,
                'base_rgb': obs_base_arr,
                'action': action_arr
            }
            replay_buffer.add_episode(episode_data, chunks=chunks, compressors=compressors)
        print(f"Loaded {replay_buffer.n_episodes} episodes with {replay_buffer.n_steps} total steps.")
        return replay_buffer

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int):
        # Sample a sequence (trajectory snippet) from the replay buffer
        data = self.sampler.sample_sequence(idx)
        # Determine slice for observation frames (usually the first n_obs_steps frames)
        T_slice = slice(None) if self.n_obs_steps is None else slice(0, self.n_obs_steps)
        # Prepare obs dict
        obs_dict = {}
        for key in self.rgb_keys:
            # Convert images to float32 in [0,1] and channel-first (C×H×W)
            img_seq = data[key][T_slice].astype(np.float32) / 255.0  # (To, H, W, 3) -> values [0,1]
            obs_dict[key] = np.moveaxis(img_seq, -1, 1)              # -> (To, 3, H, W)
            del data[key]
        for key in self.lowdim_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)    # (To, D) low-dim observations
            del data[key]
        # Get action sequence (Ta steps). Apply latency offset if any.
        action_seq = data['action'].astype(np.float32)               # (T, Da)
        if self.n_latency_steps > 0:
            action_seq = action_seq[self.n_latency_steps:]
        # Wrap into torch Tensors and return
        import torch
        from diffusion_policy.common.pytorch_util import dict_apply
        obs_tensor = dict_apply(obs_dict, lambda x: torch.from_numpy(x))
        action_tensor = torch.from_numpy(action_seq)
        return {"obs": obs_tensor, "action": action_tensor}

    def get_normalizer(self):
        """Compute normalizers for observations and actions (for normalization in the policy)."""
        normalizer = LinearNormalizer()
        # Fit normalizer for actions (compute mean/std for each dimension)
        normalizer["action"] = SingleFieldLinearNormalizer.create_fit(self.replay_buffer["action"])
        # Fit normalizer for low-dim observations
        for key in self.lowdim_keys:
            normalizer[key] = SingleFieldLinearNormalizer.create_fit(self.replay_buffer[key])
        # Use image range normalizer (0-1) for image observations (they are already scaled 0-1 in dataset)
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()  # [0,1] range
        return normalizer

    def get_validation_dataset(self):
        """Create a validation dataset object with the validation split episodes."""
        import copy
        val_set = copy.copy(self)
        # Use SequenceSampler on the validation episodes (opposite mask)
        val_set.sampler = SequenceSampler(replay_buffer=self.replay_buffer,
                                         sequence_length=self.horizon + self.n_latency_steps,
                                         pad_before=self.pad_before,
                                         pad_after=self.pad_after,
                                         episode_mask= self.val_mask )
        val_set.val_mask = ~self.val_mask  # invert mask for completeness
        return val_set
