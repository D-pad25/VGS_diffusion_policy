from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer

class XArm6ImageDataset(BaseImageDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None
            ):
        
        super().__init__()
        # Using your exact data structure: 'state', 'base_rgb', 'wrist_rgb', 'action'
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['state', 'base_rgb', 'wrist_rgb', 'action'])
        
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'],
            'state': self.replay_buffer['state']
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['image'] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        # Convert your data structure to the format expected by diffusion policy
        state = sample['state'].astype(np.float32)  # T, 7 (6 joints + 1 gripper)
        
        # Handle images - convert to (T, C, H, W) format and downsample to 224x224
        base_image = np.moveaxis(sample['base_rgb'], -1, 1) / 255.0  # T, 3, 480, 640
        wrist_image = np.moveaxis(sample['wrist_rgb'], -1, 1) / 255.0  # T, 3, 480, 640
        
        # Downsample to 224x224 (you'll implement this in your pipeline)
        # For now, we'll assume images are already 224x224 or will be resized later
        # You can add resizing here if needed: base_image = resize_images(base_image, (224, 224))
        
        # Combine images (no right wrist image in your setup)
        all_images = np.concatenate([base_image, wrist_image], axis=1)  # T, 6, 224, 224 (after downsampling)
        
        data = {
            'obs': {
                'image': all_images,  # T, 6, 224, 224 (3 base + 3 wrist, downsampled)
                'state': state,  # T, 7 (6 joints + 1 gripper)
            },
            'action': sample['action'].astype(np.float32),  # T, 7 (6 joints + 1 gripper)
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data


def test():
    import os
    # Test the dataset with a sample zarr path
    zarr_path = os.path.expanduser('~/data/xarm6/xarm6_replay.zarr')
    if os.path.exists(zarr_path):
        dataset = XArm6ImageDataset(zarr_path, horizon=16)
        print(f"Dataset length: {len(dataset)}")
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Sample keys: {sample.keys()}")
            print(f"Image shape: {sample['obs']['image'].shape}")
            print(f"State shape: {sample['obs']['state'].shape}")
            print(f"Action shape: {sample['action'].shape}")
    else:
        print(f"Test zarr path not found: {zarr_path}")
        print("Please update the zarr_path in the test function")


if __name__ == "__main__":
    test() 