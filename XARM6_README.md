# XArm6 Diffusion Policy Implementation

This directory contains the implementation of diffusion policy training for xArm6 robots, designed to follow the same data structure as your pi0 implementation.

## Overview

The implementation consists of several key components:

1. **Dataset Class** (`diffusion_policy/dataset/xarm6_image_dataset.py`) - Handles your xArm6 data structure
2. **Task Configuration** (`diffusion_policy/config/task/xarm6_image.yaml`) - Defines data structure and dataset config
3. **Environment Runner** (`diffusion_policy/env_runner/xarm6_image_runner.py`) - Handles evaluation and visualization
4. **Training Configuration** (`diffusion_policy/config/train_xarm6_diffusion_unet_image_workspace.yaml`) - Training parameters
5. **Training Script** (`train_xarm6.py`) - Main training entry point
6. **Data Conversion Script** (`convert_xarm6_data.py`) - Converts data to zarr format

## Data Structure

The implementation expects your data to follow this exact structure (matching your xArm6 environment):

```python
{
    "state": np.array,      # Shape: (T, 7) - 6 joints + 1 gripper
    "base_rgb": np.array,   # Shape: (T, 224, 224, 3) - Base camera images (downsampled)
    "wrist_rgb": np.array,  # Shape: (T, 224, 224, 3) - Wrist camera images (downsampled)
    "action": np.array,     # Shape: (T, 7) - 6 joints + 1 gripper
}
```

## Quick Start

### 1. Prepare Your Data

**Important**: This implementation expects images to be downsampled to 224x224 before training. You'll need to implement downsampling in your data pipeline.

First, convert your xArm6 data to the zarr format:

```bash
# Create sample data for testing
python convert_xarm6_data.py --sample

# Or convert your existing data
python convert_xarm6_data.py --input_data path/to/your/data.json --output data/xarm6/xarm6_replay.zarr
```

### 2. Update Configuration

Edit `diffusion_policy/config/task/xarm6_image.yaml` and update the `zarr_path` to point to your actual data:

```yaml
dataset:
  zarr_path: data/xarm6/xarm6_replay.zarr  # Update this path
```

### 3. Start Training

```bash
python train_xarm6.py
```

## Key Features

### Data Handling
- **Image Processing**: Automatically converts your (H, W, C) images to (C, H, W) format
- **Image Concatenation**: Combines base and wrist images into a single 6-channel tensor
- **State/Action Alignment**: Maintains your 7D state and action dimensions (6 joints + 1 gripper)
- **Camera Integration**: Uses your exact camera keys: `base_rgb` and `wrist_rgb`

### Training Configuration
- **Batch Size**: Set to 32 (standard for 224x224 images)
- **Image Size**: Configured for 224x224 images (after downsampling from 480x640)
- **Horizon**: Default 16 timesteps (configurable)
- **Model**: Uses ResNet18 encoder with UNet diffusion model

### Environment Integration
- **Placeholder Environment**: Includes dummy environment for testing
- **Easy Integration**: Designed to easily integrate with your actual xArm6 environment
- **Evaluation Support**: Full evaluation and visualization pipeline

## Customization

### Modifying Image Dimensions
If you need different image dimensions, update these files:
- `diffusion_policy/config/task/xarm6_image.yaml` - Update `image_shape`
- `diffusion_policy/dataset/xarm6_image_dataset.py` - Update image processing in `_sample_to_data`

### Adding New Observation Types
To add new observation types (e.g., depth images, force sensors):

1. Update the dataset class to handle new data
2. Modify the shape meta in the task configuration
3. Update the environment runner to provide new observations

### Changing Action Dimensions
If your xArm6 has different action dimensions:

1. Update the `action` shape in `xarm6_image.yaml`
2. Modify the dataset class accordingly
3. Update the environment runner

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running from the VGS_diffusion_policy directory
2. **Data Format**: Verify your data matches the expected structure exactly
3. **Memory Issues**: Reduce batch size if you encounter CUDA out of memory errors
4. **Image Dimensions**: Ensure all images are exactly 224x224x3 (after downsampling)

### Testing

Test individual components:

```bash
# Test dataset
python diffusion_policy/dataset/xarm6_image_dataset.py

# Test environment runner
python diffusion_policy/env_runner/xarm6_image_runner.py

# Test data conversion
python convert_xarm6_data.py --sample

# Test integration with your environment
python integrate_with_xarm6.py
```

## Integration with Your xArm6 Environment

This implementation is designed to work seamlessly with your existing xArm6 setup:

1. **Exact Data Structure**: Uses your exact camera keys (`base_rgb`, `wrist_rgb`)
2. **State/Action Alignment**: Maintains your 7D joint space (6 joints + 1 gripper)
3. **Camera Integration**: Works with your 224x224 downsampled images
4. **Environment Compatibility**: Designed to integrate with your `XArmRealEnv` class

## Next Steps

1. **Data Collection**: Gather xArm6 demonstration data in the specified format
2. **Environment Integration**: Connect the environment runner to your actual xArm6 setup
3. **Hyperparameter Tuning**: Adjust training parameters based on your specific task
4. **Evaluation**: Implement task-specific evaluation metrics

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify your data format matches the expected structure
3. Test individual components using the provided test functions
4. Refer to the original diffusion policy documentation for advanced features

## File Structure

```
VGS_diffusion_policy/
├── diffusion_policy/
│   ├── dataset/
│   │   └── xarm6_image_dataset.py      # Dataset class
│   ├── config/
│   │   ├── task/
│   │   │   └── xarm6_image.yaml        # Task configuration
│   │   └── train_xarm6_diffusion_unet_image_workspace.yaml  # Training config
│   └── env_runner/
│       └── xarm6_image_runner.py       # Environment runner
├── train_xarm6.py                       # Training script
├── convert_xarm6_data.py                # Data conversion script
├── integrate_with_xarm6.py              # Integration example with your environment
└── XARM6_README.md                      # This file
``` 