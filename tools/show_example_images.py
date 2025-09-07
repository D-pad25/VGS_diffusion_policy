import random
import matplotlib.pyplot as plt
import zarr
from diffusion_policy.common.replay_buffer import ReplayBuffer

def show_example_images(zarr_path: str, idx: int = None):
    """
    Show example wrist and base images from a converted ReplayBuffer.

    Args:
        zarr_path (str): Path to your .zarr.zip file (converted dataset).
        idx (int, optional): Index of the frame to show. If None, selects random frame.
    """
    # Load replay buffer
    with zarr.ZipStore(zarr_path, mode='r') as store:
        rb = ReplayBuffer.copy_from_store(store, store=zarr.MemoryStore())
    
    T = rb['base_rgb'].shape[0]
    if idx is None:
        idx = random.randint(0, T - 1)
    
    wrist_img = rb['wrist_rgb'][idx]
    base_img = rb['base_rgb'][idx]

    # Plot side by side
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(wrist_img)
    axes[0].set_title(f"Wrist RGB (idx={idx})")
    axes[0].axis("off")

    axes[1].imshow(base_img)
    axes[1].set_title(f"Base RGB (idx={idx})")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()
