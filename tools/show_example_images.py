import os
import random
import matplotlib.pyplot as plt
import zarr

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs

# register JPEG2000 and other codecs
register_codecs()

def show_example_images(folder_path=".", idx=None):
    """
    Show example wrist and base images from a converted ReplayBuffer.

    Args:
        folder_path (str): Path to folder containing the .zarr.zip dataset.
        idx (int, optional): Index of frame to display. If None, selects random frame.
    """
    # find first zarr.zip file in folder
    zarr_files = [f for f in os.listdir(folder_path) if f.endswith(".zarr.zip")]
    if not zarr_files:
        raise FileNotFoundError(f"No .zarr.zip files found in {folder_path}")
    zarr_path = os.path.join(folder_path, zarr_files[0])
    print(f"Loading dataset: {zarr_path}")

    # load replay buffer
    with zarr.ZipStore(zarr_path, mode="r") as store:
        rb = ReplayBuffer.copy_from_store(store, store=zarr.MemoryStore())

    # pick random index if not given
    T = rb["base_rgb"].shape[0]
    if idx is None:
        idx = random.randint(0, T - 1)

    wrist_img = rb["wrist_rgb"][idx]
    base_img = rb["base_rgb"][idx]

    # plot images
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(wrist_img)
    axes[0].set_title(f"Wrist RGB (idx={idx})")
    axes[0].axis("off")

    axes[1].imshow(base_img)
    axes[1].set_title(f"Base RGB (idx={idx})")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

# Example usage
show_example_images("/home/d_pad25/Thesis/Data/diffusion_test/test_pad")
