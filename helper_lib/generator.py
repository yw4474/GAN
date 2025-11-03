# helper_lib/generator.py
from __future__ import annotations
import torch
import matplotlib.pyplot as plt

def _denorm(x: torch.Tensor) -> torch.Tensor:
    # x in [-1,1] -> [0,1] for display
    return (x + 1.0) * 0.5

def generate_samples(gan, device: str = "cpu", num_samples: int = 16, nrow: int = 4):
    """
    Sample 'num_samples' images from GAN.generator and show a grid.
    """
    gan.to(device)
    with torch.no_grad():
        imgs = gan.sample(num_samples, device=device).cpu()
        imgs = _denorm(imgs).clamp(0, 1)
    # make a simple grid
    fig, axes = plt.subplots(nrow, nrow, figsize=(nrow*2, nrow*2))
    idx = 0
    for r in range(nrow):
        for c in range(nrow):
            ax = axes[r, c]
            ax.imshow(imgs[idx, 0].numpy(), cmap="gray")
            ax.axis("off")
            idx += 1
    plt.tight_layout()
    plt.show()
