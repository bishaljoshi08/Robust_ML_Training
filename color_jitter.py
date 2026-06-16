import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib-cache"

import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

NUM_CLASSES = 9
SEED = 42
SAVE_PATH = "/scratch/bjoshi/robustness/colorjitter_compare.png"

# --- Load data ---
data = np.load("/scratch/bjoshi/robustness/train.npz", allow_pickle=True)
images = torch.from_numpy(data["images"]).float() / 255.0   # (N, C, H, W), in [0,1]
labels = torch.from_numpy(data["labels"]).long()

# --- The augmentation under test ---
jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)

# Fix the seed so the jitter sample is reproducible
torch.manual_seed(SEED)

# --- Pick one example per class ---
examples = []
for c in range(NUM_CLASSES):
    idx = (labels == c).nonzero(as_tuple=True)[0]
    if len(idx) == 0:
        continue
    examples.append((c, images[idx[0]]))   # first image of class c

# --- Plot: 2 rows (original / jittered) x NUM_CLASSES columns ---
n = len(examples)
fig, axes = plt.subplots(2, n, figsize=(2 * n, 4.5))
if n == 1:
    axes = axes.reshape(2, 1)

for col, (c, img) in enumerate(examples):
    aug = jitter(img)                       # apply ColorJitter to a single (C,H,W) image

    orig_np = img.permute(1, 2, 0).numpy()
    aug_np = aug.clamp(0, 1).permute(1, 2, 0).numpy()

    axes[0, col].imshow(orig_np)
    axes[0, col].set_title(f"Class {c}", fontsize=10)
    axes[0, col].axis("off")

    axes[1, col].imshow(aug_np)
    axes[1, col].axis("off")

axes[0, 0].set_ylabel("Original", fontsize=11)
axes[1, 0].set_ylabel("ColorJitter", fontsize=11)
# set_ylabel needs the axis visible; re-add a label via text instead since axis is off
axes[0, 0].text(-0.15, 0.5, "Original", transform=axes[0, 0].transAxes,
                rotation=90, va="center", ha="center", fontsize=11)
axes[1, 0].text(-0.15, 0.5, "ColorJitter", transform=axes[1, 0].transAxes,
                rotation=90, va="center", ha="center", fontsize=11)

plt.suptitle("Original vs ColorJitter (brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)",
             fontsize=12)
plt.tight_layout()
plt.savefig(SAVE_PATH, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved comparison to {SAVE_PATH}")