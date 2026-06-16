import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib-cache"

import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

# ----------------------------------------------------------------------
# Load one sample image from your dataset.
# Adjust the path if needed; falls back to a synthetic image if missing.
# ----------------------------------------------------------------------
DATA_PATH = "/scratch/bjoshi/robustness/train.npz"

if os.path.exists(DATA_PATH):
    data = np.load(DATA_PATH, allow_pickle=True)
    images = torch.from_numpy(data["images"]).float() / 255.0   # (N, C, H, W)
    img = images[0]                                             # take first image
else:
    # Synthetic 32x32 image with an arrow-like marker so rotation is obvious
    img = torch.zeros(3, 32, 32)
    img[0, 4:8, 4:28] = 1.0       # red horizontal bar near the top
    img[1, 4:28, 4:8] = 1.0       # green vertical bar on the left
    img[2, 24:28, 24:28] = 1.0    # blue corner block (orientation marker)

# torchvision expects (C, H, W); helper to make it matplotlib-friendly (H, W, C)
def to_disp(t):
    return t.clamp(0, 1).permute(1, 2, 0).cpu().numpy()

# ----------------------------------------------------------------------
# Two interpretations of "90 rotation":
#   A) transforms.RandomRotation(90)  -> random angle in [-90, 90] (continuous!)
#   B) discrete fixed rotations 0/90/180/270 (true square-symmetry rotations)
# ----------------------------------------------------------------------

# A) Several draws from RandomRotation(90) to show it's a CONTINUOUS random angle
rand_rot = transforms.RandomRotation(90)
torch.manual_seed(0)
random_samples = [rand_rot(img) for _ in range(4)]

# B) Discrete exact rotations using fixed-range RandomRotation
discrete = {
    "0°":   transforms.RandomRotation((0, 0))(img),
    "90°":  transforms.RandomRotation((90, 90))(img),
    "180°": transforms.RandomRotation((180, 180))(img),
    "270°": transforms.RandomRotation((270, 270))(img),
}

# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------
fig, axes = plt.subplots(2, 5, figsize=(15, 6))

# Row 1: original + 4 random draws from RandomRotation(90)
axes[0, 0].imshow(to_disp(img))
axes[0, 0].set_title("Original")
for i, s in enumerate(random_samples):
    axes[0, i + 1].imshow(to_disp(s))
    axes[0, i + 1].set_title(f"RandomRotation(90)\ndraw {i+1}")
for ax in axes[0]:
    ax.axis("off")

# Row 2: original + the 4 discrete exact rotations
axes[1, 0].imshow(to_disp(img))
axes[1, 0].set_title("Original")
for i, (name, s) in enumerate(discrete.items()):
    axes[1, i + 1].imshow(to_disp(s))
    axes[1, i + 1].set_title(f"Discrete {name}")
for ax in axes[1]:
    ax.axis("off")

fig.suptitle(
    "Top: RandomRotation(90) = random angle in [-90°,90°] (note interpolation + black corners)\n"
    "Bottom: exact 0/90/180/270 rotations (no artifacts)",
    fontsize=11,
)
plt.tight_layout()
out_path = "/scratch/bjoshi/robustness/rotation_augmentation.png"
plt.savefig(out_path, dpi=120, bbox_inches="tight")
print(f"Saved visualization to {out_path}")