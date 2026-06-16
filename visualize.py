import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

data = np.load("/home/atml_team006/robustness/tml26_task3/train.npz", allow_pickle=True)
images = torch.from_numpy(data["images"]).float() / 255.0
labels = torch.from_numpy(data["labels"]).long()

print("Dataset size   :", len(images))
print("Image shape    :", images.shape)
print("Pixel range    : [{:.3f}, {:.3f}]".format(images.min().item(), images.max().item()))

NUM_CLASSES = 9
SAMPLES_PER_CLASS = 10

# ── Figure 1: class overview ─────────────────────────────────────────────────
# count how many examples exist per class
for c in range(NUM_CLASSES):
    n = (labels == c).sum().item()
    print(f"  class {c}: {n} images")

fig1, axes1 = plt.subplots(NUM_CLASSES, SAMPLES_PER_CLASS,
                            figsize=(SAMPLES_PER_CLASS * 1.5, NUM_CLASSES * 1.5))
fig1.suptitle("Dataset overview — 10 samples per class\n"
              "(row label = class index; actual class names unknown until dataset is annotated)",
              fontsize=11)

for c in range(NUM_CLASSES):
    idxs = (labels == c).nonzero(as_tuple=True)[0][:SAMPLES_PER_CLASS]
    for col, idx in enumerate(idxs):
        ax = axes1[c, col]
        ax.imshow(images[idx].permute(1, 2, 0).numpy())
        ax.axis("off")
        if col == 0:
            ax.set_ylabel(f"class {c}", fontsize=9, rotation=0,
                          labelpad=40, va="center")

plt.tight_layout()
out1 = "/home/atml_team006/robustness/tml26_task3/visualize_classes.png"
plt.savefig(out1, dpi=120, bbox_inches="tight")
print("Saved:", out1)

# ── Figure 2: epsilon perturbation reference ─────────────────────────────────
# pick one representative image per class
samples = torch.stack([
    images[(labels == c).nonzero(as_tuple=True)[0][0]]
    for c in range(NUM_CLASSES)
])

epsilons   = [0, 4/255, 8/255, 12/255, 16/255, 32/255]
eps_labels = ["original", "4/255", "8/255", "12/255", "16/255", "32/255"]

def uniform_perturb(img, eps):
    noise = torch.rand_like(img) * 2 - 1   # uniform in [-1, 1]
    return (img + eps * noise).clamp(0, 1)

fig2, axes2 = plt.subplots(NUM_CLASSES, len(epsilons),
                            figsize=(len(epsilons) * 2, NUM_CLASSES * 2))
fig2.suptitle("Perturbation magnitude reference (uniform noise scaled by epsilon)\n"
              "Use this to decide a visually imperceptible epsilon for FGSM",
              fontsize=11)

for row, (img, c) in enumerate(zip(samples, range(NUM_CLASSES))):
    for col, eps in enumerate(epsilons):
        disp = img if eps == 0 else uniform_perturb(img, eps)
        axes2[row, col].imshow(disp.permute(1, 2, 0).numpy())
        axes2[row, col].axis("off")
        if row == 0:
            axes2[row, col].set_title(eps_labels[col], fontsize=9)
        if col == 0:
            axes2[row, col].set_ylabel(f"class {c}", fontsize=9, rotation=0,
                                       labelpad=40, va="center")

plt.tight_layout()
out2 = "/home/atml_team006/robustness/tml26_task3/visualize_epsilon.png"
plt.savefig(out2, dpi=120, bbox_inches="tight")
print("Saved:", out2)
