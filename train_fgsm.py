import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet18

# FGSM adversarial training (mixed: clean + adversarial)
# Leaderboard score = 0.5 * clean_acc + 0.5 * robust_acc
# We train on both clean and adversarial examples to preserve clean accuracy
# while gaining robustness. Epsilon of 8/255 is the standard threat model.

EPSILON = 8 / 255
EPOCHS = 50
LR = 0.01
NUM_CLASSES = 9

data = np.load("/home/atml_team006/robustness/tml26_task3/train.npz", allow_pickle=True)
images = torch.from_numpy(data["images"]).float() / 255.0
labels = torch.from_numpy(data["labels"]).long()

dataset = TensorDataset(images, labels)

VAL_SIZE = 5000
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [len(dataset) - VAL_SIZE, VAL_SIZE],
    generator=torch.Generator().manual_seed(42)
)
loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

print(f"Train size: {len(train_dataset)}  Val size: {len(val_dataset)}")
print("Epsilon:", EPSILON, f"({round(EPSILON*255)}/255)")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss()


def fgsm_attack(model, imgs, lbls, eps):
    """Return FGSM adversarial examples for a batch."""
    delta = torch.empty_like(imgs).uniform_(-eps, eps)  # random start in [-ε, ε]
    imgs = (imgs + delta).clamp(0, 1).clone().detach().requires_grad_(True)
    # imgs = imgs.clone().detach().requires_grad_(True)
    loss = criterion(model(imgs), lbls)
    loss.backward()
    return (imgs + eps * imgs.grad.sign()).clamp(0, 1).detach()


def visualize_perturbations(orig, adv, n=8, save_path="/home/atml_team006/robustness/tml26_task3/fgsm_perturbations.png"):
    """Save a side-by-side grid: original | perturbed | perturbation (×10)."""
    orig = orig[:n].cpu().permute(0, 2, 3, 1).numpy()
    adv = adv[:n].cpu().permute(0, 2, 3, 1).numpy()
    diff = np.clip((adv - orig) * 10 + 0.5, 0, 1)  # amplify for visibility

    fig, axes = plt.subplots(3, n, figsize=(2 * n, 6))
    titles = ["Original", "Perturbed", "Perturbation ×10"]
    for col in range(n):
        for row, (img, title) in enumerate(zip([orig[col], adv[col], diff[col]], titles)):
            axes[row, col].imshow(img)
            axes[row, col].axis("off")
            if col == 0:
                axes[row, col].set_title(title, fontsize=10, loc="left")

    plt.suptitle(f"FGSM perturbations (ε={round(EPSILON*255)}/255)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved perturbation visualization to {save_path}")


# Visualize before training using first batch
first_imgs, first_lbls = next(iter(loader))
first_imgs, first_lbls = first_imgs.to(device), first_lbls.to(device)
model.eval()
with torch.enable_grad():
    first_adv = fgsm_attack(model, first_imgs, first_lbls, EPSILON)
visualize_perturbations(first_imgs, first_adv)


for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss, correct_clean, correct_adv, total = 0.0, 0, 0, 0

    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)

        # generate adversarial examples
        adv_imgs = fgsm_attack(model, imgs, lbls, EPSILON)

        # train on both clean and adversarial examples (mixed FGSM training)
        optimizer.zero_grad()
        out_clean = model(imgs)
        out_adv = model(adv_imgs)
        loss = 0.5 * criterion(out_clean, lbls) + 0.5 * criterion(out_adv, lbls)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            correct_clean += (out_clean.argmax(1) == lbls).sum().item()
            correct_adv += (out_adv.argmax(1) == lbls).sum().item()
            total_loss += loss.item() * lbls.size(0)
            total += lbls.size(0)

    scheduler.step()

    # validation
    model.eval()
    val_correct_clean, val_correct_adv, val_total = 0, 0, 0
    for imgs, lbls in val_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        with torch.no_grad():
            val_correct_clean += (model(imgs).argmax(1) == lbls).sum().item()
        with torch.enable_grad():
            adv_imgs = fgsm_attack(model, imgs, lbls, EPSILON)
        with torch.no_grad():
            val_correct_adv += (model(adv_imgs).argmax(1) == lbls).sum().item()
        val_total += lbls.size(0)

    print(
        f"Epoch {epoch}/{EPOCHS}  "
        f"loss={total_loss/total:.4f}  "
        f"train_clean={correct_clean/total:.4f}  "
        f"train_adv={correct_adv/total:.4f}  "
        f"val_clean={val_correct_clean/val_total:.4f}  "
        f"val_adv={val_correct_adv/val_total:.4f}"
    )

torch.save(model.state_dict(), "/home/atml_team006/robustness/tml26_task3/model_fgsm.pt")
print("Saved model_fgsm.pt")
