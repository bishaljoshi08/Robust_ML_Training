import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib-cache"

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet50
import torchvision.transforms as transforms

# --- Hyperparameters ---
EPSILON = 10 / 255
ALPHA = 2 / 255
STEPS = 20
EPOCHS = 130
BATCH_SIZE = 256
LR = 0.1
NUM_CLASSES = 9

best_score = -1.0
checkpoint_path = "/scratch/bjoshi/robustness/pgd_version4_cliping_v2.pt"

# --- Data Loading ---
data = np.load("/scratch/bjoshi/robustness/train.npz", allow_pickle=True)
images = torch.from_numpy(data["images"]).float() / 255.0
labels = torch.from_numpy(data["labels"]).long()

dataset = TensorDataset(images, labels)
VAL_SIZE = 5000
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [len(dataset) - VAL_SIZE, VAL_SIZE],
    generator=torch.Generator().manual_seed(42)
)

loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=2e-4, nesterov=True)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 90], gamma=0.1)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)


augment = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90),
])

# --- PGD Attack (Madry et al. 2018) ---
def pgd_attack(model, imgs, lbls, eps, alpha, steps):
    was_training = model.training
    model.eval()

    imgs = imgs.detach()
    for p in model.parameters():
        p.requires_grad_(False)

    attack_loss_fn = nn.CrossEntropyLoss()

    # Random start within epsilon ball, projected to valid image range
    delta = torch.zeros_like(imgs).uniform_(-eps, eps)
    delta = torch.clamp(imgs + delta, 0, 1) - imgs

    for _ in range(steps):
        delta = delta.detach().requires_grad_(True)

        loss = attack_loss_fn(model(imgs + delta), lbls)
        grad = torch.autograd.grad(loss, delta)[0]

        # Gradient ascent step, project to ε-ball, then to [0, 1]
        delta = delta + alpha * grad.sign()
        delta = torch.clamp(delta, -eps, eps)
        delta = torch.clamp(imgs + delta, 0, 1) - imgs

    for p in model.parameters():
        p.requires_grad_(True)
    model.train(was_training)

    return (imgs + delta).detach()

# --- Training Loop ---
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss, train_clean_hits, train_adv_hits, train_total = 0.0, 0, 0, 0

    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        imgs = augment(imgs)

        # 1. Generate adversarial examples
        # We use model.eval() inside the attack to ensure BatchNorm isn't messed up
        adv_imgs = pgd_attack(model, imgs, lbls, EPSILON, ALPHA, STEPS)

        # 2. Forward pass on both Clean and Adversarial
        model.train()
        combined_imgs = torch.cat([imgs, adv_imgs], dim=0)
        combined_lbls = torch.cat([lbls, lbls], dim=0)

        optimizer.zero_grad()
        out = model(combined_imgs)
        loss = criterion(out, combined_lbls)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 3. Stats calculation (using chunk(2) is much faster than argsort/unshuffle)
        with torch.no_grad():
            out_clean, out_adv = out.chunk(2)
            train_clean_hits += (out_clean.argmax(1) == lbls).sum().item()
            train_adv_hits   += (out_adv.argmax(1) == lbls).sum().item()
            train_loss       += loss.item() * combined_imgs.size(0)
            train_total      += lbls.size(0)

    scheduler.step()

    # --- Validation ---
    model.eval()
    val_clean_hits, val_adv_hits, val_total = 0, 0, 0
    for imgs, lbls in val_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)

        # In validation, we generate PGD without tracking grads for the model
        with torch.enable_grad():
            adv_imgs = pgd_attack(model, imgs, lbls, EPSILON, ALPHA, STEPS)

        with torch.no_grad():
            out_clean = model(imgs)
            out_adv = model(adv_imgs)

            val_clean_hits += (out_clean.argmax(1) == lbls).sum().item()
            val_adv_hits   += (out_adv.argmax(1) == lbls).sum().item()
            val_total      += lbls.size(0)

    # Score calculation (50/50 weighted)
    acc_clean = val_clean_hits / val_total
    acc_adv = val_adv_hits / val_total
    current_score = 0.5 * acc_clean + 0.5 * acc_adv

    print(f"Epoch {epoch:03d} | Loss: {train_loss/(train_total*2):.4f} | "
          f"Clean Acc: {acc_clean:.4f} | Adv Acc: {acc_adv:.4f} | Score: {current_score:.4f}")

    if current_score > best_score:
        best_score = current_score
        torch.save(model.state_dict(), checkpoint_path)
        print(f"--> Best Score Improved! Model saved.")

print(f"\nTraining Complete. Best Score: {best_score:.4f}")