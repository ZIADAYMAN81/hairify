"""Two-phase transfer learning for the face-shape classifier.

    python -m hairify.train --data data/clean --out models/run_name --recipe original
    python -m hairify.train --data data/clean --out models/run_name --recipe regularised

Phase 1 trains only the new head on a frozen ImageNet EfficientNet-B3; phase 2 unfreezes the top five blocks.
`original` reproduces the dissertation recipe (30 + 60 epochs, best validation accuracy kept).
`regularised` adds stronger augmentation (random resized crops, random erasing), mixed precision and early
stopping on validation loss, aimed at the 97.7% train vs 73.7% validation gap of the original model.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from . import CLASSES
from .model import HairifyModel

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

RECIPES = {
    "original": {"epochs_head": 30, "epochs_finetune": 60, "lr_head": 1e-3, "lr_finetune": 3e-5,
                 "label_smoothing": 0.1, "weight_decay": 1e-4, "strong_aug": False, "patience": None},
    "regularised": {"epochs_head": 15, "epochs_finetune": 60, "lr_head": 1e-3, "lr_finetune": 5e-5,
                    "label_smoothing": 0.1, "weight_decay": 5e-2, "strong_aug": True, "patience": 10},
}


def transforms_for(train: bool, strong: bool):
    if not train:
        return transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    t = [transforms.ToPILImage()]
    if strong:
        t += [transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1))]
    t += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(15),
          transforms.ColorJitter(0.15, 0.15, 0.15), transforms.RandomAffine(0, translate=(0.1, 0.1)),
          transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    if strong:
        t += [transforms.RandomErasing(p=0.25, scale=(0.02, 0.12))]
    return transforms.Compose(t)


class FolderDataset(Dataset):
    def __init__(self, root: Path, transform):
        self.items = [(f, i) for i, c in enumerate(CLASSES) for f in sorted((root / c).glob("*.jpg"))]
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        f, y = self.items[idx]
        img = cv2.resize(cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB), (224, 224))
        return self.transform(img), y


def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    train = optimizer is not None
    model.train(train)
    total = correct = 0
    loss_sum = 0.0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=scaler is not None):
                out = model(x)
                loss = criterion(out, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                else:
                    loss.backward(); optimizer.step()
            loss_sum += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)
    return loss_sum / total, correct / total


def unfreeze_top(model: HairifyModel, n_blocks: int = 5):
    for p in model.base.parameters():
        p.requires_grad = False
    for blk in list(model.base.blocks)[-n_blocks:]:
        for p in blk.parameters():
            p.requires_grad = True
    for mod in (model.base.conv_head, model.base.bn2):
        for p in mod.parameters():
            p.requires_grad = True


def train(data: Path, out: Path, recipe: str, seed: int = 42, batch_size: int = 32, device: str | None = None,
          max_epochs: int | None = None, pretrained: bool = True) -> dict:
    cfg = dict(RECIPES[recipe])
    if max_epochs is not None:  # for quick smoke tests
        cfg["epochs_head"] = min(cfg["epochs_head"], max_epochs)
        cfg["epochs_finetune"] = min(cfg["epochs_finetune"], max_epochs)
    seed_everything(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out.mkdir(parents=True, exist_ok=True)
    tr = FolderDataset(data / "train", transforms_for(True, cfg["strong_aug"]))
    va = FolderDataset(data / "val", transforms_for(False, False))
    g = torch.Generator().manual_seed(seed)
    tl = DataLoader(tr, batch_size, shuffle=True, generator=g, num_workers=0, pin_memory=dev.type == "cuda")
    vl = DataLoader(va, batch_size, shuffle=False, num_workers=0)
    counts = np.bincount([y for _, y in tr.items], minlength=len(CLASSES))
    weights = torch.tensor(len(tr) / (len(CLASSES) * np.maximum(counts, 1)), dtype=torch.float32, device=dev)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg["label_smoothing"])
    scaler = torch.cuda.amp.GradScaler() if (cfg["strong_aug"] and dev.type == "cuda") else None

    model = HairifyModel(pretrained=pretrained).to(dev)
    history = []
    best = {"val_acc": -1.0, "val_loss": float("inf"), "epoch": None, "phase": None}
    ckpt = out / "face_shape_classifier.pt"

    def phase(name, epochs, lr):
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=cfg["weight_decay"])
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=6)
        stale = 0
        for ep in range(1, epochs + 1):
            t0 = time.time()
            trl, tra = run_epoch(model, tl, criterion, dev, opt, scaler)
            vll, vla = run_epoch(model, vl, criterion, dev)
            sched.step(vla)
            history.append({"phase": name, "epoch": ep, "train_loss": round(trl, 4), "train_acc": round(tra, 4),
                            "val_loss": round(vll, 4), "val_acc": round(vla, 4), "seconds": round(time.time() - t0, 1)})
            # The original recipe keeps the best validation accuracy; the regularised one also stops early.
            if vla > best["val_acc"]:
                best.update(val_acc=vla, val_loss=vll, epoch=ep, phase=name)
                torch.save(model.state_dict(), ckpt)
                stale = 0
            else:
                stale += 1
            print(json.dumps(history[-1]), flush=True)
            if cfg["patience"] and stale >= cfg["patience"]:
                print(f"early stop in {name} after {ep} epochs", flush=True)
                break

    for p in model.base.parameters():
        p.requires_grad = False
    phase("head", cfg["epochs_head"], cfg["lr_head"])
    model.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=True))
    unfreeze_top(model)
    phase("finetune", cfg["epochs_finetune"], cfg["lr_finetune"])

    summary = {"recipe": recipe, "config": cfg, "seed": seed, "device": str(dev),
               "train_images": len(tr), "val_images": len(va), "best": best, "history": history}
    (out / "training_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--recipe", choices=sorted(RECIPES), default="regularised")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--max-epochs", type=int, default=None)
    args = p.parse_args(argv)
    s = train(Path(args.data), Path(args.out), args.recipe, args.seed, device=args.device, max_epochs=args.max_epochs)
    print(json.dumps(s["best"]))


if __name__ == "__main__":
    main()
