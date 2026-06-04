from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class TinyNet(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def main(out_dir: str, epochs: int = 5, seed: int = 7):
    torch.manual_seed(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    x = torch.randn(256, 3, 32, 32)
    y = torch.randint(0, 4, (256,))
    loader = DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True)
    model = TinyNet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(1, epochs + 1):
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss.item())
        ckpt = {
            "epoch": epoch,
            "loss": total / len(loader),
            "state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
        }
        path = out / f"epoch_{epoch:03d}.pth"
        torch.save(ckpt, path)
        print(f"Saved {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create demo PyTorch checkpoints for Checkpoint Training Inspector.")
    parser.add_argument("--out_dir", default="sample_training_checkpoints")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    main(args.out_dir, args.epochs, args.seed)
