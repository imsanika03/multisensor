"""Partial-feature routing experiment.

Gating question: does routing on f1's *intermediate activations* (a partial
forward pass) carry a learnable harm signal -- unlike the 96px thumbnail probe,
which ranked harm at ~chance (AUC ~0.5)?

For each tap depth in f1 (ResNet50 layer1/2/3) we:
  - extract pooled features at f1's resolution (112px),
  - train a small MLP head to predict per-route harm (balanced BCE + ranking),
  - report held-out route-AUC on thresh_calib, plus the partial-forward FLOPs
    (cost of reaching the tap) as a fraction of full f1.

If AUC >> 0.5 at a cheap tap, cost-accounted routing is worth building.

Run:  python partial_feature_probe.py --cfg cfgs/imagenette.yaml
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset

from cascade.stage import build_cascade
from models.probe import _ranking_loss, _route_auc
from score.harm import compute_harm_labels
from score.routing import count_flops
from utils import ResizedDataset, get_dataset, resize_transform

TAPS = ["layer1", "layer2", "layer3"]
FEAT_DIM = {"layer1": 256, "layer2": 512, "layer3": 1024}


class F1Tap(nn.Module):
    """f1 (ResNet50) forward up to `tap`, then global-average-pooled to a vector."""

    def __init__(self, resnet, tap):
        super().__init__()
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        bodies = {
            "layer1": [resnet.layer1],
            "layer2": [resnet.layer1, resnet.layer2],
            "layer3": [resnet.layer1, resnet.layer2, resnet.layer3],
        }[tap]
        self.body = nn.Sequential(*bodies)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return torch.flatten(self.pool(self.body(self.stem(x))), 1)


class FeatRouter(nn.Module):
    def __init__(self, dim, k, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(hidden, k),
        )

    def forward(self, x):
        return self.net(x)            # harm logits [B, K]


@torch.no_grad()
def extract(tap_model, data, device, bs, nw):
    loader = DataLoader(ResizedDataset(data, resize_transform(112)),
                        batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    tap_model = tap_model.to(device).eval()
    feats = []
    for x, _ in loader:
        feats.append(tap_model(x.to(device)).cpu())
    return torch.cat(feats)


def train_head(feat, harm, dim, k, device, epochs=200, lr=1e-3, wd=1e-3):
    """Train the MLP harm head; select the epoch with best internal-val route-AUC."""
    n = feat.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    nv = int(0.25 * n)
    vi, ti = perm[:nv], perm[nv:]
    ftr, htr = feat[ti].to(device), harm[ti].to(device)
    fva, hva = feat[vi].to(device), harm[vi].to(device)

    pos_w = (htr.sum(0) / (len(ti) - htr.sum(0)).clamp(min=1)).to(device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    head = FeatRouter(dim, k).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)

    best_auc, best_state = -1.0, None
    for _ in range(epochs):
        head.train()
        opt.zero_grad()
        logit = head(ftr)
        loss = bce(logit, htr) + _ranking_loss(logit, htr)
        loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            auc = _route_auc(torch.sigmoid(-head(fva)).cpu(), hva.cpu())
        if auc > best_auc:
            best_auc = auc
            best_state = {kk: v.detach().clone() for kk, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    return head


def main(cfg_path):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    nw = cfg.get("num_workers", 8)
    ds = get_dataset(cfg); n = len(ds); nc = len(ds.classes)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0)).tolist()
    probe_calib = Subset(ds, perm[int(.8 * n):int(.9 * n)])
    thresh = Subset(ds, perm[int(.9 * n):])

    cascade = build_cascade(cfg, nc, Subset(ds, range(1)), Subset(ds, range(1)))
    k = len(cascade) - 1
    f1 = cascade[0].model
    f1 = f1.module if isinstance(f1, nn.DataParallel) else f1
    f1_flops = count_flops(f1, cascade[0].resolution, dev)

    H_tr, _ = compute_harm_labels(cascade, probe_calib, num_workers=nw)
    H_te, _ = compute_harm_labels(cascade, thresh, num_workers=nw)

    print(f"\nthumbnail-probe baseline (prior result): held-out AUC ~0.50 (chance)")
    print(f"full f1 = {f1_flops/1e9:.3f} G\n")
    print(f"{'tap':<8}{'feat_dim':>9}{'partial/f1':>12}{'AUC f1':>9}{'AUC f2':>9}")
    for tap in TAPS:
        tapper = F1Tap(f1, tap)
        cost = count_flops(tapper, 112, dev)
        ftr = extract(tapper, probe_calib, dev, 128, nw)
        fte = extract(tapper, thresh, dev, 128, nw)
        head = train_head(ftr, H_tr.float(), FEAT_DIM[tap], k, dev)
        head.eval()
        with torch.no_grad():
            safety_te = torch.sigmoid(-head(fte.to(dev))).cpu()
        auc1 = _route_auc(safety_te[:, :1], H_te[:, :1])
        auc2 = _route_auc(safety_te[:, 1:2], H_te[:, 1:2])
        print(f"{tap:<8}{FEAT_DIM[tap]:>9}{cost/f1_flops:>11.2f}x{auc1:>9.4f}{auc2:>9.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="cfgs/imagenette.yaml")
    main(ap.parse_args().cfg)
