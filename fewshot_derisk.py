"""Few-shot FGVC de-risk on CUB with frozen DINOv2 features.

Tests the two bets behind the proposed method:
  (1) patch-level retrieval BEATS global CLS retrieval (the contribution),
  (2) both beat the standard few-shot baseline (linear probe) (it "works").

K-shot support sampled from official train; evaluated on the full official test,
averaged over seeds. Methods (all training-free except linear probe):
  CLS-proto    nearest class-prototype on DINOv2 CLS embeddings
  patch-Chamfer  mean over query patches of max cosine to a class's support patches
  CLS+patch    sum of the two (normalized) scores
  linear-probe  logistic regression on CLS features (supervised few-shot baseline)

Run:  python fewshot_derisk.py
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from cub_core import official_split
from utils import ResizedDataset, get_dataset, resize_transform

DEV = "cuda" if torch.cuda.is_available() else "cpu"
RES = 224


@torch.no_grad()
def dino_feats(dino, ds, idx, want_patch, nw=8, bs=64):
    loader = DataLoader(ResizedDataset(Subset(ds, idx), resize_transform(RES)),
                        batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    dino = dino.to(DEV).eval()
    cls, pat = [], []
    for x, _ in loader:
        out = dino.forward_features(x.to(DEV))
        cls.append(F.normalize(out["x_norm_clstoken"], dim=-1).cpu())
        if want_patch:
            pat.append(F.normalize(out["x_norm_patchtokens"], dim=-1).half().cpu())
    cls = torch.cat(cls)
    pat = torch.cat(pat) if want_patch else None
    return cls, pat


def cls_proto_acc(supC, supY, qC, qY, nc):
    protos = torch.stack([F.normalize(supC[supY == c].mean(0), dim=0) for c in range(nc)])
    return (qC @ protos.t()).argmax(1), protos


def linear_probe_acc(supC, supY, qC, qY, nc, steps=300):
    head = nn.Linear(supC.shape[1], nc).to(DEV)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-2, weight_decay=1e-3)
    X, Yt = supC.to(DEV), supY.to(DEV)
    for _ in range(steps):
        opt.zero_grad(); F.cross_entropy(head(X), Yt).backward(); opt.step()
    with torch.no_grad():
        return head(qC.to(DEV)).argmax(1).cpu()


def patch_scores(supP, supY, qP, nc, bs=64):
    """score(c) = mean over query patches of max cosine to class c's support patches."""
    classP = [supP[supY == c].reshape(-1, supP.shape[-1]).to(DEV).half() for c in range(nc)]
    out = []
    for i in range(0, qP.shape[0], bs):
        Qb = qP[i:i + bs].to(DEV).half()
        sc = torch.empty(Qb.shape[0], nc, device=DEV)
        for c in range(nc):
            sims = torch.matmul(Qb, classP[c].t())
            sc[:, c] = sims.max(dim=2).values.mean(dim=1)
        out.append(sc.cpu())
    return torch.cat(out)


def main(shots, seeds):
    ds = get_dataset("cub"); nc = len(ds.classes)
    Y = torch.tensor([lbl for _, lbl in ds.base.samples])
    train, test = official_split(ds)
    Ytr_all = Y[torch.tensor(train)]

    print("loading DINOv2 vitb14 + extracting test features (CLS+patch) ...")
    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', verbose=False)
    teC, teP = dino_feats(dino, ds, test, want_patch=True)
    teY = Y[torch.tensor(test)]
    print(f"test: {teC.shape[0]} imgs, CLS dim {teC.shape[1]}, patches {teP.shape[1]}")

    results = {}
    for K in shots:
        for method in ["CLS-proto", "patch-Chamfer", "CLS+patch", "linear-probe"]:
            results[(K, method)] = []
    for seed in range(seeds):
        g = torch.Generator().manual_seed(seed)
        for K in shots:
            # sample K support per class from train pool
            sup_idx = []
            for c in range(nc):
                pool = [train[i] for i in range(len(train)) if int(Ytr_all[i]) == c]
                pick = [pool[j] for j in torch.randperm(len(pool), generator=g)[:K].tolist()]
                sup_idx += pick
            supC, supP = dino_feats(dino, ds, sup_idx, want_patch=True)
            supY = Y[torch.tensor(sup_idx)]

            pc, _ = cls_proto_acc(supC, supY, teC, teY, nc)
            cls_acc = (pc == teY).float().mean().item()
            pscore = patch_scores(supP, supY, teP, nc)
            patch_acc = (pscore.argmax(1) == teY).float().mean().item()
            # fusion: normalize each score to z and sum
            cscore = teC @ torch.stack([F.normalize(supC[supY == c].mean(0), dim=0) for c in range(nc)]).t()
            fuse = (F.normalize(cscore, dim=1) + F.normalize(pscore, dim=1)).argmax(1)
            fuse_acc = (fuse == teY).float().mean().item()
            lp = linear_probe_acc(supC, supY, teC, teY, nc)
            lp_acc = (lp == teY).float().mean().item()

            results[(K, "CLS-proto")].append(cls_acc)
            results[(K, "patch-Chamfer")].append(patch_acc)
            results[(K, "CLS+patch")].append(fuse_acc)
            results[(K, "linear-probe")].append(lp_acc)
            print(f"  seed{seed} K={K}: CLS={cls_acc:.4f} patch={patch_acc:.4f} fuse={fuse_acc:.4f} linprobe={lp_acc:.4f}")

    print(f"\n#### few-shot CUB (DINOv2 vitb14), mean over {seeds} seeds ####")
    print(f"{'K':>3}  {'CLS-proto':>11}{'patch-Cham':>12}{'CLS+patch':>11}{'lin-probe':>11}")
    for K in shots:
        row = f"{K:>3}  "
        for m in ["CLS-proto", "patch-Chamfer", "CLS+patch", "linear-probe"]:
            v = torch.tensor(results[(K, m)])
            row += f"{v.mean():>8.4f}±{v.std():.3f}".rjust(12)
        print(row)
    print("\n  bets: patch-Chamfer > CLS-proto (contribution);  retrieval > linear-probe (it works)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, nargs="+", default=[1, 5, 16])
    ap.add_argument("--seeds", type=int, default=3)
    main(ap.parse_args().shots, ap.parse_args().seeds)
