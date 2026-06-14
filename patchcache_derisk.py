"""PatchCache de-risk (CLS version) + novelty-critical baselines, on CUB.

Indexes DINOv2 CLS embeddings of all CUB train images as nonparametric memory.
For each test image: base resnet50 proposes top-k; cache_score(c) = logsumexp of
top-m cosine sims to class c's exemplars; rerank within top-k via
  score(c) = base_logit(c) + lambda * cache_score(c).

Reports four numbers so we see both "does it work" AND "is it just DINOv2":
  base            base resnet50 top-1                         (floor)
  DINOv2-full     DINOv2 CLS kNN over ALL 200 classes         ("just use DINOv2" bar)
  uncond-rerank   base_logit + lambda*cache over ALL classes  (does conditioning matter?)
  PatchCache      base_logit + lambda*cache over base top-k   (the method)

lambda/k chosen on a calib half of test, reported on the eval half.
Run:  python patchcache_derisk.py
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from cascade.stage import build_backbone
from cub_core import official_split
from utils import ResizedDataset, get_dataset, resize_transform

DEV = "cuda" if torch.cuda.is_available() else "cpu"
RES = 224


@torch.no_grad()
def base_logits(model, ds, idx, nw=8):
    loader = DataLoader(ResizedDataset(Subset(ds, idx), resize_transform(RES)),
                        batch_size=128, shuffle=False, num_workers=nw, pin_memory=True)
    model = model.to(DEV).eval()
    return torch.cat([model(x.to(DEV)).cpu() for x, _ in loader])


@torch.no_grad()
def dino_embed(dino, ds, idx, nw=8):
    loader = DataLoader(ResizedDataset(Subset(ds, idx), resize_transform(RES)),
                        batch_size=128, shuffle=False, num_workers=nw, pin_memory=True)
    dino = dino.to(DEV).eval()
    out = []
    for x, _ in loader:
        e = dino(x.to(DEV))                       # CLS embedding [B, D]
        out.append(nn.functional.normalize(e, dim=1).cpu())
    return torch.cat(out)


def cache_scores(q, T, Tlab, nc, m=5):
    """logsumexp of top-m cosine sims to each class's exemplars. q,T L2-normalized."""
    sims = q @ T.t()                              # [Nq, Ntrain]
    scores = torch.full((q.shape[0], nc), -1e9)
    for c in range(nc):
        cols = (Tlab == c)
        if cols.sum() == 0:
            continue
        cs = sims[:, cols]
        topm = cs.topk(min(m, cs.shape[1]), dim=1).values
        scores[:, c] = torch.logsumexp(topm, dim=1)
    return scores


def topk_acc(logits, y, k):
    return (logits.topk(k, 1).indices == y.unsqueeze(1)).any(1).float().mean().item()


def main():
    ds = get_dataset("cub"); nc = len(ds.classes)
    Y = torch.tensor([lbl for _, lbl in ds.base.samples])
    train, test = official_split(ds)
    g = torch.Generator().manual_seed(0)
    tp = [test[i] for i in torch.randperm(len(test), generator=g).tolist()]
    calib, evl = tp[:len(tp) // 2], tp[len(tp) // 2:]

    # base resnet50 (augmented, cached)
    base = build_backbone("resnet50", nc, pretrained=True)
    base.load_state_dict(torch.load("checkpoints/cub_base/resnet50_224_aug.pt", map_location=DEV))
    Ztr_unused = None
    Zc = base_logits(base, ds, calib); Ze = base_logits(base, ds, evl)
    Yc, Ye = Y[torch.tensor(calib)], Y[torch.tensor(evl)]
    print(f"base test top1={ (Ze.argmax(1)==Ye).float().mean():.4f}  top3={topk_acc(Ze,Ye,3):.4f}  top5={topk_acc(Ze,Ye,5):.4f}")

    # DINOv2 memory
    print("loading DINOv2 vitb14 + indexing CUB train ...")
    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', verbose=False)
    T = dino_embed(dino, ds, train); Tlab = Y[torch.tensor(train)]
    Ec = dino_embed(dino, ds, calib); Ee = dino_embed(dino, ds, evl)
    print(f"indexed {T.shape[0]} train embeddings, dim={T.shape[1]}")

    cache_c = cache_scores(Ec, T, Tlab, nc); cache_e = cache_scores(Ee, T, Tlab, nc)

    # DINOv2-full kNN ("just use DINOv2") -- argmax of cache over all classes
    dino_full = (cache_e.argmax(1) == Ye).float().mean().item()

    def rerank(Zlogits, cache, lam, k=None):
        s = Zlogits.clone()
        if k is None:                              # unconditioned
            s = Zlogits + lam * cache
        else:                                      # only adjust base top-k
            mask = torch.zeros_like(Zlogits)
            tk = Zlogits.topk(k, 1).indices
            mask.scatter_(1, tk, 1.0)
            s = Zlogits + lam * cache * mask
        return s.argmax(1)

    # pick lambda (and k) on calib, report on eval
    lams = [0.1, 0.3, 0.5, 1, 2, 4, 8, 16, 32]
    best = {}
    for name, k in [("uncond", None), ("k3", 3), ("k5", 5)]:
        bestv, bl = -1, None
        for lam in lams:
            acc = (rerank(Zc, cache_c, lam, k) == Yc).float().mean().item()
            if acc > bestv:
                bestv, bl = acc, lam
        ev = (rerank(Ze, cache_e, bl, k) == Ye).float().mean().item()
        best[name] = (bl, ev)

    base_e = (Ze.argmax(1) == Ye).float().mean().item()
    print(f"\n#### PatchCache de-risk (eval n={len(evl)}) ####")
    print(f"  base resnet50            top1 = {base_e:.4f}   (top3 ceiling for k=3 rerank = {topk_acc(Ze,Ye,3):.4f})")
    print(f"  DINOv2-full kNN          top1 = {dino_full:.4f}   <-- 'just use DINOv2' bar")
    print(f"  uncond rerank (lam={best['uncond'][0]})    top1 = {best['uncond'][1]:.4f}")
    print(f"  PatchCache k=3 (lam={best['k3'][0]})        top1 = {best['k3'][1]:.4f}   ({(best['k3'][1]-base_e)*100:+.2f} vs base)")
    print(f"  PatchCache k=5 (lam={best['k5'][0]})        top1 = {best['k5'][1]:.4f}   ({(best['k5'][1]-base_e)*100:+.2f} vs base)")
    print(f"\n  conditioning effect (k3 - uncond) = {(best['k3'][1]-best['uncond'][1])*100:+.2f} pts")
    print(f"  gap closed by k3 = {(best['k3'][1]-base_e)/(topk_acc(Ze,Ye,3)-base_e)*100:.0f}% of the top-3 repairable gap")


if __name__ == "__main__":
    main()
