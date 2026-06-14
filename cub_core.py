"""CoRE on CUB-200, V4-first protocol (done properly).

Fixes over the broken first pass: data augmentation (CUB overfits without it),
official train/test split (-> ~58 test images per pair, not 8-15), and a
regularized specialist (fine-tune layer4+head with aug + early stop, not a full
resnet50 memorizing ~100 images).

Step 1 (gate): can a strong specialist separate the base's top confusion pairs on
the official TEST set?  >= +5 pts mean gain -> run the V1-V4 ladder; else CoRE dead.

Run:  python cub_core.py
"""

import argparse
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from cascade.stage import build_backbone, _wrap, _unwrap
from utils import ResizedDataset, get_dataset, resize_transform, aug_transform

DEV = "cuda" if torch.cuda.is_available() else "cpu"
RES = 224
ROOT = "data/CUB_200_2011"


def official_split(ds):
    id2rel = {}
    for line in open(f"{ROOT}/images.txt"):
        i, p = line.split(); id2rel[i] = p
    id2tr = {}
    for line in open(f"{ROOT}/train_test_split.txt"):
        i, t = line.split(); id2tr[i] = int(t)
    rel2tr = {id2rel[i]: id2tr[i] for i in id2rel}
    train, test = [], []
    for k, (path, _) in enumerate(ds.base.samples):
        rel = path.split("/images/")[-1]
        (train if rel2tr[rel] == 1 else test).append(k)
    return train, test


@torch.no_grad()
def eval_logits(model, ds, idx, nw=8):
    loader = DataLoader(ResizedDataset(Subset(ds, idx), resize_transform(RES)),
                        batch_size=128, shuffle=False, num_workers=nw, pin_memory=True)
    model = model.to(DEV).eval()
    return torch.cat([model(x.to(DEV)).cpu() for x, _ in loader])


def train_aug(model, ds, train_idx, es_idx, Y, epochs, lr, nw=8, patience=6, freeze_to_l4=False):
    if freeze_to_l4:
        for p in model.parameters():
            p.requires_grad = False
        for p in list(model.layer4.parameters()) + list(model.fc.parameters()):
            p.requires_grad = True
    model = _wrap(model, DEV, True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    loader = DataLoader(ResizedDataset(Subset(ds, train_idx), aug_transform(RES)),
                        batch_size=64, shuffle=True, num_workers=nw, pin_memory=True, drop_last=True)
    best, best_state, noimp = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        for x, y in loader:
            x = x.to(DEV); t = y.to(DEV)
            opt.zero_grad(); nn.functional.cross_entropy(model(x), t).backward(); opt.step()
        Ze = eval_logits(model, ds, es_idx, nw)
        acc = (Ze.argmax(1) == Y[torch.tensor(es_idx)]).float().mean().item()
        if acc > best:
            best, best_state, noimp = acc, {k: v.detach().cpu().clone() for k, v in _unwrap(model).state_dict().items()}, 0
        else:
            noimp += 1
            if noimp >= patience:
                break
    _unwrap(model).load_state_dict(best_state)
    return _unwrap(model).to(DEV).eval(), best


def train_pair_specialist(ds, fit_idx, es_idx, a, b, Y, nw=8):
    """Regularized 2-way specialist: layer4+head fine-tune, aug, early stop."""
    net = build_backbone("resnet50", 2, pretrained=True)
    for p in net.parameters():
        p.requires_grad = False
    for p in list(net.layer4.parameters()) + list(net.fc.parameters()):
        p.requires_grad = True
    net = _wrap(net, DEV, True)
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-2)
    ymap = {a: 0, b: 1}
    loader = DataLoader(ResizedDataset(Subset(ds, fit_idx), aug_transform(RES)),
                        batch_size=32, shuffle=True, num_workers=nw, pin_memory=True)
    best, best_state, noimp = -1.0, None, 0
    es_t = torch.tensor([ymap[int(Y[i])] for i in es_idx])
    for ep in range(25):
        net.train()
        for x, y in loader:
            x = x.to(DEV); t = torch.tensor([ymap[int(v)] for v in y], device=DEV)
            opt.zero_grad(); nn.functional.cross_entropy(net(x), t).backward(); opt.step()
        with torch.no_grad():
            zl = eval_logits(net, ds, es_idx, nw)
        acc = (zl.argmax(1) == es_t).float().mean().item()
        if acc > best:
            best, best_state, noimp = acc, {k: v.detach().cpu().clone() for k, v in _unwrap(net).state_dict().items()}, 0
        else:
            noimp += 1
            if noimp >= 6:
                break
    _unwrap(net).load_state_dict(best_state)
    return _unwrap(net).to(DEV).eval()


def main(n_pairs):
    ds = get_dataset("cub"); nc = len(ds.classes); cls = ds.classes
    Y = torch.tensor([lbl for _, lbl in ds.base.samples])
    train, test = official_split(ds)
    g = torch.Generator().manual_seed(0)
    tperm = [train[i] for i in torch.randperm(len(train), generator=g).tolist()]
    es = tperm[:len(tperm) // 10]; trfit = tperm[len(tperm) // 10:]   # base early-stop carve
    print(f"CUB official split: train={len(train)} (fit {len(trfit)}, es {len(es)}), test={len(test)}")

    rdir = Path("checkpoints/cub_base"); rdir.mkdir(parents=True, exist_ok=True)
    ckpt = rdir / "resnet50_224_aug.pt"
    base = build_backbone("resnet50", nc, pretrained=True)
    if ckpt.exists():
        base.load_state_dict(torch.load(ckpt, map_location=DEV)); base = base.to(DEV).eval()
        print(f"loaded base from {ckpt}")
    else:
        print("training base resnet50@224 on CUB (augmented) ...")
        base, bacc = train_aug(base, ds, trfit, es, Y, epochs=40, lr=1e-3)
        torch.save(base.state_dict(), ckpt); print(f"saved base (es acc={bacc:.4f})")

    Zt = eval_logits(base, ds, test)
    Yt = Y[torch.tensor(test)]
    top1 = (Zt.argmax(1) == Yt).float().mean().item()
    top3 = (Zt.topk(3, 1).indices == Yt.unsqueeze(1)).any(1).float().mean().item()
    print(f"base TEST top-1={top1:.4f}  top-3={top3:.4f}  repairable gap={top3-top1:+.4f}")

    # confusion pairs from base errors on the es carve (held out from base training)
    Zes = eval_logits(base, ds, es); Yes = Y[torch.tensor(es)]
    pa = Zes.argmax(1)
    pairs = Counter()
    for k in range(len(es)):
        if pa[k] != Yes[k]:
            pairs[frozenset((int(Yes[k]), int(pa[k])))] += 1
    top = [tuple(sorted(s)) for s, _ in pairs.most_common(n_pairs)]

    print(f"\nV4-FIRST (regularized specialist, official-test eval), top-{n_pairs} pairs:")
    gains = []
    test_t = torch.tensor(test)
    for (a, b) in top:
        fit_idx = [i for i in trfit if int(Y[i]) in (a, b)]
        es_pair = [i for i in es if int(Y[i]) in (a, b)]
        ev_idx = [test[k] for k in range(len(test)) if int(Yt[k]) in (a, b)]
        if len(ev_idx) < 15 or len(fit_idx) < 20 or len(es_pair) < 4:
            continue
        zab = Zt[torch.tensor([k for k in range(len(test)) if int(Yt[k]) in (a, b)])][:, [a, b]]
        tloc = torch.tensor([0 if int(Y[i]) == a else 1 for i in ev_idx])
        base_pair = (zab.argmax(1) == tloc).float().mean().item()
        spec = train_pair_specialist(ds, fit_idx, es_pair, a, b, Y)
        sl = eval_logits(spec, ds, ev_idx)
        spec_pair = (sl.argmax(1) == tloc).float().mean().item()
        gains.append(spec_pair - base_pair)
        print(f"  {cls[a][:26]:<27}|{cls[b][:26]:<27} fit={len(fit_idx)} test={len(ev_idx)}  "
              f"base={base_pair:.3f} spec={spec_pair:.3f} ({(spec_pair-base_pair)*100:+.1f})")

    avg = sum(gains) / max(len(gains), 1)
    print(f"\n  mean specialist gain (official-test pair-val) = {avg*100:+.2f} pts over {len(gains)} pairs")
    print("  GATE PASSED -> run V1-V4 ladder." if avg >= 0.05 else
          "  GATE FAILED (<+5 pts) -> even regularized specialists can't separate -> CoRE dead.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_pairs", type=int, default=8)
    main(ap.parse_args().n_pairs)
