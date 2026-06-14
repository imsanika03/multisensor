"""CoRE diagnostic: is the Confusion-Conditioned Residual Experts idea alive?

Tests, in order, on Imagewoof with a resnet50@224 base, evaluated on the OFFICIAL
held-out val folder (split 50/50: experts fit on half A, tested on half B):

  H1  errors are concentrated in recurring top-k confusion pairs (not arbitrary)
  gap top-k repairable gap = top-k acc - top-1 acc (room for experts to work)
  H2/H3  small frozen-feature confusion experts repair top-1, esp. where y is
         already in the base top-3.

Expert: for each top confused pair {A,B}, a small MLP on FROZEN backbone features
trained to separate A vs B; at inference, for examples whose base top-2 == {A,B},
the expert re-decides among {A,B}. Also a global residual-MLP rerank baseline.

GO:   +1.0 to +2.0 top-1 points (esp. on y-in-base-top3).
KILL: top-3 gap small, or frozen experts don't beat base.

Run:  python core_diag.py
"""

import argparse
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import yaml
import torchvision
from torch.utils.data import DataLoader

from cascade.stage import build_cascade
from utils import ResizedDataset, get_dataset, resize_transform, _RGBView


@torch.no_grad()
def extract(feature_net, head, data, res, device, bs, nw):
    loader = DataLoader(ResizedDataset(data, resize_transform(res)),
                        batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    feats, logits, ys = [], [], []
    for x, y in loader:
        x = x.to(device)
        f = feature_net(x).flatten(1)
        feats.append(f.cpu()); logits.append(head(f).cpu()); ys.append(y)
    return torch.cat(feats), torch.cat(logits), torch.cat(ys)


def topk_acc(logits, y, k):
    return (logits.topk(k, 1).indices == y.unsqueeze(1)).any(1).float().mean().item()


def train_pair_expert(feat, lab, dev, steps=300):
    """lab in {0,1}; small MLP on frozen features."""
    f, l = feat.to(dev), lab.float().to(dev)
    net = nn.Sequential(nn.Linear(f.shape[1], 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 1)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    pw = ((l == 0).sum() / (l == 1).sum().clamp(min=1)).to(dev)
    bce = nn.BCEWithLogitsLoss(pos_weight=pw)
    net.train()
    for _ in range(steps):
        opt.zero_grad(); bce(net(f).squeeze(1), l).backward(); opt.step()
    return net.eval()


def main(cfg_path, valdir, n_pairs, alpha):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    nw = cfg.get("num_workers", 8)
    base_ds = get_dataset(cfg); nc = len(base_ds.classes)
    casc = build_cascade(cfg, nc, base_ds, base_ds)
    base = casc[1].model                                   # resnet50 @224
    base = base.module if isinstance(base, nn.DataParallel) else base
    base = base.to(dev).eval()
    res = casc[1].resolution
    feature_net = nn.Sequential(*list(base.children())[:-1])   # resnet50 up to avgpool -> 2048
    head = base.fc

    val = _RGBView(torchvision.datasets.ImageFolder(valdir))
    F, Z, Y = extract(feature_net, head, val, res, dev, 128, nw)
    N = Y.numel()
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(N, generator=g)
    A, B = perm[:N // 2], perm[N // 2:]                     # calib (fit) / eval (test)

    def sub(idx): return F[idx], Z[idx], Y[idx]
    Fa, Za, Ya = sub(A); Fb, Zb, Yb = sub(B)

    print(f"\n#### CoRE diagnostic: base=resnet50@{res} on {valdir} (n={N}; fit={len(A)}, eval={len(B)}) ####")

    # ---- gap ----
    print("  [gap] base accuracy on eval half:")
    for k in [1, 2, 3, 5]:
        print(f"     top-{k} = {topk_acc(Zb, Yb, k):.4f}"
              + (f"   repairable gap top{k}-top1 = {topk_acc(Zb,Yb,k)-topk_acc(Zb,Yb,1):+.4f}" if k > 1 else ""))

    # ---- H1: error concentration (on calib half) ----
    pa = Za.argmax(1)
    err = pa != Ya
    pairs = Counter()
    for t, p in zip(Ya[err].tolist(), pa[err].tolist()):
        pairs[frozenset((t, p))] += 1
    tot_err = int(err.sum())
    top = pairs.most_common(n_pairs)
    cov = sum(c for _, c in top) / max(tot_err, 1)
    print(f"  [H1] calib errors={tot_err}; top-{n_pairs} confusion pairs cover {cov:.1%} of errors:")
    cls = base_ds.classes
    for s, c in top[:6]:
        i, j = sorted(s)
        print(f"       {cls[i]:<22} <-> {cls[j]:<22}  {c} errs")

    # ---- H2/H3: pair experts ----
    base_top1 = (Zb.argmax(1) == Yb).float().mean().item()
    pred = Zb.argmax(1).clone()
    top2 = Zb.topk(2, 1).indices
    matched = 0
    for s, _ in top:
        i, j = sorted(s)
        m_fit = (Ya == i) | (Ya == j)
        if m_fit.sum() < 10:
            continue
        lab = (Ya[m_fit] == j).long()                       # 1 == class j
        expert = train_pair_expert(Fa[m_fit], lab, dev)
        # eval examples whose base top-2 is exactly {i, j}
        e_mask = ((top2 == i).any(1) & (top2 == j).any(1))
        if e_mask.sum() == 0:
            continue
        with torch.no_grad():
            ej = torch.sigmoid(expert(Fb[e_mask].to(dev)).squeeze(1)).cpu()   # P(class j)
        choice = torch.where(ej >= 0.5, torch.tensor(j), torch.tensor(i))
        pred[e_mask] = choice
        matched += int(e_mask.sum())
    core_top1 = (pred == Yb).float().mean().item()

    # subset where y already in base top-3 (the repairable population)
    in3 = (Zb.topk(3, 1).indices == Yb.unsqueeze(1)).any(1)
    base_in3 = (Zb.argmax(1)[in3] == Yb[in3]).float().mean().item()
    core_in3 = (pred[in3] == Yb[in3]).float().mean().item()

    print(f"  [H2/H3] pair experts applied to {matched}/{len(B)} eval examples (matched a confusion pair):")
    print(f"     base top-1   = {base_top1:.4f}")
    print(f"     CoRE top-1   = {core_top1:.4f}   ({(core_top1-base_top1)*100:+.2f} pts)")
    print(f"     on y-in-top3 subset (n={int(in3.sum())}): base={base_in3:.4f}  CoRE={core_in3:.4f}  ({(core_in3-base_in3)*100:+.2f} pts)")
    verdict = "ALIVE (>=+1.0 pt)" if (core_top1 - base_top1) >= 0.01 else "weak/kill (<+1.0 pt)"
    print(f"     VERDICT: {verdict}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="cfgs/imagewoof.yaml")
    ap.add_argument("--valdir", default="data/imagewoof2/val")
    ap.add_argument("--n_pairs", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=1.0)
    a = ap.parse_args()
    main(a.cfg, a.valdir, a.n_pairs, a.alpha)
