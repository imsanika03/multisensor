"""De-risk disagreement-based routing.

Hypothesis: among errors of a CHEAP committee (two diverse cheap models f1,f2),
DISAGREEMENT predicts recoverability (will the expensive fM fix it?) far better
than a single model's confidence -- which was ~chance (AUC 0.5) and killed RGC.
Intuition: agree-but-wrong = fundamentally confusing (fM also wrong, unrecoverable);
disagree-but-wrong = ambiguous-but-resolvable (fM fixes it, recoverable).

Committee = logit-avg(f1,f2). recoverable = committee wrong & fM right.
We report, among committee ERRORS:
  - P(fM right | disagree) vs P(fM right | agree)        [direct hypothesis test]
  - AUC of predicting "fM fixes" from: JS-divergence alone, agree/disagree alone,
    an MLP on committee+disagreement features, vs single-model confidence (baseline).

Run:  python derisk_disagree.py --cfg cfgs/imagewoof_diverse.yaml
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset

from cascade.stage import build_cascade
from utils import ResizedDataset, get_dataset, resize_transform


@torch.no_grad()
def stage_logits(model, data, res, device, bs, nw):
    loader = DataLoader(ResizedDataset(data, resize_transform(res)),
                        batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    model = model.to(device).eval()
    L, Y = [], []
    for x, y in loader:
        L.append(model(x.to(device)).cpu()); Y.append(y)
    return torch.cat(L), torch.cat(Y)


def auc_bin(score, label):
    pos, neg = score[label == 1], score[label == 0]
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    a = torch.cat([pos, neg]); o = a.argsort()
    r = torch.empty_like(a); r[o] = torch.arange(1, len(a) + 1, dtype=a.dtype)
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * neg.numel()))


def js_div(p, q):
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * (a.clamp_min(1e-12).log() - b.clamp_min(1e-12).log())).sum(1)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def committee_features(L1, L2):
    p1, p2 = torch.softmax(L1, 1), torch.softmax(L2, 1)
    a1, a2 = L1.argmax(1), L2.argmax(1)
    Lc = L1 + L2
    pc = torch.softmax(Lc, 1)
    conf1, conf2, confc = p1.max(1).values, p2.max(1).values, pc.max(1).values
    agree = (a1 == a2).float()
    js = js_div(p1, p2)
    feats = torch.stack([conf1, conf2, confc, agree, js,
                         conf1 - conf2, (confc - conf1).abs()], 1)
    return feats, Lc.argmax(1), agree, js, confc


def train_mlp(f, l, dev, steps=400):
    f, l = f.to(dev), l.float().to(dev)
    net = nn.Sequential(nn.Linear(f.shape[1], 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    pw = ((l == 0).sum() / (l == 1).sum().clamp(min=1)).to(dev)
    bce = nn.BCEWithLogitsLoss(pos_weight=pw)
    net.train()
    for _ in range(steps):
        opt.zero_grad(); bce(net(f).squeeze(1), l).backward(); opt.step()
    return net.eval()


def main(cfg_path):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    nw = cfg.get("num_workers", 8)
    ds = get_dataset(cfg); n = len(ds); nc = len(ds.classes)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0)).tolist()
    cal = Subset(ds, perm[int(.9 * n):]); val = Subset(ds, perm[int(.7 * n):int(.8 * n)])
    casc = build_cascade(cfg, nc, Subset(ds, range(1)), Subset(ds, range(1)))

    def prep(split):
        L1, y = stage_logits(casc[0].model, split, casc[0].resolution, dev, 128, nw)
        L2, _ = stage_logits(casc[1].model, split, casc[1].resolution, dev, 128, nw)
        L3, _ = stage_logits(casc[-1].model, split, casc[-1].resolution, dev, 128, nw)
        feats, cpred, agree, js, confc = committee_features(L1, L2)
        cwrong = (cpred != y); fmright = (L3.argmax(1) == y)
        return dict(y=y, feats=feats, agree=agree, js=js, confc=confc,
                    cwrong=cwrong, fmright=fmright, recov=(cwrong & fmright))

    c, v = prep(cal), prep(val)
    print(f"\n#### {cfg_path} ####")
    print(f"  committee acc={1-v['cwrong'].float().mean():.4f}  fM acc={v['fmright'].float().mean():.4f}")

    e = v["cwrong"]                                   # committee errors (val)
    ec = c["cwrong"]
    print(f"  committee errors: {int(e.sum())}/{v['y'].numel()};  of those fM fixes (recoverable)="
          f"{v['fmright'][e].float().mean():.3f}")

    # direct hypothesis: recoverability conditioned on agree vs disagree (among errors)
    dis = v["agree"] == 0
    er_dis, er_ag = e & dis, e & (~dis)
    if er_dis.sum() > 0 and er_ag.sum() > 0:
        print(f"  P(fM fixes | committee error & DISAGREE) = {v['fmright'][er_dis].float().mean():.3f}  (n={int(er_dis.sum())})")
        print(f"  P(fM fixes | committee error & AGREE)    = {v['fmright'][er_ag].float().mean():.3f}  (n={int(er_ag.sum())})")

    # AUC among committee errors: predict fM-fixes
    lab = v["fmright"][e].long()
    mlp = train_mlp(c["feats"][ec], c["fmright"][ec].long(), dev)
    with torch.no_grad():
        s_mlp = torch.sigmoid(mlp(v["feats"][e].to(dev)).squeeze(1)).cpu()
    print(f"  among-errors AUC (predict fM fixes):")
    print(f"     committee MLP (conf+agree+JS) = {auc_bin(s_mlp, lab):.4f}")
    print(f"     JS-divergence alone           = {auc_bin(v['js'][e], lab):.4f}")
    print(f"     agree/disagree alone          = {auc_bin(-v['agree'][e], lab):.4f}")
    print(f"     committee confidence alone    = {auc_bin(-v['confc'][e], lab):.4f}   (single-signal baseline)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="cfgs/imagewoof_diverse.yaml")
    main(ap.parse_args().cfg)
