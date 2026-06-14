"""De-risk the Recoverability-Gated Cascade (RGC) on cached stage outputs.

RGC skips escalation when the cheap model's error is UNRECOVERABLE (f1 wrong AND
fM wrong) -- escalating those changes nothing, so returning f1 is accuracy-neutral
and saves the f2+f3 compute. The bet needs two things, both measured here from
f1's own state z1 (no training of backbones):

  (1) unrecoverable mass -- P(f1 wrong & fM wrong): the ceiling on safe savings.
  (2) recoverability separability -- AUC of predicting "recoverable" (f1 wrong &
      fM right) from z1; high AUC => RGC can cut futile escalations without
      dropping recoverable ones (which would cost accuracy).

Also simulates one RGC operating point: among a confidence cascade's escalation
set, cut futile escalations while retaining 95% of recoverable mass; report
compute saved and accuracy delta.

Run:  python derisk_rgc.py --cfg cfgs/imagewoof_diverse.yaml
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


def z1_features(L1):
    p = torch.softmax(L1, dim=1)
    top = p.topk(min(5, p.shape[1]), dim=1).values
    conf = top[:, 0]
    margin = top[:, 0] - top[:, 1]
    ent = -(p * p.clamp_min(1e-12).log()).sum(1) / torch.log(torch.tensor(float(p.shape[1])))
    return torch.stack([conf, margin, ent], 1), torch.cat([torch.stack([conf, margin, ent], 1), top], 1), conf


def auc_bin(score, label):
    pos, neg = score[label == 1], score[label == 0]
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    alls = torch.cat([pos, neg]); order = alls.argsort()
    ranks = torch.empty_like(alls); ranks[order] = torch.arange(1, len(alls) + 1, dtype=alls.dtype)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * neg.numel()))


def train_predictor(feat, label, device, steps=400):
    f, l = feat.to(device), label.float().to(device)
    net = nn.Sequential(nn.Linear(f.shape[1], 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    pw = ((l == 0).sum() / (l == 1).sum().clamp(min=1)).to(device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pw)
    net.train()
    for _ in range(steps):
        opt.zero_grad(); loss = bce(net(f).squeeze(1), l); loss.backward(); opt.step()
    return net.eval()


def main(cfg_path):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    nw = cfg.get("num_workers", 8)
    ds = get_dataset(cfg); n = len(ds); nc = len(ds.classes)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0)).tolist()
    cal = Subset(ds, perm[int(.9 * n):])
    val = Subset(ds, perm[int(.7 * n):int(.8 * n)])
    cascade = build_cascade(cfg, nc, Subset(ds, range(1)), Subset(ds, range(1)))

    def prep(split):
        L1, y = stage_logits(cascade[0].model, split, cascade[0].resolution, dev, 128, nw)
        L3, _ = stage_logits(cascade[-1].model, split, cascade[-1].resolution, dev, 128, nw)
        f1c = (L1.argmax(1) == y); f3c = (L3.argmax(1) == y)
        _, zfull, conf = z1_features(L1)
        recoverable = (~f1c) & f3c               # f1 wrong, fM right -> MUST escalate
        unrecoverable = (~f1c) & (~f3c)           # both wrong -> futile escalation
        return dict(y=y, f1c=f1c, f3c=f3c, conf=conf, z=zfull,
                    recov=recoverable, unrec=unrecoverable)

    c, v = prep(cal), prep(val)
    N = v["y"].numel()
    print(f"\n#### {cfg_path}  (val n={N}) ####")
    print(f"  f1 acc={v['f1c'].float().mean():.4f}  fM acc={v['f3c'].float().mean():.4f}  "
          f"f1 err={1-v['f1c'].float().mean():.4f}")
    print(f"  recoverable  (f1 wrong, fM right) = {v['recov'].float().mean():.4f}")
    print(f"  unrecoverable(f1 wrong, fM wrong) = {v['unrec'].float().mean():.4f}   <-- RGC savings ceiling")
    fw = ~v["f1c"]
    print(f"  among f1 ERRORS: {v['recov'][fw].float().mean():.3f} recoverable / "
          f"{v['unrec'][fw].float().mean():.3f} unrecoverable")

    # (2) separability: predict recoverable from z1 (train on cal, eval on val)
    net = train_predictor(c["z"], c["recov"].long(), dev)
    with torch.no_grad():
        score = torch.sigmoid(net(v["z"].to(dev)).squeeze(1)).cpu()    # P(recoverable)
    print(f"  recoverability AUC (z1 -> recoverable):  MLP={auc_bin(score, v['recov'].long()):.4f}  "
          f"conf-only={auc_bin(-v['conf'], v['recov'].long()):.4f}")
    # restricted to the hard sub-question: among f1 errors, does fM fix it?
    if fw.sum() > 5:
        print(f"  among-f1-errors AUC (z1 -> fM fixes):     "
              f"MLP={auc_bin(score[fw], (v['recov'][fw]).long()):.4f}  "
              f"conf-only={auc_bin(-v['conf'][fw], (v['recov'][fw]).long()):.4f}")

    # (3) RGC operating point: confidence cascade escalates S={conf<tau}; RGC skips
    # the lowest-P(recoverable) escalations while retaining 95% of recoverable mass.
    tau = torch.quantile(v["conf"], 0.5).item()        # escalate ~half (illustrative)
    S = v["conf"] < tau
    nS = int(S.sum())
    recov_in_S = v["recov"] & S
    if recov_in_S.sum() > 0 and nS > 0:
        # keep escalations with score >= thr; choose thr to retain 95% recoverable
        thr = torch.quantile(score[recov_in_S], 0.05).item()
        skip = S & (score < thr)                        # escalations RGC cuts
        acc_full = v["f1c"].clone(); acc_full[S] = v["f3c"][S]   # confidence cascade
        acc_rgc = acc_full.clone(); acc_rgc[skip] = v["f1c"][skip]  # cut -> return f1
        print(f"  RGC op-point (escalate {nS}/{N}={nS/N:.0%}, retain 95% recoverable):")
        print(f"     escalations cut = {int(skip.sum())}/{nS} ({skip.sum()/max(nS,1):.0%})  "
              f"-> ~{skip.sum()/max(nS,1):.0%} of escalation compute saved")
        print(f"     acc: conf-cascade={acc_full.float().mean():.4f}  RGC={acc_rgc.float().mean():.4f}  "
              f"drop={acc_full.float().mean()-acc_rgc.float().mean():+.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="cfgs/imagewoof_diverse.yaml")
    main(ap.parse_args().cfg)
