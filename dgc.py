"""Disagreement-Gated Cascade (DGC): scale validation + frontier comparison.

Committee = the cascade's cheap stages (all but last); fM = last stage. Always
run the committee; escalate to fM on DISAGREEMENT. Baseline gate: escalate on low
committee CONFIDENCE. Same committee+fM and identical cost = c_committee +
e*c_fM, so at a matched escalation rate e the two gates have identical cost --
the ONLY difference is which examples each escalates. If DGC's accuracy(e) curve
dominates the confidence gate's, the disagreement signal wins.

Evaluated on the OFFICIAL held-out val folder (e.g. data/imagewoof2/val), which
the backbones never saw -> big n, tight error bars, training-free.

Run:  python dgc.py --cfg cfgs/imagewoof_diverse.yaml --valdir data/imagewoof2/val
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import yaml
import torchvision
from torch.utils.data import DataLoader, Subset

from cascade.stage import build_cascade
from utils import ResizedDataset, get_dataset, resize_transform

CORRUPTION = ""
SEVERITY = 3


def flops_of(model, res, device):
    model = (model.module if isinstance(model, nn.DataParallel) else model).to(device).eval()
    x = torch.zeros(1, 3, res, res, device=device)
    try:
        from torch.utils.flop_counter import FlopCounterMode
        fc = FlopCounterMode(display=False)
        with fc, torch.no_grad():
            model(x)
        return float(fc.get_total_flops())
    except Exception:
        from score.routing import count_flops
        return count_flops(model, res, device)


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


def acc_curve(committee_c, fm_c, score_high_escalates):
    """Accuracy as a function of escalation rate e, escalating highest-score first.
    Returns list of (e, acc). Escalated -> fM pred, else committee pred."""
    N = committee_c.numel()
    order = torch.argsort(score_high_escalates, descending=True)   # escalate these first
    base = committee_c.sum().item()
    pts = []
    # cumulative gain from escalating in order: delta = fm_c - committee_c
    delta = (fm_c.float() - committee_c.float())[order]
    cum = torch.cat([torch.zeros(1), torch.cumsum(delta, 0)])
    for k in range(0, N + 1, max(1, N // 50)):
        pts.append((k / N, (base + cum[k].item()) / N))
    return pts


def main(cfg_path, valdir):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    nw = cfg.get("num_workers", 8)
    # Build cascade (loads cached checkpoints); discover num_classes from train ds.
    base_ds = get_dataset(cfg); nc = len(base_ds.classes)
    casc = build_cascade(cfg, nc, Subset(base_ds, range(1)), Subset(base_ds, range(1)))
    committee, fM = casc[:-1], casc[-1]

    # Official held-out val folder (never seen in training); optionally corrupted
    # to test whether disagreement beats confidence under distribution shift.
    from utils import _RGBView, _CorruptedView
    base = torchvision.datasets.ImageFolder(valdir)
    if CORRUPTION:
        val = _CorruptedView(base, CORRUPTION, SEVERITY)
        print(f"  [corruption: {CORRUPTION} severity {SEVERITY}]")
    else:
        val = _RGBView(base)

    # Per-model logits on the official val.
    Ls = [stage_logits(s.model, val, s.resolution, dev, 128, nw)[0] for s in committee]
    Lf, y = stage_logits(fM.model, val, fM.resolution, dev, 128, nw)
    N = y.numel()

    probs = [torch.softmax(L, 1) for L in Ls]
    Lc = sum(Ls)                                   # committee = logit-average
    cpred = Lc.argmax(1)
    committee_c = (cpred == y)
    fm_c = (Lf.argmax(1) == y)
    conf_c = torch.softmax(Lc, 1).max(1).values
    js = js_div(probs[0], probs[1]) if len(probs) >= 2 else torch.zeros(N)
    agree = torch.stack([L.argmax(1) for L in Ls], 0)
    agree = (agree == agree[0:1]).all(0).float()   # all committee members agree

    cflops = sum(flops_of(s.model, s.resolution, dev) for s in committee)
    fflops = flops_of(fM.model, fM.resolution, dev)

    print(f"\n#### DGC on {valdir}  (n={N}) ####")
    print(f"  committee={'+'.join(s.backbone for s in committee)} ({cflops/1e9:.2f} G), "
          f"fM={fM.backbone} ({fflops/1e9:.2f} G), c_committee/c_fM={cflops/fflops:.2f}")
    print(f"  committee acc={committee_c.float().mean():.4f}  fM acc={fm_c.float().mean():.4f}")

    e = ~committee_c
    dis = agree == 0
    recov = (e & fm_c)
    print(f"  committee errors={int(e.sum())}; recoverable (fM fixes)={recov.float().sum()/e.sum():.3f}")
    ed, ea = e & dis, e & (~dis)
    print(f"  P(fM fixes | error & DISAGREE)={fm_c[ed].float().mean():.3f} (n={int(ed.sum())})  "
          f"| AGREE={fm_c[ea].float().mean():.3f} (n={int(ea.sum())})")
    lab = fm_c[e].long()
    print(f"  among-errors AUC (predict fM fixes):  JS={auc_bin(js[e], lab):.4f}  "
          f"disagree={auc_bin((1-agree)[e], lab):.4f}  conf={auc_bin(-conf_c[e], lab):.4f}")

    # Frontier: accuracy vs escalation rate for each gate (identical cost at each e).
    gates = {
        "DGC (JS disagreement)": js,
        "DGC (1-agree)":         1 - agree,
        "confidence gate":       -conf_c,             # escalate lowest confidence
        "oracle gate":           recov.float(),       # escalate recoverable first
        "random":                torch.rand(N, generator=torch.Generator().manual_seed(0)),
    }
    curves = {k: acc_curve(committee_c, fm_c, s) for k, s in gates.items()}
    print(f"\n  accuracy at matched escalation rate e (cost = {cflops/fflops:.2f}x + e*1.00x of fM):")
    header = "    e     " + "".join(f"{k.split('(')[0].strip()[:12]:>14}" for k in gates)
    print(header)
    for e_t in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]:
        row = f"   {e_t:4.2f}  "
        for k in gates:
            a = min((a for ee, a in curves[k] if ee >= e_t - 1e-9), default=curves[k][-1][1])
            # pick the curve point nearest e_t
            a = min(curves[k], key=lambda p: abs(p[0] - e_t))[1]
            row += f"{a:>14.4f}"
        print(row)
    print(f"\n  cost at e: avg_xfM = {cflops/fflops:.2f} + e ; (committee always runs)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="cfgs/imagewoof_diverse.yaml")
    ap.add_argument("--valdir", default="data/imagewoof2/val")
    ap.add_argument("--corruption", default="")
    ap.add_argument("--severity", type=int, default=3)
    a = ap.parse_args()
    CORRUPTION = a.corruption
    SEVERITY = a.severity
    main(a.cfg, a.valdir)
