"""Cost-aware oracle Pareto frontiers: select-oracle vs action-oracle.

The oracle has foreknowledge, so each action costs only the models it actually
runs (standalone costs):

    return f1            c1
    return f2            c2
    return f3            c3
    ensemble f1+f2       c1+c2
    ensemble f1+f3       c1+c3
    ensemble f2+f3       c2+c3
    ensemble f1+f2+f3    c1+c2+c3   (logit-average of members)

select-oracle uses singles only; action-oracle adds the ensembles. For each, we
trace the exact accuracy-vs-cost Pareto frontier by a Lagrangian sweep: for cost
penalty lambda, every example picks argmax_a (correct_a - lambda * cost_a). The
meaningful claim is FRONTIER DOMINATION -- at matched cost budgets, does adding
ensemble actions strictly lift achievable accuracy?

FLOPs via torch FlopCounterMode (counts attention; fvcore undercounts ViT).
Run:  python oracle_frontier.py --cfg cfgs/imagewoof_diverse.yaml
"""

import argparse
import itertools
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset

from cascade.stage import build_cascade
from utils import ResizedDataset, get_dataset, resize_transform


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


def pareto(points):
    """Upper-left envelope: keep (cost, acc) not dominated by a cheaper-or-equal,
    higher-or-equal point."""
    pts = sorted(set(points))
    out, best = [], -1.0
    for c, a in pts:               # ascending cost; keep strictly increasing acc
        if a > best + 1e-9:
            out.append((c, a)); best = a
    return out


def frontier(correct, cost, cols):
    C = correct[:, cols].float()                       # [N, a]
    K = cost[cols]                                      # [a]
    N = C.shape[0]
    lams = [0.0] + torch.logspace(-5, 1, 400).tolist()
    pts = []
    for lam in lams:
        best = (C - lam * K[None, :]).argmax(1)
        pts.append((float(K[best].mean()), float(C[torch.arange(N), best].mean())))
    return pareto(pts)


def acc_at_budget(frontier_pts, budget):
    ok = [a for c, a in frontier_pts if c <= budget + 1e-9]
    return max(ok) if ok else float("nan")


def main(cfg_path):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    nw = cfg.get("num_workers", 8)
    ds = get_dataset(cfg); n = len(ds); nc = len(ds.classes)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0)).tolist()
    val = Subset(ds, perm[int(.7 * n):int(.8 * n)])
    cascade = build_cascade(cfg, nc, Subset(ds, range(1)), Subset(ds, range(1)))
    m = len(cascade)

    cstage = [flops_of(s.model, s.resolution, dev) for s in cascade]
    Ls, y = [], None
    for s in cascade:
        L, y = stage_logits(s.model, val, s.resolution, dev, 128, nw)
        Ls.append(L)

    # Enumerate actions: every non-empty subset of stages (logit-avg of members).
    actions = []
    for r in range(1, m + 1):
        for combo in itertools.combinations(range(m), r):
            name = "+".join(f"f{i+1}" for i in combo)
            logit = sum(Ls[i] for i in combo)
            correct = (logit.argmax(1) == y)
            cost = sum(cstage[i] for i in combo)
            actions.append((name, combo, correct, cost))

    names = [a[0] for a in actions]
    correct = torch.stack([a[2] for a in actions], dim=1)          # [N, A]
    cost = torch.tensor([a[3] for a in actions])
    f3 = cstage[-1]

    singles = [i for i, a in enumerate(actions) if len(a[1]) == 1]
    allacts = list(range(len(actions)))

    fr_sel = frontier(correct, cost, singles)
    fr_act = frontier(correct, cost, allacts)

    print(f"\nstage GFLOPs: " + ", ".join(f"f{i+1}={cstage[i]/1e9:.2f}" for i in range(m)))
    print("per-action standalone acc @ cost(xf3):")
    for nm, combo, cr, cst in actions:
        print(f"  {nm:<10} acc={cr.float().mean():.4f}  {cst/1e9:6.2f} G  {cst/f3:.2f}x")

    print("\nselect-oracle vs action-oracle accuracy at matched cost budgets:")
    print(f"{'budget xf3':>11}{'select':>9}{'action':>9}{'gain':>8}")
    for b in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.00]:
        s = acc_at_budget(fr_sel, b * f3)
        a = acc_at_budget(fr_act, b * f3)
        g = (a - s) if (s == s and a == a) else float("nan")
        print(f"{b:>11.2f}{s:>9.4f}{a:>9.4f}{g:>+8.4f}")

    strict = max(((a - acc_at_budget(fr_sel, c)) for c, a in fr_act
                  if acc_at_budget(fr_sel, c) == acc_at_budget(fr_sel, c)), default=0.0)
    print(f"\nmax strict accuracy lift of action-oracle over select-oracle: {strict:+.4f}")
    print("select frontier:", [(round(c/f3, 2), round(a, 4)) for c, a in fr_sel])
    print("action frontier:", [(round(c/f3, 2), round(a, 4)) for c, a in fr_act])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="cfgs/imagewoof_diverse.yaml")
    main(ap.parse_args().cfg)
