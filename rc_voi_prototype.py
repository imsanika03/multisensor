"""RC-VoI cascade -- kernel prototype on cached stage outputs (Imagenette).

Tests whether an expanded action space {return f1, run f2, skip->f3, return f2,
ensemble(f1,f2), run f3}, driven by real model-state evidence (z1 after f1, z12
after f2) via a value-of-information lookahead, beats the plain confidence cascade
(B) and recovers part of the oracle gap that pure stage-selection cannot.

Policy (2-step expectimax, predictors estimate P(action correct | state)):
  at z1:  max[ return f1 ; run f2 then act ; skip->f3 ]
  at z12: max[ return f1 ; return f2 ; ensemble(f1,f2) ; run f3 ]
each scored as  P(correct) - lambda * total_FLOPs.  Sweeping lambda traces the
cost/accuracy frontier; predictors drive DECISIONS, evaluation uses ground truth.

Ablations: --no-ensemble, --no-skip (handled inline; both reported).
Run:  python rc_voi_prototype.py --cfg cfgs/imagenette.yaml
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset

from cascade.stage import build_cascade
from score.routing import count_flops
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


def _stats(logits):
    p = torch.softmax(logits, dim=1)
    top = p.topk(min(5, p.shape[1]), dim=1)
    conf = top.values[:, 0]
    margin = top.values[:, 0] - top.values[:, 1]
    ent = -(p * p.clamp_min(1e-12).log()).sum(1) / torch.log(torch.tensor(float(p.shape[1])))
    return p, conf, margin, ent, top.indices


def build_states(L1, L2, L3, y):
    """Return (z1, z12, correctness) where correctness has columns [f1,f2,ens,f3]."""
    p1, c1, m1, e1, t1 = _stats(L1)
    p2, c2, m2, e2, t2 = _stats(L2)
    Lens = L1 + L2
    pe, ce, me, ee, te = _stats(Lens)

    arg1, arg2, arge, arg3 = L1.argmax(1), L2.argmax(1), Lens.argmax(1), L3.argmax(1)
    correct = torch.stack([(arg1 == y), (arg2 == y), (arge == y), (arg3 == y)], dim=1).float()

    z1 = torch.stack([c1, m1, e1], dim=1)
    z1 = torch.cat([z1, p1.topk(min(5, p1.shape[1]), dim=1).values], dim=1)

    agree = (arg1 == arg2).float()
    overlap = torch.tensor([len(set(a.tolist()) & set(b.tolist())) / t1.shape[1]
                            for a, b in zip(t1, t2)])
    z12 = torch.stack([c1, c2, m1, m2, e1, e2, agree, overlap,
                       e1 - e2, m2 - m1, ce, me, (arge == arg1).float()], dim=1)
    return z1, z12, correct


class CorrNet(nn.Module):
    def __init__(self, d, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(hidden, 4))   # P(correct) for [f1,f2,ens,f3]

    def forward(self, x):
        return self.net(x)


def fit(net, z, corr, device, steps=400, lr=1e-3):
    z, corr = z.to(device), corr.to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-3)
    bce = nn.BCEWithLogitsLoss()
    net.to(device).train()
    for _ in range(steps):
        opt.zero_grad(); loss = bce(net(z), corr); loss.backward(); opt.step()
    return net.eval()


def rollout(p1, q12, corr, costs, lam, use_ens, use_skip):
    """Vectorized 2-step policy. p1/q12: predicted P(correct)[N,4] from z1/z12.
    Returns (mean_cost, accuracy, action_counts). Real correctness from `corr`."""
    c1, c2, c3 = costs
    N = corr.shape[0]
    F1, F2, ENS, F3 = 0, 1, 2, 3
    neg = -1e9

    # z1 utilities
    u_ret1 = p1[:, F1] - lam * c1
    post_f2 = torch.stack([p1[:, F1], p1[:, F2],
                           p1[:, ENS] if use_ens else torch.full((N,), neg)], dim=1).max(1).values
    u_runf2 = post_f2 - lam * (c1 + c2)
    u_skip = (p1[:, F3] - lam * (c1 + c3)) if use_skip else torch.full((N,), neg)

    z1_choice = torch.stack([u_ret1, u_runf2, u_skip], dim=1).argmax(1)  # 0 stop,1 runf2,2 skip

    cost = torch.empty(N); pick = torch.empty(N, dtype=torch.long); acts = {}
    # skip -> f3
    sk = z1_choice == 2
    cost[sk] = c1 + c3; pick[sk] = F3; acts["skip->f3"] = int(sk.sum())
    # stop at f1
    st = z1_choice == 0
    cost[st] = c1; pick[st] = F1; acts["return f1"] = int(st.sum())
    # run f2 -> z12 decision
    rf = z1_choice == 1
    opts = [q12[:, F1], q12[:, F2]]
    names = [F1, F2]
    if use_ens:
        opts.append(q12[:, ENS]); names.append(ENS)
    opts.append(q12[:, F3] - lam * c3)            # run f3 (extra cost c3)
    names.append(F3)
    U = torch.stack(opts, dim=1)
    sub = U.argmax(1)
    chosen = torch.tensor(names)[sub]
    cost_rf = torch.where(chosen == F3, torch.tensor(c1 + c2 + c3), torch.tensor(c1 + c2))
    cost[rf] = cost_rf[rf]; pick[rf] = chosen[rf]
    for nm, idx in [("f2->return f1", F1), ("f2->return f2", F2), ("f2->ensemble", ENS), ("f2->run f3", F3)]:
        acts[nm] = int((rf & (chosen == idx)).sum())

    acc = corr[torch.arange(N), pick].mean().item()
    return cost.mean().item(), acc, acts


def frontier(p1, q12, corr, costs, use_ens, use_skip):
    lams = [0.0] + torch.logspace(-4, 0, 30).tolist()
    pts = []
    for lam in lams:
        c, a, _ = rollout(p1, q12, corr, costs, lam, use_ens, use_skip)
        pts.append((c, a))
    # Pareto: cheapest cost achieving >= each accuracy
    return pts


def conf_cascade_frontier(L1, L2, L3, y, costs):
    """Plain confidence cascade B: f1 if conf>=t else f2 if conf>=t else f3."""
    c1, c2, c3 = costs
    _, cf1, _, _, _ = _stats(L1); _, cf2, _, _, _ = _stats(L2)
    a1, a2, a3 = (L1.argmax(1) == y), (L2.argmax(1) == y), (L3.argmax(1) == y)
    pts = []
    for t in torch.linspace(0, 1, 51).tolist():
        at_f1 = cf1 >= t
        at_f2 = (~at_f1) & (cf2 >= t)
        at_f3 = ~(at_f1 | at_f2)
        cost = (at_f1 * c1 + at_f2 * (c1 + c2) + at_f3 * (c1 + c2 + c3)).float().mean().item()
        acc = (at_f1 & a1).float().sum() + (at_f2 & a2).float().sum() + (at_f3 & a3).float().sum()
        pts.append((cost, (acc / len(y)).item()))
    return pts


def min_cost_for(pts, target, f3):
    ok = [c for c, a in pts if a >= target]
    return min(ok) / f3 if ok else float("nan")


def main(cfg_path):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    nw = cfg.get("num_workers", 8)
    ds = get_dataset(cfg); n = len(ds); nc = len(ds.classes)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0)).tolist()
    cal = Subset(ds, perm[int(.9 * n):])              # calibration (train predictors)
    val = Subset(ds, perm[int(.7 * n):int(.8 * n)])   # evaluation

    cascade = build_cascade(cfg, nc, Subset(ds, range(1)), Subset(ds, range(1)))
    costs = [count_flops(s.model, s.resolution, dev) for s in cascade]
    f3 = costs[-1]

    def states_for(split):
        Ls = [stage_logits(s.model, split, s.resolution, dev, 128, nw) for s in cascade]
        y = Ls[0][1]
        return build_states(Ls[0][0], Ls[1][0], Ls[2][0], y) + (y,)

    z1c, z12c, corrc, _ = states_for(cal)
    z1v, z12v, corrv, yv = states_for(val)

    torch.manual_seed(0)
    net1 = fit(CorrNet(z1c.shape[1]), z1c, corrc, dev)
    net12 = fit(CorrNet(z12c.shape[1]), z12c, corrc, dev)
    with torch.no_grad():
        p1 = torch.sigmoid(net1(z1v.to(dev))).cpu()
        q12 = torch.sigmoid(net12(z12v.to(dev))).cpu()

    fm = corrv[:, 3].mean().item()
    # action oracle: cheapest correct action; select oracle: cheapest correct STAGE
    act_cost = torch.tensor([costs[0], costs[0] + costs[1], costs[0] + costs[1], costs[0] + costs[2]])
    oracle_act = torch.where(corrv.bool(), act_cost.unsqueeze(0), torch.tensor(1e18))
    oa_cost = oracle_act.min(1).values.mean().item() / f3
    oa_acc = (corrv.sum(1) > 0).float().mean().item()

    variants = {
        "RC-VoI (full)":      frontier(p1, q12, corrv, costs, True, True),
        "  no-ensemble":      frontier(p1, q12, corrv, costs, False, True),
        "  no-skip":          frontier(p1, q12, corrv, costs, True, False),
        "B (conf cascade)":   conf_cascade_frontier(*[stage_logits(s.model, val, s.resolution, dev, 128, nw)[0] for s in cascade], yv, costs),
    }

    print(f"\nf3(fM) acc={fm:.4f} @1.00x | f1/f2/f3 flops={[round(c/1e9,2) for c in costs]} G")
    print(f"action-oracle: acc={oa_acc:.4f} @ {oa_cost:.3f}x  (ceiling for this action set)\n")
    print(f"{'method':<20}{'minx@fm':>9}{'minx@fm-.5%':>12}{'minx@fm-1%':>11}{'minx@fm-2%':>11}")
    for name, pts in variants.items():
        row = [min_cost_for(pts, fm, f3), min_cost_for(pts, fm - .005, f3),
               min_cost_for(pts, fm - .01, f3), min_cost_for(pts, fm - .02, f3)]
        print(f"{name:<20}" + "".join(f"{v:>{w}.3f}" for v, w in zip(row, [9, 12, 11, 11])))

    # action mix at a mid operating point (lambda chosen to land near fm-1%)
    print("\naction mix (RC-VoI full, lambda=0.02):")
    _, a, acts = rollout(p1, q12, corrv, costs, 0.02, True, True)
    print(f"  acc={a:.4f}  " + " ".join(f"{k}={v}" for k, v in acts.items() if v))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="cfgs/imagenette.yaml")
    main(ap.parse_args().cfg)
