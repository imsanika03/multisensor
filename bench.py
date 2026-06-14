"""Benchmark routing strategies on a common cost / accuracy / risk frontier.

Strategies (all calibrated on thresh_calib, evaluated on val):

    A  Probe one-shot router      cost = g_phi + flops[chosen]
    B  Confidence cascade         cost = cumsum(flops[0..chosen])   (signal: max-softmax)
    C  Hybrid probe-then-conf     cost = g_phi + (flops[k] if probe certifies else cumsum)
    D  Always fM                  cost = flops[-1]
    E  Always f1 / always f2      cost = flops[0] / flops[1]
    F  Oracle cheapest-correct    cost = flops[cheapest stage that is correct, else fM]

A, B and C share the SAME calibration machinery (calibrate_thresholds + binomial
UCB at epsilon); only the decision signal and cost model differ. Realized harm on
the cheap-routed examples is reported so the actual risk is visible, not assumed.

Run from the project dir:  python bench.py --cfg cfgs/resolution_cascade.yaml
"""

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Subset

from cascade.stage import build_cascade, run_dir
from models.probe import Probe
from score.harm import harm_labels_from_predictions
from score.routing import (
    calibrate_thresholds,
    count_flops,
    probe_safety,
    route_decisions,
)
from utils import ResizedDataset, get_dataset, resize_transform

EPSILONS = [0.05, 0.10]


@torch.no_grad()
def stage_outputs(model, data, resolution, device, batch_size, num_workers):
    """Return (preds [N], max_softmax_conf [N], labels [N]) for one stage."""
    loader = DataLoader(
        ResizedDataset(data, resize_transform(resolution)),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    model = model.to(device).eval()
    preds, conf, labels = [], [], []
    for x, y in loader:
        probs = torch.softmax(model(x.to(device)), dim=1)
        mp, pred = probs.max(dim=1)
        preds.append(pred.cpu()); conf.append(mp.cpu()); labels.append(y)
    return torch.cat(preds), torch.cat(conf), torch.cat(labels)


def gather(cascade, data, device, bs, nw):
    """Per-stage preds/conf over `data`. Returns preds[M,N], conf[M,N], labels[N]."""
    P, C, labels = [], [], None
    for s in cascade:
        p, c, labels = stage_outputs(s.model, data, s.resolution, device, bs, nw)
        P.append(p); C.append(c)
    return torch.stack(P), torch.stack(C), labels


def metrics(name, chosen, cost, preds, labels, flops, harm, fm_acc, dist=True):
    """Assemble a result row from per-example chosen-stage + cost."""
    n = labels.numel()
    routed_pred = preds[chosen, torch.arange(n)]
    acc = (routed_pred == labels).float().mean().item()
    cheap = chosen < (preds.shape[0] - 1)
    realized_harm = harm[torch.arange(n)[cheap], chosen[cheap]].float().mean().item() if cheap.any() else 0.0
    row = {
        "name": name, "acc": acc, "acc_drop": fm_acc - acc,
        "gflops": cost.mean().item() / 1e9,
        "xf3": cost.mean().item() / flops[-1],
        "cheap_frac": cheap.float().mean().item(),
        "harm_routed": realized_harm,
    }
    if dist:
        row["dist"] = {k: (chosen == k).float().mean().item() for k in range(preds.shape[0])}
    return row


def main(cfg_path):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bs, nw = 128, cfg.get("num_workers", 0)

    ds = get_dataset(cfg); n = len(ds); nc = len(ds.classes)
    # Must match main.py's shuffled split exactly (same seed) so the probe is
    # calibrated/evaluated on the same held-out rows it was during training.
    i1, i2, i3 = int(.70 * n), int(.80 * n), int(.90 * n)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0)).tolist()
    val = Subset(ds, perm[i1:i2])              # evaluation split (as in main.py)
    thresh = Subset(ds, perm[i3:n])            # held-out calibration split

    cascade = build_cascade(cfg, nc, Subset(ds, range(1)), Subset(ds, range(1)))
    m = len(cascade)
    pr = cfg["probe"]["resolution"]
    probe = Probe(nc, m, pr, cfg["probe"]["z_dim"])
    probe.load_state_dict(torch.load(
        run_dir(cfg["checkpoint_dir"], cfg["cascade_id"]) / "probe.pt", map_location="cpu"))

    # FLOPs: per stage + probe; cumulative for the sequential cascade.
    flops = [count_flops(s.model, s.resolution, device) for s in cascade]
    pf = count_flops(probe, pr, device)
    cum = torch.tensor(flops).cumsum(0)        # cum[j] = sum flops[0..j]
    flops_t = torch.tensor(flops)

    # Calibration-split signals.
    Pc, Cc, yc = gather(cascade, thresh, device, bs, nw)
    Hc = harm_labels_from_predictions(Pc, yc)              # [Nc, M-1]
    Sc = probe_safety(probe, thresh, pr, device, bs, nw)   # [Nc, M-1]
    conf_cheap_c = Cc[:m - 1].t().contiguous()             # [Nc, M-1] f1..f_{M-1} confidence

    # Eval-split signals.
    Pv, Cv, yv = gather(cascade, val, device, bs, nw)
    Hv = harm_labels_from_predictions(Pv, yv)
    Sv = probe_safety(probe, val, pr, device, bs, nw)
    conf_cheap_v = Cv[:m - 1].t().contiguous()
    nval = yv.numel()
    fm_acc = (Pv[-1] == yv).float().mean().item()
    delta = cfg["routing"]["delta"]

    # ---- ε-independent references (D, E, F) ----
    print(f"\n=== references (n={nval}, f3={flops[-1]/1e9:.2f} G) ===")
    fixed = []
    for k in range(m):
        ch = torch.full((nval,), k)
        cost = flops_t[ch].float()
        fixed.append(metrics(f"Always {cascade[k].name}", ch, cost, Pv, yv, flops, Hv, fm_acc, dist=False))
    # Oracle: cheapest correct stage; if none correct, fM (still wrong).
    correct = (Pv == yv.unsqueeze(0))                      # [M, nval]
    oracle_choice = torch.full((nval,), m - 1)
    for k in range(m - 1, -1, -1):                         # cheapest wins (overwrite downward)
        oracle_choice[correct[k]] = k
    oracle_cost = flops_t[oracle_choice].float()
    oracle = metrics("Oracle cheapest", oracle_choice, oracle_cost, Pv, yv, flops, Hv, fm_acc, dist=False)
    for r in fixed + [oracle]:
        print(f"  {r['name']:<16} acc={r['acc']:.4f}  drop={r['acc_drop']:+.4f}  "
              f"{r['gflops']:6.2f} G  {r['xf3']:.2f}x")

    # ---- calibrated strategies (A, B, C) at each epsilon ----
    for eps in EPSILONS:
        tau_p, _ = calibrate_thresholds(Sc, Hc, eps, delta)            # probe
        tau_c, _ = calibrate_thresholds(conf_cheap_c, Hc, eps, delta)  # confidence

        # A: probe one-shot.
        chA = route_decisions(Sv, tau_p)
        costA = pf + flops_t[chA].float()
        rA = metrics("A probe one-shot", chA, costA, Pv, yv, flops, Hv, fm_acc)

        # B: confidence cascade (cumulative cost).
        chB = route_decisions(conf_cheap_v, tau_c)
        costB = cum[chB].float()
        rB = metrics("B conf cascade", chB, costB, Pv, yv, flops, Hv, fm_acc)

        # C: hybrid. Probe certifies -> jump (cost pf+flops[k]); else conf cascade (cost pf+cum).
        chC = chA.clone()
        costC = pf + flops_t[chA].float()
        fell = (chA == m - 1)                                # probe didn't certify
        chC[fell] = chB[fell]
        costC[fell] = pf + cum[chB[fell]].float()
        rC = metrics("C hybrid", chC, costC, Pv, yv, flops, Hv, fm_acc)

        print(f"\n=== calibrated strategies @ epsilon={eps} (conf={1-delta:.0%}) ===")
        for r in (rA, rB, rC):
            d = ", ".join(f"{cascade[k].name}={r['dist'][k]:.0%}" for k in range(m))
            print(f"  {r['name']:<18} acc={r['acc']:.4f}  drop={r['acc_drop']:+.4f}  "
                  f"{r['gflops']:6.2f} G  {r['xf3']:.2f}x  "
                  f"cheap={r['cheap_frac']:.0%} harm@routed={r['harm_routed']:.3f}  [{d}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    main(ap.parse_args().cfg)
