"""Sweep the multi-task router's loss weights (lambdas) on Imagenette.

Backbones are cached, so the expensive per-stage signals (router targets on
probe_calib, harm + per-stage preds on thresh_calib/val) are computed ONCE; each
lambda setting is just a fast router retrain + strategy-A evaluation.

For each lambda set we report: held-out route-AUC (on thresh_calib), and the
one-shot router (strategy A) frontier at epsilon 0.05 and 0.10 (FLOPs ratio,
cheap-routed %, accuracy, realized harm).

Run:  python sweep_lambdas.py --cfg cfgs/imagenette.yaml
"""

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import Subset

from cascade.stage import build_cascade, run_dir
from models.probe import Probe, train_probe, _route_auc
from score.harm import (
    compute_harm_labels,
    compute_router_targets,
    harm_labels_from_predictions,
    predict_stage,
)
from score.routing import calibrate_thresholds, count_flops, probe_safety, route_decisions
from utils import get_dataset

# Hypothesis: down-weight conf-distill + oracle, up-weight ranking + lost-correction.
LAMBDA_SETS = {
    "baseline_all1":   {"distill": 1.0, "correctness": 1.0, "lost_correction": 1.0, "oracle_route": 1.0, "ranking": 1.0},
    "distill_off":     {"distill": 0.0, "correctness": 1.0, "lost_correction": 1.0, "oracle_route": 0.0, "ranking": 1.0},
    "rank_heavy":      {"distill": 0.2, "correctness": 1.0, "lost_correction": 2.0, "oracle_route": 0.5, "ranking": 4.0},
    "rank_dominant":   {"distill": 0.0, "correctness": 0.5, "lost_correction": 4.0, "oracle_route": 0.0, "ranking": 8.0},
    "harm_rank_only":  {"distill": 0.0, "correctness": 0.0, "lost_correction": 4.0, "oracle_route": 0.0, "ranking": 4.0},
    "balanced_light":  {"distill": 0.5, "correctness": 1.0, "lost_correction": 2.0, "oracle_route": 0.5, "ranking": 2.0},
}
EPSILONS = [0.05, 0.10]


def main(cfg_path):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    nw = cfg.get("num_workers", 8)
    pcfg = cfg["probe"]; pr = pcfg["resolution"]

    ds = get_dataset(cfg); n = len(ds); nc = len(ds.classes)
    i2, i3 = int(.80 * n), int(.90 * n)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0)).tolist()
    probe_calib = Subset(ds, perm[i2:i3])
    thresh = Subset(ds, perm[i3:])
    val = Subset(ds, perm[int(.70 * n):int(.80 * n)])

    cascade = build_cascade(cfg, nc, Subset(ds, range(1)), Subset(ds, range(1)))
    m = len(cascade)

    print("precomputing stage signals (once)...")
    targets = compute_router_targets(cascade, probe_calib, num_workers=nw)        # router training
    H_thr, _ = compute_harm_labels(cascade, thresh, num_workers=nw)               # calibration harm
    val_preds, labels = [], None
    for s in cascade:
        p, labels = predict_stage(s.model, val, s.resolution, dev, 128, nw)
        val_preds.append(p)
    val_preds = torch.stack(val_preds)                                            # [M, Nv]
    H_val = harm_labels_from_predictions(val_preds, labels)
    nval = labels.numel()

    flops = torch.tensor([count_flops(s.model, s.resolution, dev) for s in cascade])
    pf = count_flops(Probe(nc, m, pr, pcfg["z_dim"]), pr, dev)

    def eval_A(probe):
        s_thr = probe_safety(probe, thresh, pr, dev, 128, nw)
        s_val = probe_safety(probe, val, pr, dev, 128, nw)
        auc = _route_auc(s_thr, H_thr)
        rows = {}
        for eps in EPSILONS:
            tau, _ = calibrate_thresholds(s_thr, H_thr, eps, cfg["routing"]["delta"])
            chosen = route_decisions(s_val, tau)
            cost = (pf + flops[chosen].float()).mean().item()
            acc = (val_preds[chosen, torch.arange(nval)] == labels).float().mean().item()
            cheap = chosen < (m - 1)
            harm = H_val[torch.arange(nval)[cheap], chosen[cheap]].float().mean().item() if cheap.any() else 0.0
            dist = {cascade[k].name: (chosen == k).float().mean().item() for k in range(m)}
            rows[eps] = (cost / flops[-1].item(), cheap.float().mean().item(), acc, harm, dist)
        return auc, rows

    fm_acc = (val_preds[-1] == labels).float().mean().item()
    print(f"\nf3(fM) acc={fm_acc:.4f}  | sweeping {len(LAMBDA_SETS)} lambda sets "
          f"(epochs={pcfg['epochs']})\n")

    results = {}
    for name, lam in LAMBDA_SETS.items():
        torch.manual_seed(0)                                  # same init across settings
        probe = Probe(nc, m, pr, pcfg["z_dim"])
        print(f"==== {name}: {lam} ====")
        probe = train_probe(probe, probe_calib, targets, pr, dev,
                            epochs=pcfg["epochs"], lr=pcfg["lr"], batch_size=pcfg["batch_size"],
                            weight_decay=pcfg.get("weight_decay", 0.0), lambdas=lam, num_workers=nw)
        results[name] = eval_A(probe)

    print("\n================== LAMBDA SWEEP (strategy A, one-shot router) ==================")
    print(f"{'lambda set':<16}{'AUC':>7}   "
          f"{'A@.05 xf3/cheap/acc/harm':<30}{'A@.10 xf3/cheap/acc/harm':<30}")
    for name, (auc, rows) in results.items():
        def fmt(e):
            x, ch, acc, h, _ = rows[e]
            return f"{x:.2f}x {ch*100:3.0f}% {acc:.3f} h{h:.3f}"
        print(f"{name:<16}{auc:>7.4f}   {fmt(0.05):<30}{fmt(0.10):<30}")
    print(f"\nreference: always-f3 acc={fm_acc:.4f} @1.00x ; f1/f2/f3 flops="
          f"{[round(f/1e9,2) for f in flops.tolist()]} G ; probe={pf/1e9:.4f} G")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="cfgs/imagenette.yaml")
    main(ap.parse_args().cfg)
