import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import Subset

from cascade.stage import build_cascade, run_dir
from models.probe import Probe, train_probe
from score.harm import compute_harm_labels, compute_router_targets
from score.routing import (
    calibrate_thresholds,
    evaluate_routing,
    probe_safety,
)
from utils import get_dataset


def main(cfg_path):
    cfg_path = Path(cfg_path)
    cfg = yaml.safe_load(cfg_path.read_text())

    ds = get_dataset(cfg)
    n = len(ds)
    num_classes = len(ds.classes)
    num_workers = cfg.get("num_workers", 0)

    # 4-way split, each role disjoint:
    #   train        70%  cascade weights
    #   val          10%  cascade early stopping  +  routing evaluation
    #   probe_calib  10%  train the probe (its harm labels)
    #   thresh_calib 10%  calibrate thresholds (probe never saw this -> honest UCB)
    # Shuffle first (fixed seed): some datasets (e.g. Imagenette/ImageFolder) are
    # ordered by class, which would make a sequential split non-iid.
    i1, i2, i3 = int(0.70 * n), int(0.80 * n), int(0.90 * n)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0)).tolist()
    train = Subset(ds, perm[:i1])
    val = Subset(ds, perm[i1:i2])
    probe_calib = Subset(ds, perm[i2:i3])
    thresh_calib = Subset(ds, perm[i3:n])

    # val is used for early-stopping/model selection during training.
    cascade = build_cascade(cfg, num_classes, train, val)

    for stage in cascade:
        print(f"{stage.name}: {stage.backbone} @ {stage.resolution}px "
              f"({num_classes} classes)")

    # Multi-task router targets on the probe-training split (confidence/correctness
    # /harm/oracle-route, per stage). harm[:, k] = 1 where cheap route f(k+1) is
    # wrong but fM is right.
    targets = compute_router_targets(cascade, probe_calib, num_workers=num_workers)
    for k, rate in enumerate(targets["harm"].float().mean(dim=0).tolist(), start=1):
        print(f"route f{k} harm rate (probe_calib): {rate:.4f}")

    # Build the multi-task one-shot router.
    probe_cfg = cfg.get("probe", {})
    probe_res = probe_cfg.get("resolution", 96)
    probe = Probe(
        num_classes=num_classes,
        num_routes=len(cascade),
        input_dim=probe_res,
        z_dim=probe_cfg.get("z_dim", 128),
    )

    # Probe checkpoint lives in the same per-run folder as the cascade.
    rdir = run_dir(cfg.get("checkpoint_dir", "checkpoints"),
                   cfg.get("cascade_id", "default"))
    probe_ckpt = rdir / "probe.pt"
    if probe_ckpt.exists():
        probe.load_state_dict(torch.load(probe_ckpt, map_location="cpu"))
        print(f"loaded probe from {probe_ckpt}")
    else:
        print("training probe")
        probe = train_probe(
            probe, probe_calib, targets, probe_res, device=None,
            epochs=probe_cfg.get("epochs", 10),
            lr=probe_cfg.get("lr", 1e-3),
            batch_size=probe_cfg.get("batch_size", 64),
            weight_decay=probe_cfg.get("weight_decay", 0.0),
            lambdas=probe_cfg.get("lambdas"),
            num_workers=num_workers,
        )
        rdir.mkdir(parents=True, exist_ok=True)
        torch.save(probe.state_dict(), probe_ckpt)
        print(f"saved probe to {probe_ckpt}")

    # Calibrate thresholds on a SEPARATE split the probe never trained on, so
    # the binomial UCB on harmful routing is honest. Pick the most aggressive
    # tau_k whose UCB stays <= epsilon.
    routing_cfg = cfg.get("routing", {})
    epsilon = routing_cfg.get("epsilon", 0.02)
    delta = routing_cfg.get("delta", 0.05)
    H_thresh, _ = compute_harm_labels(cascade, thresh_calib, num_workers=num_workers)
    safety_thresh = probe_safety(
        probe, thresh_calib, probe_res, device=None, batch_size=128, num_workers=num_workers
    )
    thresholds, info = calibrate_thresholds(safety_thresh, H_thresh, epsilon, delta)
    _print_calibration_table(cascade, thresholds, info, epsilon, delta)

    # Evaluate routing on val against the always-fM reference.
    metrics = evaluate_routing(
        probe, cascade, val, thresholds, probe_res, num_workers=num_workers
    )
    _print_routing_table(metrics)

    return cascade, probe, metrics


def _print_calibration_table(cascade, thresholds, info, epsilon, delta):
    print(f"\nthreshold calibration (epsilon={epsilon}, conf={1 - delta:.0%})")
    print(f"{'route':<8}{'tau':>8}{'pool':>7}{'routed':>8}"
          f"{'%pool':>8}{'emp risk':>10}{'UCB':>9}")
    for k, d in enumerate(info):
        name = cascade[k].name
        tau = "inf" if d["tau"] == float("inf") else f"{d['tau']:.3f}"
        print(f"{name:<8}{tau:>8}{d['pool']:>7}{d['n_accepted']:>8}"
              f"{d['frac']:>7.1%}{d['emp_risk']:>10.4f}{d['ucb']:>9.4f}")


def _print_routing_table(m):
    g = 1e9  # report FLOPs in GFLOPs
    print(f"\nrouting evaluation (n={m['n']}, thresholds={m['thresholds']})")
    dist = ", ".join(f"{name}={frac:.1%}" for name, frac in m["distribution"].items())
    print(f"routing distribution: {dist}")
    print(f"{'system':<16}{'top1 acc':>10}{'acc drop':>10}{'avg GFLOPs':>13}{'FLOPs vs fM':>13}")
    print(f"{'Always fM':<16}{m['fm_acc']:>10.4f}{0.0:>10.4f}"
          f"{m['fm_flops'] / g:>13.3f}{1.0:>12.2f}x")
    print(f"{'Routed (gphi)':<16}{m['routed_acc']:>10.4f}{m['acc_drop']:>10.4f}"
          f"{m['routed_flops'] / g:>13.3f}{m['flops_ratio']:>12.2f}x")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", required=True)
    args = p.parse_args()
    main(args.cfg)
