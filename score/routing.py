"""Test-time routing through the cascade, and routing evaluation metrics.

Routing rule (per input x), using the probe's safety estimates s_k(x):

    run probe g_phi(x) -> s_1(x), ..., s_{M-1}(x)
    if   s_1(x) >= tau_1: choose f1
    elif s_2(x) >= tau_2: choose f2
    ...
    else:                 choose fM   (the expensive oracle)

s_k(x) = sigmoid(route_logit_k). Higher means the cheap route k is estimated
safe. Stages are indexed 0..M-1; fM is the last (index M-1).
"""

import torch
from fvcore.nn import FlopCountAnalysis
from scipy.stats import beta
from torch.utils.data import DataLoader

from score.harm import predict_stage
from utils import ResizedDataset, resize_transform


@torch.no_grad()
def probe_safety(probe, data, resolution, device, batch_size, num_workers=0):
    """Run the probe over `data` and return safety scores [N, M-1].

    Order is preserved (shuffle=False) so rows align with stage predictions.
    device=None resolves to CUDA when available, else CPU.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    loader = DataLoader(
        ResizedDataset(data, resize_transform(resolution)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    probe = probe.to(device).eval()
    out = []
    for x, _ in loader:
        harm_logits = probe(x.to(device))["harm"]
        out.append(torch.sigmoid(-harm_logits).cpu())   # safety = P(not harmful)
    return torch.cat(out)


def binomial_ucb(num_harmful, n, delta):
    """One-sided Clopper-Pearson upper confidence bound on the harm probability.

    Returns the (1-delta) upper bound for `num_harmful` harmful routings out of
    `n` accepted examples. Small/empty accepted sets are penalized (bound -> 1),
    which makes the calibrated threshold conservative when little data passes.
    """
    if n == 0:
        return 1.0
    if num_harmful >= n:
        return 1.0
    return float(beta.ppf(1.0 - delta, num_harmful + 1, n - num_harmful))


def _calibrate_one(safety_k, harm_k, epsilon, delta):
    """Pick the least conservative (lowest) threshold whose harm UCB <= epsilon.

    The UCB is high at BOTH ends of the threshold range: at high tau the accepted
    set is tiny (wide confidence interval) and at low tau harmful examples start
    entering. The safe region is a contiguous band in between. We scan candidate
    thresholds (observed safety values) from high to low, skip the initial
    small-sample unsafe region, then walk down through the safe band and stop at
    the first breach after entering it. The chosen tau is the band's lower edge
    (most aggressive / accepts the most while staying safe).

    Returns (tau, diagnostics). If no threshold is ever safe, tau = inf so the
    route accepts nothing and inputs fall through to the next route / fM.
    """
    candidates = sorted(set(safety_k.tolist()), reverse=True)
    chosen = float("inf")
    diag = {"tau": chosen, "n_accepted": 0, "frac": 0.0, "emp_risk": 0.0, "ucb": 1.0}
    entered_safe = False

    for tau in candidates:
        accepted = safety_k >= tau
        n = int(accepted.sum().item())
        h = int(harm_k[accepted].sum().item())
        ucb = binomial_ucb(h, n, delta)
        if ucb <= epsilon:
            entered_safe = True
            chosen = tau
            diag = {
                "tau": tau, "n_accepted": n, "frac": n / len(safety_k),
                "emp_risk": h / n, "ucb": ucb,
            }
        elif entered_safe:
            break  # left the safe band on the low (harmful) side
        # else: still in the high-tau small-sample region; keep scanning down
    return chosen, diag


def calibrate_thresholds(safety, harm, epsilon, delta):
    """Sequentially calibrate one threshold per cheap route.

    Each route k is calibrated only on the examples that did NOT pass any earlier
    threshold (the test-time fall-through pool), so the thresholds compose with
    the cascade decision rule. Uses harm labels L_k = H[:, k] (1 = routing to
    f_k loses a correction fM would have made).

    Args:
        safety: FloatTensor [N, M-1] — probe safety s_k on the calibration set.
        harm:   LongTensor  [N, M-1] — harm labels L_k on the same set.
        epsilon: target harmful-routing risk tolerance.
        delta:   1 - delta is the confidence level of the upper bound.

    Returns:
        (thresholds, info) — list of M-1 thresholds and per-route diagnostics.
    """
    n, k_cheap = safety.shape
    pool = torch.ones(n, dtype=torch.bool)   # B_1 = all calibration examples
    thresholds, info = [], []
    for k in range(k_cheap):
        idx = pool.nonzero(as_tuple=True)[0]
        tau, diag = _calibrate_one(safety[idx, k], harm[idx, k], epsilon, delta)
        diag["pool"] = int(pool.sum().item())
        thresholds.append(tau)
        info.append(diag)
        # examples passing this threshold are routed to f_k and leave the pool
        pool = pool & ~(safety[:, k] >= tau)
    return thresholds, info


def route_decisions(safety, thresholds):
    """Apply the cascade decision rule.

    Args:
        safety:     FloatTensor [N, M-1] — s_k per example.
        thresholds: sequence of M-1 floats — tau_k per cheap route.

    Returns:
        LongTensor [N] — chosen stage index in 0..M-1 (M-1 == fM fallback).
    """
    n, k_cheap = safety.shape
    chosen = torch.full((n,), k_cheap, dtype=torch.long)  # default: fM (index M-1)
    assigned = torch.zeros(n, dtype=torch.bool)
    for k in range(k_cheap):
        take = (~assigned) & (safety[:, k] >= thresholds[k])
        chosen[take] = k
        assigned |= take
    return chosen


@torch.no_grad()
def count_flops(model, resolution, device):
    """Per-image forward FLOPs of `model` at the given square resolution.

    Measured on the underlying module (unwrapped from DataParallel): FLOPs are
    an architecture property, and fvcore cannot trace DataParallel's
    scatter/gather wrapper.
    """
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model = model.to(device).eval()
    dummy = torch.zeros(1, 3, resolution, resolution, device=device)
    return float(FlopCountAnalysis(model, dummy).total())


def normalize_thresholds(thresholds, k_cheap):
    """Accept a scalar (broadcast) or a length-(M-1) sequence."""
    if isinstance(thresholds, (int, float)):
        return [float(thresholds)] * k_cheap
    thresholds = list(thresholds)
    if len(thresholds) != k_cheap:
        raise ValueError(
            f"expected {k_cheap} thresholds (one per cheap route), got {len(thresholds)}"
        )
    return [float(t) for t in thresholds]


def evaluate_routing(probe, cascade, eval_data, thresholds, probe_resolution,
                     device=None, batch_size=128, num_workers=0):
    """Evaluate the routed cascade against the always-fM reference.

    Returns a dict of metrics: routed/always-fM Top-1 accuracy, accuracy drop,
    average per-image FLOPs for each, FLOPs ratio, and the routing distribution.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    m = len(cascade)
    k_cheap = m - 1
    thresholds = normalize_thresholds(thresholds, k_cheap)

    # Per-stage predictions on the eval set (rows aligned across stages).
    per_stage_preds, labels = [], None
    for stage in cascade:
        preds, labels = predict_stage(
            stage.model, eval_data, stage.resolution, device, batch_size, num_workers
        )
        per_stage_preds.append(preds)
    preds = torch.stack(per_stage_preds)                  # [M, N]
    n = labels.numel()

    # Probe safety -> routing decision.
    safety = probe_safety(probe, eval_data, probe_resolution, device, batch_size, num_workers)
    chosen = route_decisions(safety, thresholds)          # [N]

    # Routed system: prediction of the chosen stage for each example.
    routed_pred = preds[chosen, torch.arange(n)]
    routed_acc = (routed_pred == labels).float().mean().item()

    # Reference: always run fM.
    fm_acc = (preds[-1] == labels).float().mean().item()

    # FLOPs per image: probe always runs; add the chosen stage's cost.
    stage_flops = [count_flops(s.model, s.resolution, device) for s in cascade]
    probe_flops = count_flops(probe, probe_resolution, device)
    chosen_flops = torch.tensor([stage_flops[c] for c in chosen.tolist()])
    routed_flops = probe_flops + chosen_flops.mean().item()
    fm_flops = stage_flops[-1]

    distribution = {
        cascade[k].name: int((chosen == k).sum().item()) / n for k in range(m)
    }

    return {
        "n": n,
        "routed_acc": routed_acc,
        "fm_acc": fm_acc,
        "acc_drop": fm_acc - routed_acc,
        "routed_flops": routed_flops,
        "fm_flops": fm_flops,
        "flops_ratio": routed_flops / fm_flops if fm_flops else float("nan"),
        "probe_flops": probe_flops,
        "stage_flops": {cascade[k].name: stage_flops[k] for k in range(m)},
        "distribution": distribution,
        "thresholds": thresholds,
    }
