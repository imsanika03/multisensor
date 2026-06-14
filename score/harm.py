"""Calibration split and harm-label construction for probe training.

A cascade is an ordered list of stages [f1, ..., fM], cheap -> expensive. The
last stage fM is the expensive "oracle" we fall back to; the first M-1 stages
are the candidate cheap routes.

For every image x and every cheap route k:

    Hk(x) = 1  if  fk is wrong  and  fM is correct
    Hk(x) = 0  otherwise

i.e. Hk flags the examples where taking cheap route k would *harm* accuracy
relative to running the expensive model. These are the training targets for
the probe's route-safety head (M-1 logits).
"""

import math

import torch
from torch.utils.data import DataLoader, random_split

from utils import ResizedDataset, resize_transform


def split_calibration(ds, frac=0.2, seed=0):
    """Split a dataset into a calibration subset and the remainder.

    Returns (calibration, remainder). The split is reproducible for a given
    seed so the same calibration set can be reused across runs.
    """
    n = len(ds)
    n_cal = int(round(frac * n))
    generator = torch.Generator().manual_seed(seed)
    calibration, remainder = random_split(ds, [n_cal, n - n_cal], generator=generator)
    return calibration, remainder


@torch.no_grad()
def predict_stage(model, data, resolution, device, batch_size, num_workers=0):
    """Run one stage over `data` at its resolution; return (preds, labels).

    Order is preserved (shuffle=False) so predictions from different stages
    align by row.
    """
    loader = DataLoader(
        ResizedDataset(data, resize_transform(resolution)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    model = model.to(device).eval()
    preds, labels = [], []
    for x, y in loader:
        logits = model(x.to(device))
        preds.append(logits.argmax(dim=1).cpu())
        labels.append(y)
    return torch.cat(preds), torch.cat(labels)


def harm_labels_from_predictions(preds, labels):
    """Build harm labels Hk from per-stage predictions.

    Args:
        preds:  LongTensor [M, N] — predicted class per stage, per example.
        labels: LongTensor [N]    — ground-truth class.

    Returns:
        LongTensor [N, M-1] — Hk for each example and each cheap route k.
    """
    correct = preds == labels.unsqueeze(0)        # [M, N]
    correct_final = correct[-1]                   # [N]  -> fM
    cheap_correct = correct[:-1]                  # [M-1, N] -> f1..f_{M-1}
    harm = (~cheap_correct) & correct_final.unsqueeze(0)
    return harm.long().t()                        # [N, M-1]


def compute_harm_labels(cascade, calibration, device=None, batch_size=128, num_workers=0):
    """Run every cascade stage over the calibration set and build harm labels.

    Returns:
        H:      LongTensor [N, M-1] — harm label per example, per cheap route.
        labels: LongTensor [N]      — ground-truth class (probe class target).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    per_stage_preds = []
    labels = None
    for stage in cascade:
        preds, labels = predict_stage(
            stage.model, calibration, stage.resolution, device, batch_size, num_workers
        )
        per_stage_preds.append(preds)

    preds = torch.stack(per_stage_preds)          # [M, N]
    H = harm_labels_from_predictions(preds, labels)
    return H, labels


@torch.no_grad()
def compute_router_targets(cascade, data, device=None, batch_size=128, num_workers=0):
    """Per-example supervision for the multi-task one-shot router.

    Runs every stage over `data` (order preserved) and derives, for the K = M-1
    cheap stages, the targets the router distills/predicts:

        margin   [N, K]  top1-top2 softmax margin of f_k        (in [0,1])
        entropy  [N, K]  prediction entropy of f_k / log(C)     (in [0,1])
        maxsoft  [N, K]  max-softmax of f_k                     (in [0,1])
        correct  [N, K]  1 if f_k is correct                    (float 0/1)
        harm     [N, K]  1 if f_k wrong and fM right            (float 0/1)
        oracle   [N]     cheapest correct stage index, else M-1 (long)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    m = len(cascade)
    k = m - 1
    n = len(data)
    margin = torch.zeros(n, k)
    entropy = torch.zeros(n, k)
    maxsoft = torch.zeros(n, k)
    correct = torch.zeros(n, m)   # correctness of ALL stages (fM needed for harm/oracle)

    for si, stage in enumerate(cascade):
        loader = DataLoader(
            ResizedDataset(data, resize_transform(stage.resolution)),
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )
        model = stage.model.to(device).eval()
        off = 0
        for x, y in loader:
            probs = torch.softmax(model(x.to(device)), dim=1)
            top2 = probs.topk(2, dim=1).values
            mp, pred = probs.max(dim=1)
            ent = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1) / math.log(probs.shape[1])
            b = y.size(0)
            sl = slice(off, off + b)
            correct[sl, si] = (pred.cpu() == y).float()
            if si < k:
                margin[sl, si] = (top2[:, 0] - top2[:, 1]).cpu()
                maxsoft[sl, si] = mp.cpu()
                entropy[sl, si] = ent.cpu()
            off += b

    fm_correct = correct[:, -1]
    harm = (1.0 - correct[:, :k]) * fm_correct.unsqueeze(1)     # [N, K] in {0,1}
    oracle = torch.full((n,), m - 1, dtype=torch.long)
    for j in range(m - 1, -1, -1):                             # cheapest correct wins
        oracle[correct[:, j].bool()] = j

    return {
        "margin": margin, "entropy": entropy, "maxsoft": maxsoft,
        "correct": correct[:, :k], "harm": harm, "oracle": oracle,
    }
