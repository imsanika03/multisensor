import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from utils import resize_transform


class DepthwiseSeparableConv(nn.Module):

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size=3, stride=stride, padding=1,
            groups=in_ch, bias=False,
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class Probe(nn.Module):
    """Multi-task one-shot router g_phi(x_low).

    A shared low-res embedding feeds several heads (K = num_routes-1 cheap routes,
    M = num_routes stages):

        margin   [K]  predicted top1-top2 softmax margin of f_k   (in [0,1] via sigmoid)
        entropy  [K]  predicted normalized prediction entropy of f_k
        maxsoft  [K]  predicted max-softmax of f_k
        correct  [K]  predicted P(f_k correct)                    (logit)
        harm     [K]  predicted lost-correction risk of f_k       (logit; routing signal)
        oracle   [M]  predicted cheapest sufficient route          (logits over stages)

    Routing uses safety_k = sigmoid(-harm_k) (P(not harmful)). The other heads are
    auxiliary supervision that shapes the embedding so the harm head ranks well.
    """

    def __init__(self, num_classes, num_routes, input_dim, z_dim):
        super().__init__()
        k = num_routes - 1   # cheap routes

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.backbone = nn.Sequential(
            DepthwiseSeparableConv(32, 64, stride=2),
            DepthwiseSeparableConv(64, 128, stride=2),
            DepthwiseSeparableConv(128, 256, stride=2),
            DepthwiseSeparableConv(256, 256, stride=1),
        )
        self.z_embedding_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, z_dim),
        )

        # Confidence-distillation heads (regress f_k's margin / entropy / maxsoftmax).
        self.margin_head = nn.Linear(z_dim, k)
        self.entropy_head = nn.Linear(z_dim, k)
        self.maxsoft_head = nn.Linear(z_dim, k)
        # Decision heads.
        self.correct_head = nn.Linear(z_dim, k)        # q_k: P(f_k correct)
        self.harm_head = nn.Linear(z_dim, k)           # r_k: lost-correction risk (routing signal)
        self.oracle_head = nn.Linear(z_dim, num_routes)  # rho: cheapest sufficient route

    def forward(self, x):
        z = self.z_embedding_head(self.backbone(self.stem(x)))
        return {
            "margin": self.margin_head(z),
            "entropy": self.entropy_head(z),
            "maxsoft": self.maxsoft_head(z),
            "correct": self.correct_head(z),
            "harm": self.harm_head(z),
            "oracle": self.oracle_head(z),
            "z": z,
        }


class _RouterDataset(Dataset):
    """Pairs each calibration image with its precomputed router targets.

    `targets` is a dict of row-aligned tensors (see compute_router_targets).
    Yields (image@probe-resolution, {target_name: row}).
    """

    def __init__(self, base, targets, transform):
        self.base = base
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img, _ = self.base[i]
        return self.transform(img), {k: v[i] for k, v in self.targets.items()}


def _route_auc(safety, harm):
    """Mean per-route AUC of safety scores vs the safe label (1 - harm).

    Threshold-free: measures how well the router *ranks* safe routings above
    harmful ones, which is exactly what calibration needs. Rank-based (average
    ties); 0.5 for a route with only one class present.
    """
    aucs = []
    for k in range(safety.shape[1]):
        s = safety[:, k]
        pos = (harm[:, k] == 0)                  # safe == positive class
        n_pos = int(pos.sum()); n_neg = int((~pos).sum())
        if n_pos == 0 or n_neg == 0:
            aucs.append(0.5); continue
        order = s.argsort()
        ranks = torch.empty_like(s)
        ranks[order] = torch.arange(1, len(s) + 1, dtype=s.dtype)
        auc = (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        aucs.append(float(auc))
    return sum(aucs) / len(aucs)


def _ranking_loss(harm_logit, harm_target):
    """Pairwise logistic ranking: push safe examples' safety above harmful ones.

    Safety score = -harm_logit (higher == safer). For each route, every
    (safe, harmful) pair contributes softplus(-(s_safe - s_harm)), so the loss is
    minimized when safe routings rank strictly above harmful ones (a soft AUC).
    """
    s = -harm_logit
    total, n = 0.0, 0
    for k in range(s.shape[1]):
        sk, tk = s[:, k], harm_target[:, k]
        safe, bad = sk[tk == 0], sk[tk == 1]
        if safe.numel() == 0 or bad.numel() == 0:
            continue
        diff = safe.unsqueeze(1) - bad.unsqueeze(0)     # [n_safe, n_bad]
        total = total + F.softplus(-diff).mean()
        n += 1
    if n == 0:
        return harm_logit.sum() * 0.0
    return total / n


_DEFAULT_LAMBDAS = {
    "distill": 1.0,          # weight on L_conf_distill (margin/entropy/maxsoftmax)
    "correctness": 1.0,      # lambda1
    "lost_correction": 1.0,  # lambda2
    "oracle_route": 1.0,     # lambda3
    "ranking": 1.0,          # lambda4
}


def train_probe(probe, calib_data, targets, resolution, device,
                *, epochs, lr, batch_size, weight_decay=0.0, lambdas=None,
                val_frac=0.2, num_workers=0):
    """Train the multi-task one-shot router.

        L = L_conf_distill
          + lambda1 * L_correctness        (BCE: f_k correct?)
          + lambda2 * L_lost_correction    (BCE: f_k wrong & f3 right? -- routing head)
          + lambda3 * L_oracle_route       (CE: cheapest correct stage)
          + lambda4 * L_ranking            (safe routings ranked above harmful)

    L_conf_distill regresses the margin / entropy / max-softmax heads onto f_k's
    actual values. The lost-correction BCE is class-balanced (harm is rare). A
    val_frac slice is held out and the epoch with the best validation route-AUC
    (of the safety score sigmoid(-harm)) is kept. device=None -> CUDA if available.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    lam = {**_DEFAULT_LAMBDAS, **(lambdas or {})}

    # Held-out split for epoch selection (ranking quality, not train loss).
    n = len(calib_data)
    n_val = int(val_frac * n)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))  # fixed split
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    full = _RouterDataset(calib_data, targets, resize_transform(resolution))
    train_loader = DataLoader(
        Subset(full, train_idx.tolist()),
        batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        Subset(full, val_idx.tolist()),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    probe = probe.to(device).train()
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)

    # Class-balance the lost-correction BCE (harm is the rare class).
    harm_counts = targets["harm"][train_idx].float().sum(dim=0)     # [K]
    safe_counts = len(train_idx) - harm_counts
    pos_weight = (harm_counts / safe_counts.clamp(min=1.0)).to(device)

    distill = nn.SmoothL1Loss()
    correct_crit = nn.BCEWithLogitsLoss()
    harm_crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    oracle_crit = nn.CrossEntropyLoss()

    best_auc, best_state = -1.0, None
    for epoch in range(epochs):
        probe.train()
        running, seen = 0.0, 0
        for x, t in train_loader:
            x = x.to(device)
            t = {k: v.to(device) for k, v in t.items()}
            optimizer.zero_grad()
            h = probe(x)

            l_distill = (distill(torch.sigmoid(h["margin"]), t["margin"])
                         + distill(torch.sigmoid(h["entropy"]), t["entropy"])
                         + distill(torch.sigmoid(h["maxsoft"]), t["maxsoft"]))
            l_correct = correct_crit(h["correct"], t["correct"])
            l_harm = harm_crit(h["harm"], t["harm"])
            l_oracle = oracle_crit(h["oracle"], t["oracle"])
            l_rank = _ranking_loss(h["harm"], t["harm"])

            loss = (lam["distill"] * l_distill
                    + lam["correctness"] * l_correct
                    + lam["lost_correction"] * l_harm
                    + lam["oracle_route"] * l_oracle
                    + lam["ranking"] * l_rank)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
            seen += x.size(0)

        # Validation route-AUC of the routing safety score; keep best epoch.
        probe.eval()
        s_val, h_val = [], []
        with torch.no_grad():
            for x, t in val_loader:
                h = probe(x.to(device))
                s_val.append(torch.sigmoid(-h["harm"]).cpu())   # safety = P(not harmful)
                h_val.append(t["harm"])
        auc = _route_auc(torch.cat(s_val), torch.cat(h_val))
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()}
        print(f"    probe epoch {epoch + 1}/{epochs}  loss={running / max(seen, 1):.4f}  "
              f"val_auc={auc:.4f}  best={best_auc:.4f}")

    if best_state is not None:
        probe.load_state_dict(best_state)
    print(f"    selected router epoch with val_auc={best_auc:.4f}")
    return probe.eval()
