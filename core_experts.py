"""CoRE decisive experiment: where does fine-grained confusion live?

For the top-3 confusion pairs of a resnet50@224 base on Imagewoof, train four
residual-repair experts per pair and compare:

  V1 frozen pooled-feature MLP      (baseline; weak)
  V2 frozen feature-map + conv      (is spatial info present but pooled away?)
  V3 layer4 adapter (fine-tune L4)  (can late features learn the distinction?)
  V4 full pair specialist           (upper bound: is the pair learnable at all?)

Expert = residual on the pair's two logits:  z_final[a]=z_base[a]+dz_a, [b]+dz_b
(trained with base logits detached). Evaluated on three sets:
  (1) pair-val   : forced a-vs-b accuracy on label in {a,b}
  (2) triggered  : base top-2 == {a,b}; correction/damage/net rates
  (3) full-val   : apply on triggered; global top-1 repairs/new-errors/net

Held-out official val, split 50/50 (fit / eval). GO needs the feature-learning
variants to clear: pair +5-10pts, triggered net positive, full-val >= +0.8-1.0pt,
low damage. Run:  python core_experts.py
"""

import argparse
import copy
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import yaml
import torchvision
from torch.utils.data import DataLoader, Subset

from cascade.stage import build_cascade
from utils import ResizedDataset, get_dataset, resize_transform, _RGBView

DEV = "cuda" if torch.cuda.is_available() else "cpu"


class Backbone(nn.Module):
    """resnet50 split so we can tap layer3 / layer4 / pooled / logits."""
    def __init__(self, rn):
        super().__init__()
        self.pre = nn.Sequential(rn.conv1, rn.bn1, rn.relu, rn.maxpool, rn.layer1, rn.layer2, rn.layer3)
        self.layer4 = rn.layer4
        self.avgpool = rn.avgpool
        self.fc = rn.fc

    @torch.no_grad()
    def forward(self, x):
        l3 = self.pre(x)
        l4 = self.layer4(l3)
        pooled = torch.flatten(self.avgpool(l4), 1)
        return l3, l4, pooled, self.fc(pooled)


@torch.no_grad()
def run_indices(bb, val, res, idx, bs=128, nw=8, want=("pooled", "logits")):
    out = {k: [] for k in want}
    loader = DataLoader(ResizedDataset(Subset(val, idx), resize_transform(res)),
                        batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    for x, _ in loader:
        l3, l4, pooled, logits = bb(x.to(DEV))
        for k, t in [("l3", l3), ("l4", l4), ("pooled", pooled), ("logits", logits)]:
            if k in want:
                out[k].append(t.cpu())
    return {k: torch.cat(v) for k, v in out.items()}


def train_residual(expert, fit_feat, zbase_ab, target, steps=300, lr=1e-3, wd=1e-2):
    """expert(feat)->2 residual logits; minimize CE on (zbase_ab + residual)."""
    expert = expert.to(DEV).train()
    opt = torch.optim.AdamW(expert.parameters(), lr=lr, weight_decay=wd)
    f = fit_feat.to(DEV); zb = zbase_ab.to(DEV); t = target.to(DEV)
    for _ in range(steps):
        opt.zero_grad()
        dz = expert(f)
        nn.functional.cross_entropy(zb + dz, t).backward()
        opt.step()
    return expert.eval()


def choice_from_residual(expert, feat, zbase_ab):
    with torch.no_grad():
        dz = expert(feat.to(DEV)).cpu()
    return (zbase_ab + dz).argmax(1)            # 0 -> class a, 1 -> class b


def main(cfg_path, valdir):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    nw = cfg.get("num_workers", 8)
    base_ds = get_dataset(cfg); nc = len(base_ds.classes); cls = base_ds.classes
    casc = build_cascade(cfg, nc, base_ds, base_ds)
    rn = casc[1].model
    rn = (rn.module if isinstance(rn, nn.DataParallel) else rn).to(DEV).eval()
    res = casc[1].resolution
    bb = Backbone(rn).to(DEV).eval()

    val = _RGBView(torchvision.datasets.ImageFolder(valdir))
    N = len(val)
    allidx = list(range(N))
    full = run_indices(bb, val, res, allidx, nw=nw, want=("pooled", "logits"))
    Z = full["logits"]; pooled = full["pooled"]
    Y = torch.tensor([val.base.samples[i][1] for i in allidx])
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(N, generator=g)
    calib, eval = perm[:N // 2].tolist(), perm[N // 2:].tolist()

    # confusion pairs from calib errors
    pa = Z.argmax(1)
    pairs = Counter()
    for i in calib:
        if pa[i] != Y[i]:
            pairs[frozenset((int(Y[i]), int(pa[i])))] += 1
    top_pairs = [tuple(sorted(s)) for s, _ in pairs.most_common(3)]
    print(f"\n#### CoRE experts: base resnet50@{res}, eval n={len(eval)} ####")
    print("top-3 confusion pairs:", [f"{cls[a]}|{cls[b]}" for a, b in top_pairs])

    top2_all = Z.topk(2, 1).indices
    base_top1 = (Z[eval].argmax(1) == Y[eval]).float().mean().item()

    # global predictions per variant (start from base, override triggered)
    variants = ["V1_pooled", "V2_featmap", "V3_adapter", "V4_specialist"]
    gpred = {v: Z[eval].argmax(1).clone() for v in variants}
    eval_pos = {idx: k for k, idx in enumerate(eval)}     # eval index -> row

    for (a, b) in top_pairs:
        fit_idx = [i for i in calib if int(Y[i]) in (a, b)]
        epair_idx = [i for i in eval if int(Y[i]) in (a, b)]
        etrig_idx = [i for i in eval if {int(top2_all[i, 0]), int(top2_all[i, 1])} == {a, b}]
        print(f"\n== pair {cls[a]} | {cls[b]} ==  fit={len(fit_idx)} eval_pair={len(epair_idx)} triggered={len(etrig_idx)}")

        # frozen-prefix features for fit / eval-pair / eval-trig
        feats = {s: run_indices(bb, val, res, idx, nw=nw, want=("l3", "l4", "pooled", "logits"))
                 for s, idx in [("fit", fit_idx), ("pair", epair_idx), ("trig", etrig_idx)]}

        def zab(split): return feats[split]["logits"][:, [a, b]]
        def lab(idx_list): return torch.tensor([0 if int(Y[i]) == a else 1 for i in idx_list])
        tfit = lab(fit_idx)

        experts = {}
        # V1: pooled MLP
        e1 = nn.Sequential(nn.Linear(2048, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 2))
        experts["V1_pooled"] = (train_residual(e1, feats["fit"]["pooled"], zab("fit"), tfit), "pooled")
        # V2: feature-map conv
        e2 = nn.Sequential(nn.Conv2d(2048, 256, 3, padding=1), nn.ReLU(),
                           nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, 2))
        experts["V2_featmap"] = (train_residual(e2, feats["fit"]["l4"], zab("fit"), tfit), "l4")
        # V3: layer4 adapter (trainable copy of layer4 on frozen l3)
        class L4Adapter(nn.Module):
            def __init__(self):
                super().__init__()
                self.l4 = copy.deepcopy(rn.layer4)
                self.head = nn.Linear(2048, 2)
            def forward(self, l3):
                return self.head(torch.flatten(nn.functional.adaptive_avg_pool2d(self.l4(l3), 1), 1))
        experts["V3_adapter"] = (train_residual(L4Adapter(), feats["fit"]["l3"], zab("fit"), tfit, steps=200, lr=3e-4), "l3")

        # V4: full specialist (fine-tune base resnet50, 2-way, residual)
        class Specialist(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = copy.deepcopy(rn)
                self.net.fc = nn.Linear(2048, 2)
            def forward(self, x):
                return self.net(x)
        spec = Specialist().to(DEV).train()
        opt = torch.optim.AdamW(spec.parameters(), lr=1e-4, weight_decay=1e-2)
        loader = DataLoader(ResizedDataset(Subset(val, fit_idx), resize_transform(res)),
                            batch_size=64, shuffle=True, num_workers=nw, pin_memory=True)
        ymap = {a: 0, b: 1}
        for _ in range(15):
            for x, yy in loader:
                x = x.to(DEV); t = torch.tensor([ymap[int(v)] for v in yy], device=DEV)
                opt.zero_grad()
                nn.functional.cross_entropy(spec(x), t).backward(); opt.step()
        spec.eval()

        # ---- evaluate ----
        def base_pair_choice(split):
            return zab(split).argmax(1)
        bp_pair = (base_pair_choice("pair") == lab(epair_idx)).float().mean().item() if epair_idx else float("nan")
        print(f"   pair-val forced a/b acc:   base={bp_pair:.4f}", end="")
        choices_pair, choices_trig = {}, {}
        for v in ["V1_pooled", "V2_featmap", "V3_adapter"]:
            exp, key = experts[v]
            choices_pair[v] = choice_from_residual(exp, feats["pair"][key], zab("pair")) if epair_idx else torch.tensor([])
            choices_trig[v] = choice_from_residual(exp, feats["trig"][key], zab("trig")) if etrig_idx else torch.tensor([])
            acc = (choices_pair[v] == lab(epair_idx)).float().mean().item() if epair_idx else float("nan")
            print(f"  {v}={acc:.4f}", end="")
        # specialist choices (absolute)
        with torch.no_grad():
            sp_pair = spec(torch.stack([resize_transform(res)(val[i][0]) for i in epair_idx]).to(DEV)).argmax(1).cpu() if epair_idx else torch.tensor([])
            sp_trig = spec(torch.stack([resize_transform(res)(val[i][0]) for i in etrig_idx]).to(DEV)).argmax(1).cpu() if etrig_idx else torch.tensor([])
        sp_acc = (sp_pair == lab(epair_idx)).float().mean().item() if epair_idx else float("nan")
        print(f"  V4_specialist={sp_acc:.4f}")
        choices_pair["V4_specialist"] = sp_pair; choices_trig["V4_specialist"] = sp_trig

        # triggered: correction/damage/net + accumulate global overrides
        if etrig_idx:
            tl = lab(etrig_idx); base_c = (base_pair_choice("trig") == tl)
            print(f"   triggered (n={len(etrig_idx)}): base_acc={base_c.float().mean():.4f}")
            for v in variants:
                ch = choices_trig[v]
                corr = ((~base_c) & (ch == tl)).float().mean().item()
                dmg = (base_c & (ch != tl)).float().mean().item()
                print(f"      {v:<14} correction={corr:.3f} damage={dmg:.3f} net={corr-dmg:+.3f}")
                # apply to global eval preds (map local choice -> class id)
                for k, i in enumerate(etrig_idx):
                    gpred[v][eval_pos[i]] = a if int(ch[k]) == 0 else b

    print(f"\n#### FULL-VAL global top-1 (base={base_top1:.4f}) ####")
    yb = Y[eval]
    for v in variants:
        acc = (gpred[v] == yb).float().mean().item()
        base_pred = Z[eval].argmax(1)
        repairs = int(((base_pred != yb) & (gpred[v] == yb)).sum())
        newerr = int(((base_pred == yb) & (gpred[v] != yb)).sum())
        print(f"   {v:<14} top1={acc:.4f}  ({(acc-base_top1)*100:+.2f} pts)  repairs={repairs} new_errors={newerr} net={repairs-newerr:+d}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="cfgs/imagewoof.yaml")
    ap.add_argument("--valdir", default="data/imagewoof2/val")
    a = ap.parse_args()
    main(a.cfg, a.valdir)
