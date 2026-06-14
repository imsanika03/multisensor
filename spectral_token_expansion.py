"""Spectral Token Expansion (STE) de-risk on EuroSAT-MS.

Mechanism: 13 Sentinel-2 bands -> per-band 4x4 patch embeddings -> + physical
WAVELENGTH embedding (sinusoidal) + RESOLUTION embedding (10/20/60m) -> attention
pool over bands per patch position -> project -> ADD to frozen DINOv2's patch
tokens -> frozen blocks -> head. Only the adapter + head train.

De-risks the DISTINCTIVE claims (not just "MS helps"):
  1. STE >= late-fusion (0.986) / MS-CNN (0.982) / DINOv2-RGB (0.957), tiny adapter.
  2. trainable-parameter budget (vs frozen 86M DINOv2).
  3. graceful missing-band degradation, and WAVELENGTH conditioning beats a learned
     channel-INDEX embedding when bands are dropped (the cross-sensor payoff).

Run:  python spectral_token_expansion.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sat_ms_headroom import ensure_unzipped, load_all

DEV = "cuda" if torch.cuda.is_available() else "cpu"
WL = torch.tensor([443., 490, 560, 665, 705, 740, 783, 842, 865, 945, 1375, 1610, 2190])  # nm
RES = torch.tensor([2, 0, 0, 0, 1, 1, 1, 0, 1, 2, 2, 1, 1])   # 0=10m 1=20m 2=60m
RGB_IDX = [3, 2, 1]
IMN_M = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMN_S = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def sinusoid(vals, d):
    pos = (vals / 100.0).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
    pe = torch.zeros(len(vals), d)
    pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
    return pe


class STE(nn.Module):
    def __init__(self, dino, nc, d=128, use_wavelength=True):
        super().__init__()
        self.dino = dino
        for p in self.dino.parameters():
            p.requires_grad = False
        self.use_wl = use_wavelength
        self.band_embed = nn.Linear(16, d)
        if use_wavelength:
            self.register_buffer("wl", sinusoid(WL, d))
            self.wl_mlp = nn.Sequential(nn.Linear(d, d), nn.ReLU(True), nn.Linear(d, d))
        else:
            self.band_idx = nn.Embedding(13, d)
        self.res_emb = nn.Embedding(3, d)
        self.register_buffer("res_cls", RES)
        self.q = nn.Parameter(torch.randn(d) * 0.02)
        self.proj = nn.Linear(d, 768)
        self.head = nn.Linear(768, nc)

    def band_cond(self):
        c = self.wl_mlp(self.wl) if self.use_wl else self.band_idx.weight   # [13,d]
        return c + self.res_emb(self.res_cls)

    def forward(self, x, rgb, mask):                        # x[B,13,64,64] mask[B,13]
        B = x.shape[0]
        p = x.reshape(B, 13, 16, 4, 16, 4).permute(0, 1, 2, 4, 3, 5).reshape(B, 13, 256, 16)
        e = self.band_embed(p) + self.band_cond()[None, :, None, :]          # [B,13,256,d]
        scores = (e * self.q).sum(-1)                                        # [B,13,256]
        scores = scores.masked_fill(~mask[:, :, None].bool(), -1e9)
        w = F.softmax(scores, dim=1).unsqueeze(-1)
        spec = self.proj((w * e).sum(1))                                     # [B,256,768]
        tok = self.dino.prepare_tokens_with_masks(rgb)                       # [B,257,768]
        tok = torch.cat([tok[:, :1], tok[:, 1:] + spec], dim=1)
        for blk in self.dino.blocks:
            tok = blk(tok)
        return self.head(self.dino.norm(tok)[:, 0])


class DS(Dataset):
    def __init__(self, X, Y, idx, mean, std, train, drop_p=0.3):
        self.X, self.Y, self.idx, self.mean, self.std, self.train, self.dp = X, Y, idx, mean, std, train, drop_p
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        j = self.idx[i]; x = self.X[j]
        rgb = (x[RGB_IDX] / 3000.0).clamp(0, 1)
        xn = (x - self.mean) / self.std
        mask = torch.ones(13)
        if self.train and torch.rand(1).item() < 0.8:       # random band dropout (keep RGB)
            drop = (torch.rand(13) < self.dp)
            for r in RGB_IDX: drop[r] = False
            mask[drop] = 0
            xn = xn * mask[:, None, None]
        if self.train:
            if torch.rand(1).item() < 0.5: xn = xn.flip(-1); rgb = rgb.flip(-1)
            if torch.rand(1).item() < 0.5: xn = xn.flip(-2); rgb = rgb.flip(-2)
        return xn, rgb, mask, self.Y[j]


def rgb224(rgb):
    rgb = F.interpolate(rgb, size=224, mode="bilinear", align_corners=False)
    return (rgb - IMN_M.to(rgb.device)) / IMN_S.to(rgb.device)


def train_eval(X, Y, tr, te, mean, std, nc, use_wl, epochs=20):
    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', verbose=False).eval()
    net = STE(dino, nc, use_wavelength=use_wl).to(DEV)
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    ngpu = torch.cuda.device_count()
    dp = nn.DataParallel(net) if ngpu > 1 else net
    bs = 96 * max(ngpu, 1)                                   # ~96/GPU across all GPUs
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    trl = DataLoader(DS(X, Y, tr, mean, std, True), batch_size=bs, shuffle=True, num_workers=16, drop_last=True)

    def evaluate(drop_bands=None):
        net.eval(); cor = tot = 0
        tel = DataLoader(DS(X, Y, te, mean, std, False), batch_size=bs, shuffle=False, num_workers=16)
        with torch.no_grad():
            for xn, rgb, mask, y in tel:
                if drop_bands is not None:
                    mask = mask.clone(); mask[:, drop_bands] = 0
                    xn = xn * mask[:, :, None, None]
                p = dp(xn.to(DEV), rgb224(rgb.to(DEV)), mask.to(DEV)).argmax(1).cpu()
                cor += (p == y).sum().item(); tot += y.numel()
        return cor / tot

    for ep in range(epochs):
        net.train(); net.dino.eval()
        for xn, rgb, mask, y in trl:
            opt.zero_grad()
            out = dp(xn.to(DEV), rgb224(rgb.to(DEV)), mask.to(DEV))
            F.cross_entropy(out, y.to(DEV)).backward(); opt.step()
        sched.step()
    full = evaluate()
    drop = [0, 9, 10, 12]                                   # drop B1,B9,B10,B12 (60m + SWIR edge)
    miss = evaluate(drop_bands=drop)
    return full, miss, trainable


def main():
    tifs = ensure_unzipped(); X, Y, classes = load_all(tifs); nc = len(classes); N = len(Y)
    g = torch.Generator().manual_seed(0); perm = torch.randperm(N, generator=g)
    tr, te = perm[:int(.8 * N)], perm[int(.8 * N):]
    mean = X[tr].mean(dim=(0, 2, 3), keepdim=True)[0]; std = X[tr].std(dim=(0, 2, 3), keepdim=True)[0] + 1e-6
    print(f"STE de-risk on EuroSAT-MS: N={N} train={len(tr)} test={len(te)}")
    print("  baselines: DINOv2-RGB=0.957  MS-CNN=0.982  late-fusion=0.986")

    fw, mw, tp = train_eval(X, Y, tr, te, mean, std, nc, use_wl=True)
    print(f"  STE-wavelength : full={fw:.4f}  missing-4band={mw:.4f}  trainable={tp/1e6:.3f}M")
    fi, mi, _ = train_eval(X, Y, tr, te, mean, std, nc, use_wl=False)
    print(f"  STE-index(abl) : full={fi:.4f}  missing-4band={mi:.4f}")
    print(f"\n  STE vs late-fusion = {(fw-0.986)*100:+.2f} pts;  trainable {tp/1e6:.3f}M vs 86M frozen DINOv2")
    print(f"  wavelength vs index, FULL    = {(fw-fi)*100:+.2f} pts")
    print(f"  wavelength vs index, MISSING = {(mw-mi)*100:+.2f} pts   <-- cross-sensor payoff")


if __name__ == "__main__":
    main()
