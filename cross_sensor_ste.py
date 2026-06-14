"""Cross-sensor de-risk for wavelength-conditioned STE (EuroSAT-MS).

Simulates a sensor change WITHOUT a second dataset: train on band-set A (3 bands
held out, never seen); at test, feed a "new sensor" = RGB + the HELD-OUT bands.
  - wavelength STE: embeds never-trained bands by physical wavelength (interpolated
    from trained ones) -> can exploit them.
  - index STE: untrained channel-index embeddings for those bands -> cannot.
If wavelength >> index on the new-sensor config, the conditioning is the method.

Reports same-sensor (control) and new-sensor accuracy for both. DataParallel/8-GPU.
Run:  python cross_sensor_ste.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sat_ms_headroom import ensure_unzipped, load_all

DEV = "cuda" if torch.cuda.is_available() else "cpu"
WL = torch.tensor([443., 490, 560, 665, 705, 740, 783, 842, 865, 945, 1375, 1610, 2190])
RES = torch.tensor([2, 0, 0, 0, 1, 1, 1, 0, 1, 2, 2, 1, 1])
RGB_IDX = [3, 2, 1]
HELDOUT = [4, 5, 8, 11]  # B5(705) B6(740) B8A(865) B11(1610): informative red-edge/NIR/SWIR, interpolatable
TRAIN_BANDS = [b for b in range(13) if b not in HELDOUT]      # 10 bands
NEW_SENSOR = sorted(set(RGB_IDX) | set(HELDOUT))             # RGB + held-out bands at test
IMN_M = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMN_S = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def sinusoid(vals, d):
    pos = (vals / 100.0).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
    pe = torch.zeros(len(vals), d); pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
    return pe


class STE(nn.Module):
    def __init__(self, dino, nc, d=128, use_wavelength=True):
        super().__init__()
        self.dino = dino
        for p in self.dino.parameters(): p.requires_grad = False
        self.use_wl = use_wavelength
        self.band_embed = nn.Linear(16, d)
        if use_wavelength:
            self.register_buffer("wl", sinusoid(WL, d))
            self.wl_mlp = nn.Sequential(nn.Linear(d, d), nn.ReLU(True), nn.Linear(d, d))
        else:
            self.band_idx = nn.Embedding(13, d)
        self.res_emb = nn.Embedding(3, d); self.register_buffer("res_cls", RES)
        self.q = nn.Parameter(torch.randn(d) * 0.02)
        self.proj = nn.Linear(d, 768); self.head = nn.Linear(768, nc)
        self.inj_norm = nn.LayerNorm(768)            # bound spectral magnitude
        self.gate = nn.Parameter(torch.tensor(0.1))  # gentle, learnable injection -> graceful RGB fallback

    def cond(self):
        c = self.wl_mlp(self.wl) if self.use_wl else self.band_idx.weight
        return c + self.res_emb(self.res_cls)

    def forward(self, x, rgb, mask):
        B = x.shape[0]
        p = x.reshape(B, 13, 16, 4, 16, 4).permute(0, 1, 2, 4, 3, 5).reshape(B, 13, 256, 16)
        e = self.band_embed(p) + self.cond()[None, :, None, :]
        s = (e * self.q).sum(-1).masked_fill(~mask[:, :, None].bool(), -1e9)
        w = F.softmax(s, dim=1).unsqueeze(-1)
        spec = self.gate * self.inj_norm(self.proj((w * e).sum(1)))
        tok = self.dino.prepare_tokens_with_masks(rgb)
        tok = torch.cat([tok[:, :1], tok[:, 1:] + spec], dim=1)
        for blk in self.dino.blocks: tok = blk(tok)
        return self.head(self.dino.norm(tok)[:, 0])


class DS(Dataset):
    def __init__(self, X, Y, idx, mean, std, present, train):
        self.X, self.Y, self.idx, self.mean, self.std, self.present, self.train = X, Y, idx, mean, std, present, train
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        j = self.idx[i]; x = self.X[j]
        rgb = (x[RGB_IDX] / 3000.0).clamp(0, 1)
        xn = (x - self.mean) / self.std
        mask = torch.zeros(13); mask[self.present] = 1
        if self.train:                               # random band-subset dropout (keep RGB)
            for b in self.present:
                if b not in RGB_IDX and torch.rand(1).item() < 0.4:
                    mask[b] = 0
        xn = xn * mask[:, None, None]
        if self.train:
            if torch.rand(1).item() < 0.5: xn = xn.flip(-1); rgb = rgb.flip(-1)
            if torch.rand(1).item() < 0.5: xn = xn.flip(-2); rgb = rgb.flip(-2)
        return xn, rgb, mask, self.Y[j]


def rgb224(rgb):
    rgb = F.interpolate(rgb, size=224, mode="bilinear", align_corners=False)
    return (rgb - IMN_M.to(rgb.device)) / IMN_S.to(rgb.device)


def train_eval(X, Y, tr, te, mean, std, nc, use_wl, epochs=30):
    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', verbose=False).eval()
    net = STE(dino, nc, use_wavelength=use_wl).to(DEV)
    ng = torch.cuda.device_count(); dp = nn.DataParallel(net) if ng > 1 else net
    bs = 96 * max(ng, 1)
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    trl = DataLoader(DS(X, Y, tr, mean, std, TRAIN_BANDS, True), batch_size=bs, shuffle=True, num_workers=16, drop_last=True)

    def evaluate(present):
        net.eval(); cor = tot = 0
        tel = DataLoader(DS(X, Y, te, mean, std, present, False), batch_size=bs, shuffle=False, num_workers=16)
        with torch.no_grad():
            for xn, rgb, mask, y in tel:
                p = dp(xn.to(DEV), rgb224(rgb.to(DEV)), mask.to(DEV)).argmax(1).cpu()
                cor += (p == y).sum().item(); tot += y.numel()
        return cor / tot

    for ep in range(epochs):
        net.train(); net.dino.eval()
        for xn, rgb, mask, y in trl:
            opt.zero_grad()
            F.cross_entropy(dp(xn.to(DEV), rgb224(rgb.to(DEV)), mask.to(DEV)), y.to(DEV)).backward(); opt.step()
        sched.step()
    return evaluate(TRAIN_BANDS), evaluate(NEW_SENSOR)


def main():
    tifs = ensure_unzipped(); X, Y, classes = load_all(tifs); nc = len(classes); N = len(Y)
    g = torch.Generator().manual_seed(0); perm = torch.randperm(N, generator=g)
    tr, te = perm[:int(.8 * N)], perm[int(.8 * N):]
    mean = X[tr].mean(dim=(0, 2, 3), keepdim=True)[0]; std = X[tr].std(dim=(0, 2, 3), keepdim=True)[0] + 1e-6
    print(f"cross-sensor STE: train bands={TRAIN_BANDS} (held out {HELDOUT}); new-sensor test bands={NEW_SENSOR}")

    sw, nw = train_eval(X, Y, tr, te, mean, std, nc, use_wl=True)
    print(f"  wavelength : same-sensor={sw:.4f}  NEW-sensor={nw:.4f}")
    si, ni = train_eval(X, Y, tr, te, mean, std, nc, use_wl=False)
    print(f"  index(abl) : same-sensor={si:.4f}  NEW-sensor={ni:.4f}")
    print(f"\n  same-sensor (control) wavelength vs index = {(sw-si)*100:+.2f} pts (expect ~tie)")
    print(f"  NEW-sensor  wavelength vs index           = {(nw-ni)*100:+.2f} pts   <-- the novelty test")

if __name__ == "__main__":
    main()
