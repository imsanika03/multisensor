"""Quick diagnostic: DINOv2-only linear probe on EuroSAT to verify data pipeline."""
import math, torch, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sat_ms_headroom import ensure_unzipped, load_all

DEV = "cuda"
RGB_IDX = [3, 2, 1]
IMN_M = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMN_S = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class DS(Dataset):
    def __init__(self, X, Y, idx):
        self.X, self.Y, self.idx = X, Y, idx

    def __len__(self): return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        x = self.X[j].float()
        rgb = (x[RGB_IDX] / 3000.0).clamp(0, 1)
        rgb = F.interpolate(rgb[None], size=224, mode='bilinear', align_corners=False)[0]
        rgb = (rgb - IMN_M) / IMN_S
        return rgb, self.Y[j]


torch.manual_seed(0)
tifs = ensure_unzipped()
X, Y, classes = load_all(tifs)
print(f"X={tuple(X.shape)} Y={tuple(Y.shape)} classes={classes}")
print(f"X range: {X.min():.1f} - {X.max():.1f}, RGB band mean: {X[:, RGB_IDX].mean():.1f}")
print(f"Y dist: {[(Y==i).sum().item() for i in range(10)]}")

N = len(X)
perm = torch.randperm(N).tolist()
tr, te = perm[:int(N*0.8)], perm[int(N*0.8):]

dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14',
                      pretrained=True, verbose=False).to(DEV)
for p in dino.parameters(): p.requires_grad_(False)

# Extract features for train set
print("\nExtracting train features...")
trl = DataLoader(DS(X, Y, tr), batch_size=256, shuffle=False, num_workers=0)
feats, labels = [], []
dino.eval()
with torch.no_grad():
    for rgb, y in trl:
        f = dino(rgb.to(DEV))
        if isinstance(f, dict):
            f = f["x_norm_clstoken"]
        feats.append(f.cpu()); labels.append(y)
feats_tr = torch.cat(feats); labels_tr = torch.cat(labels)
print(f"Train features: {tuple(feats_tr.shape)}, mean={feats_tr.mean():.3f}, std={feats_tr.std():.3f}")

# Linear probe
head = torch.nn.Linear(768, 10).to(DEV)
opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
for ep in range(50):
    idx = torch.randperm(len(feats_tr))
    for i in range(0, len(feats_tr), 256):
        b = idx[i:i+256]
        f = feats_tr[b].to(DEV); y = labels_tr[b].to(DEV)
        loss = F.cross_entropy(head(f), y)
        opt.zero_grad(); loss.backward(); opt.step()
    if (ep+1) % 10 == 0:
        print(f"  epoch {ep+1}/50 loss={loss.item():.3f}")

# Eval
print("\nExtracting test features...")
tel = DataLoader(DS(X, Y, te), batch_size=256, shuffle=False, num_workers=0)
feats_te, labels_te = [], []
with torch.no_grad():
    for rgb, y in tel:
        f = dino(rgb.to(DEV))
        if isinstance(f, dict):
            f = f["x_norm_clstoken"]
        feats_te.append(f.cpu()); labels_te.append(y)
feats_te = torch.cat(feats_te); labels_te = torch.cat(labels_te)

head.eval()
with torch.no_grad():
    logits = head(feats_te.to(DEV)).cpu()
acc = (logits.argmax(1) == labels_te).float().mean().item()
print(f"\nDINOv2 LP accuracy: {acc*100:.2f}%")
print("(expected ~96% for EuroSAT)")
