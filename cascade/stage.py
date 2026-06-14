from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

from utils import ResizedDataset, resize_transform


@dataclass
class CascadeStage:
    """One stage of the cascade: a (trained) model run at a given input resolution."""
    name: str
    backbone: str
    resolution: int
    model: nn.Module


def _unwrap(model):
    """The underlying module, whether or not it is DataParallel-wrapped."""
    return model.module if isinstance(model, nn.DataParallel) else model


def _wrap(model, device, data_parallel):
    """Move to device and, if requested and >1 GPU is visible, wrap in
    DataParallel so a batch is split across GPUs."""
    model = model.to(device)
    if data_parallel and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    return model


def _replace_head(model, num_classes):
    """Resize a torchvision model's classification head to num_classes,
    keeping the (pretrained) feature extractor. Handles fc (resnet) and
    classifier (vgg/mobilenet/densenet/...) heads."""
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if hasattr(model, "heads"):  # vision transformer
        h = model.heads
        if isinstance(h, nn.Linear):
            model.heads = nn.Linear(h.in_features, num_classes)
        else:  # Sequential ending in a Linear
            idx = max(i for i, m in enumerate(h) if isinstance(m, nn.Linear))
            h[idx] = nn.Linear(h[idx].in_features, num_classes)
        return model
    if hasattr(model, "classifier"):
        clf = model.classifier
        if isinstance(clf, nn.Linear):
            model.classifier = nn.Linear(clf.in_features, num_classes)
        else:  # Sequential ending in a Linear
            idx = max(i for i, m in enumerate(clf) if isinstance(m, nn.Linear))
            clf[idx] = nn.Linear(clf[idx].in_features, num_classes)
        return model
    raise ValueError(f"Don't know how to resize the head of {type(model).__name__}.")


def build_backbone(backbone, num_classes, pretrained=False):
    """Instantiate a torchvision classification model by name, with its head
    sized to num_classes. Any model in torchvision.models works. With
    pretrained=True, load ImageNet weights and replace only the head."""
    try:
        ctor = getattr(torchvision.models, backbone)
    except AttributeError as e:
        raise ValueError(
            f"Unknown backbone '{backbone}': not found in torchvision.models."
        ) from e
    if pretrained:
        model = ctor(weights="DEFAULT")
        return _replace_head(model, num_classes)
    return ctor(num_classes=num_classes)


def _stage_loader(data, resolution, batch_size, shuffle, num_workers=0):
    return DataLoader(
        ResizedDataset(data, resize_transform(resolution)),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


@torch.no_grad()
def _evaluate(model, loader, criterion, device):
    """Return (mean loss, top-1 accuracy) over the loader."""
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        seen += x.size(0)
    return total_loss / max(seen, 1), correct / max(seen, 1)


def train_model(model, train_data, val_data, resolution, device,
                *, epochs, lr, batch_size, patience, num_workers=0):
    """Train one stage at its input resolution, with early stopping on the
    validation top-1 accuracy.

    `epochs` is the max number of epochs; training stops early if the val
    accuracy does not improve for `patience` consecutive epochs. The best
    (highest val-accuracy) weights are restored before returning. Returns the
    model in eval mode, ready for inference in the cascade.
    """
    train_loader = _stage_loader(train_data, resolution, batch_size, True, num_workers)
    val_loader = _stage_loader(val_data, resolution, batch_size, False, num_workers)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = -1.0
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        running, seen = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
            seen += x.size(0)

        val_loss, val_acc = _evaluate(model, val_loader, criterion, device)
        print(f"    epoch {epoch + 1}/{epochs}  "
              f"train_loss={running / max(seen, 1):.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"    early stopping at epoch {epoch + 1} "
                      f"(best val_acc={best_acc:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model.to(device).eval()


def run_dir(ckpt_dir, cascade_id):
    """Per-run checkpoint folder. Bumping cascade_id starts a fresh run;
    reusing it loads that run's saved weights."""
    return Path(ckpt_dir) / f"{cascade_id}_chkpts"


def _checkpoint_path(run_dir, name, backbone, resolution):
    return run_dir / f"{name}_{backbone}_{resolution}px.pt"


def build_cascade(cfg, num_classes, train_data, val_data, device=None):
    """Build and train the ordered list of cascade stages from the config.

    Iterates whatever stages the config defines (f1, f2, f3, ...) in order;
    the number of stages, the backbones, and the resolutions all come from
    the config, nothing is hardcoded. Each stage is trained on `train_data`
    at that stage's resolution with early stopping on `val_data`.

    Checkpoints are namespaced by cfg["cascade_id"]: each run's weights live in
    checkpoint_dir/<cascade_id>_chkpts/. If a stage's checkpoint already exists
    there it is loaded and training is skipped; otherwise the trained weights
    are saved. To try a different dataset/lr/epochs, bump cascade_id to start a
    fresh run; reuse the same cascade_id to load previously saved weights.

    Training hyperparameters (epochs, lr, batch_size, patience, pretrained,
    num_workers) come from cfg["train"] and the checkpoint root from
    cfg["checkpoint_dir"], all with defaults.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    train_cfg = cfg.get("train", {})
    epochs = train_cfg.get("epochs", 10)
    lr = train_cfg.get("lr", 1e-3)
    batch_size = train_cfg.get("batch_size", 128)
    patience = train_cfg.get("patience", 3)
    pretrained = train_cfg.get("pretrained", False)
    num_workers = cfg.get("num_workers", 0)
    data_parallel = cfg.get("data_parallel", False)

    ckpt_dir = cfg.get("checkpoint_dir", "checkpoints")
    cascade_id = cfg.get("cascade_id", "default")
    rdir = run_dir(ckpt_dir, cascade_id)

    stages = []
    for name, spec in cfg["cascade"].items():
        backbone, resolution = spec["backbone"], spec["resolution"]
        # Per-stage lr/epochs overrides (transformers need a much lower lr than
        # ResNets to fine-tune; one global lr breaks heterogeneous cascades).
        stage_lr = spec.get("lr", lr)
        stage_epochs = spec.get("epochs", epochs)
        base = build_backbone(backbone, num_classes, pretrained=pretrained)
        ckpt = _checkpoint_path(rdir, name, backbone, resolution)

        if ckpt.exists():
            base.load_state_dict(torch.load(ckpt, map_location=device))
            model = _wrap(base, device, data_parallel).eval()
            print(f"loaded {name}: {backbone} @ {resolution}px from {ckpt}")
        else:
            print(f"training {name}: {backbone} @ {resolution}px (lr={stage_lr})")
            model = train_model(
                _wrap(base, device, data_parallel), train_data, val_data, resolution, device,
                epochs=stage_epochs, lr=stage_lr, batch_size=batch_size, patience=patience,
                num_workers=num_workers,
            )
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            # save the unwrapped weights so checkpoints load into a plain model
            torch.save(_unwrap(model).state_dict(), ckpt)
            print(f"saved {name} to {ckpt}")

        stages.append(
            CascadeStage(name=name, backbone=backbone, resolution=resolution, model=model)
        )
    return stages
