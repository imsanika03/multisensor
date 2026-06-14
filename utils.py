import io
import json
import os
import shutil
import urllib.request

import pyarrow.parquet as pq
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# torchvision backbones are ImageNet-style; use the standard normalization.
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

# torchvision.datasets.CIFAR100 downloads from cs.toronto.edu, which 403s from
# this environment. Read the uoft-cs/cifar100 parquet mirror instead; it carries
# the same fine/coarse labels.
_CIFAR100_HF_BASE = "https://huggingface.co/datasets/uoft-cs/cifar100/resolve/main/cifar100"
_CIFAR100_FILES = {
    "train": "train-00000-of-00001.parquet",
    "test": "test-00000-of-00001.parquet",
}
_DATA_DIR = os.path.join("data", "cifar100_hf")


def get_dataset(cfg):
    """Dispatch to a dataset loader. `cfg` may be the full config dict or just
    the dataset-name string (back-compat). All loaders yield (PIL RGB, int
    label) and expose `.classes`, matching the cascade's contract.

    ImageNet-family datasets require a manual download; point `data_root` at a
    directory laid out as ImageFolder (root/train/<wnid>/*.JPEG).
    """
    if isinstance(cfg, str):
        cfg = {"dataset": cfg}
    name = cfg["dataset"]

    match name:
        case "cifar100":
            return get_cifar100()
        case "imagenette":
            return get_imagenette()
        case "imagewoof":
            return _fastai_imageclas("imagewoof2", "imagewoof2.tgz")
        case "cub":
            return _fastai_imageclas("CUB_200_2011", "CUB_200_2011.tgz", split="images")
        case "imagenette_c":
            return _CorruptedView(get_imagenette(), cfg.get("corruption", "gaussian_noise"),
                                  int(cfg.get("severity", 3)))
        case "imagenet100":
            return _imagefolder(cfg["data_root"], num_classes=100)
        case "imagenet" | "imagenet1k":
            return _imagefolder(cfg["data_root"], num_classes=None)
        case _:
            raise ValueError(f"unknown dataset '{name}'")


_FASTAI_BASE = "https://s3.amazonaws.com/fast-ai-imageclas"


def get_imagenette():
    """Imagenette (full-size, 10 ImageNet classes) train split, via torchvision's
    built-in loader (matches the validated cached checkpoints)."""
    ds = torchvision.datasets.Imagenette(
        root="./data", split="train", size="full",
        download=not os.path.isdir(os.path.join("data", "imagenette2")),
    )
    return _RGBView(ds)


def _fastai_imageclas(folder, archive, split="train"):
    """Download+extract a fast.ai imageclas tgz (e.g. Imagewoof) and return its
    train split as an RGB ImageFolder. Native high-res photos, so the 112/224/448
    stages are genuine downsampled views."""
    from torchvision.datasets.utils import download_and_extract_archive
    root = os.path.join("data", folder)
    if not os.path.isdir(root):
        download_and_extract_archive(f"{_FASTAI_BASE}/{archive}", download_root="data")
    return _RGBView(torchvision.datasets.ImageFolder(os.path.join(root, split)))


def _imagefolder(data_root, num_classes=None, split="train"):
    """ImageNet-style ImageFolder at `data_root/<split>`. With `num_classes` set,
    keep only the first N classes (sorted wnid order) -> a deterministic
    ImageNet-100 / subset. Labels stay 0..N-1 (ImageFolder sorts by class)."""
    base = torchvision.datasets.ImageFolder(os.path.join(data_root, split))
    if num_classes is None:
        return _RGBView(base)
    keep = [i for i, (_, c) in enumerate(base.samples) if c < num_classes]
    return _RGBSubset(base, keep, base.classes[:num_classes])


class _RGBView(Dataset):
    """Force RGB (some photos are grayscale/CMYK); expose `.classes`."""

    def __init__(self, base):
        self.base = base
        self.classes = base.classes

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img, label = self.base[i]
        return img.convert("RGB"), label


class _RGBSubset(Dataset):
    """RGB subset of an ImageFolder restricted to a set of sample indices."""

    def __init__(self, base, indices, classes):
        self.base = base
        self.indices = indices
        self.classes = classes

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, label = self.base[self.indices[i]]
        return img.convert("RGB"), label


class _CorruptedView(Dataset):
    """Apply an ImageNet-C-style corruption to each image (at native resolution,
    before the per-stage resize). Used to stress-test routing under distribution
    shift."""

    def __init__(self, base, corruption, severity):
        from corruptions import corrupt
        self.base = base
        self.classes = base.classes
        self._corrupt = corrupt
        self._name = corruption
        self._severity = severity

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img, label = self.base[i]
        return self._corrupt(img, self._name, self._severity), label


def _ensure_parquet(split):
    """Return the local path to the split's parquet, downloading it from the
    Hugging Face mirror on first use."""
    fname = _CIFAR100_FILES[split]
    path = os.path.join(_DATA_DIR, fname)
    if not os.path.exists(path):
        os.makedirs(_DATA_DIR, exist_ok=True)
        url = f"{_CIFAR100_HF_BASE}/{fname}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        tmp = path + ".part"
        with urllib.request.urlopen(req) as resp, open(tmp, "wb") as f:
            shutil.copyfileobj(resp, f)
        os.replace(tmp, path)
    return path


class CIFAR100Parquet(Dataset):
    """CIFAR-100 read from the Hugging Face parquet mirror.

    Yields (PIL RGB image, fine label), matching the behaviour of
    torchvision.datasets.CIFAR100 (which defaults to the 100-class fine labels)
    so the rest of the pipeline is unchanged. Also exposes `.classes` (the
    100 fine-label names) like the torchvision dataset.
    """

    def __init__(self, split="train"):
        table = pq.read_table(_ensure_parquet(split), columns=["img", "fine_label"])
        cols = table.to_pydict()
        self._imgs = cols["img"]          # list of {"bytes": ..., "path": ...}
        self._labels = cols["fine_label"]  # list of int
        hf = json.loads(table.schema.metadata[b"huggingface"].decode())
        self.classes = hf["info"]["features"]["fine_label"]["names"]

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, i):
        img = Image.open(io.BytesIO(self._imgs[i]["bytes"])).convert("RGB")
        return img, self._labels[i]


def get_cifar100():
    return CIFAR100Parquet(split="train")


def aug_transform(resolution):
    """Train-time augmentation (random resized crop + flip) — essential on small
    fine-grained datasets like CUB where resize-only training overfits badly."""
    return transforms.Compose([
        transforms.RandomResizedCrop(resolution, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


def resize_transform(resolution):
    """Resize -> tensor -> ImageNet-normalize, for running a stage at a
    given input resolution."""
    return transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


class ResizedDataset(Dataset):
    """Applies a transform to a (transform-less) image dataset on access.

    The base dataset is expected to yield (PIL image, label); this wraps each
    image with the per-stage resize/normalize transform.
    """

    def __init__(self, base, transform):
        self.base = base
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img, label = self.base[i]
        return self.transform(img), label
