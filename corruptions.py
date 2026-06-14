"""Self-contained ImageNet-C-style corruptions (no external deps).

Each corruption maps (PIL.Image RGB, severity in 1..5) -> PIL.Image RGB. These
approximate the Hendrycks & Dietterich (2019) ImageNet-C corruptions using only
numpy + PIL; the severity tables follow their scale closely but are not bit-exact
(the reference uses scipy/skimage/cv2 kernels we deliberately avoid here).

Used to build a corrupted Imagenette eval set for routing-robustness tests:
applying a corruption shifts the input distribution, which should raise the cheap
stages' harm rate and exercise the router under stress.
"""

import io

import numpy as np
from PIL import Image, ImageEnhance


def _to_arr(img):
    return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def _to_img(arr):
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))


def gaussian_noise(img, severity):
    c = [0.08, 0.12, 0.18, 0.26, 0.38][severity - 1]
    x = _to_arr(img)
    return _to_img(x + np.random.normal(size=x.shape, scale=c))


def shot_noise(img, severity):
    c = [60, 25, 12, 5, 3][severity - 1]
    x = _to_arr(img)
    return _to_img(np.random.poisson(x * c) / float(c))


def impulse_noise(img, severity):
    c = [0.03, 0.06, 0.09, 0.17, 0.27][severity - 1]
    x = _to_arr(img).copy()
    mask = np.random.choice(3, size=x.shape[:2], p=[1 - c, c / 2, c / 2])
    x[mask == 1] = 0.0       # pepper
    x[mask == 2] = 1.0       # salt
    return _to_img(x)


def gaussian_blur(img, severity):
    from PIL import ImageFilter
    c = [1, 2, 3, 4, 6][severity - 1]
    return img.convert("RGB").filter(ImageFilter.GaussianBlur(radius=c))


def defocus_blur(img, severity):
    # Approximated with a Gaussian kernel (reference uses a disk kernel via cv2).
    from PIL import ImageFilter
    c = [2, 3, 4, 5, 7][severity - 1]
    return img.convert("RGB").filter(ImageFilter.GaussianBlur(radius=c))


def brightness(img, severity):
    c = [1.1, 1.2, 1.3, 1.4, 1.5][severity - 1]
    return ImageEnhance.Brightness(img.convert("RGB")).enhance(c)


def contrast(img, severity):
    c = [0.6, 0.5, 0.4, 0.3, 0.2][severity - 1]
    return ImageEnhance.Contrast(img.convert("RGB")).enhance(c)


def pixelate(img, severity):
    c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]
    img = img.convert("RGB")
    w, h = img.size
    small = img.resize((max(1, int(w * c)), max(1, int(h * c))), Image.BOX)
    return small.resize((w, h), Image.NEAREST)


def jpeg_compression(img, severity):
    c = [25, 18, 15, 10, 7][severity - 1]
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=c)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


CORRUPTIONS = {
    "gaussian_noise": gaussian_noise,
    "shot_noise": shot_noise,
    "impulse_noise": impulse_noise,
    "gaussian_blur": gaussian_blur,
    "defocus_blur": defocus_blur,
    "brightness": brightness,
    "contrast": contrast,
    "pixelate": pixelate,
    "jpeg_compression": jpeg_compression,
}


def corrupt(img, name, severity):
    """Apply a named corruption at the given severity (1..5)."""
    if name not in CORRUPTIONS:
        raise ValueError(f"unknown corruption '{name}'; choices: {sorted(CORRUPTIONS)}")
    if not 1 <= severity <= 5:
        raise ValueError(f"severity must be in 1..5, got {severity}")
    return CORRUPTIONS[name](img, severity)
