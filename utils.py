import numpy as np


def _add_glare(x, p=0.9, max_alpha=0.6):
    """Add a simple lens-glare spot with a radial falloff.
    Expects x in [0,255] float or uint8, shape (H,W,C) with C=1 or 3."""
    if np.random.rand() > p:
        return x

    x = x.astype(np.float32, copy=True)
    h, w = x.shape[:2]
    # random glare center (often near top/edge looks natural)
    cx = np.random.uniform(-0.2 * w, 1.2 * w)
    cy = np.random.uniform(-0.2 * h, 0.7 * h)
    # random radius
    R = np.random.uniform(0.2, 0.6) * min(h, w)

    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mask = np.clip(1.0 - dist / R, 0.0, 1.0)
    mask = mask ** 2  # smoother falloff

    # intensity and (slightly warm) color tint
    alpha = np.random.uniform(0.25, max_alpha)
    if x.shape[-1] == 3:
        tint = np.array([1.0, 0.95, 0.85], dtype=np.float32)  # warm glare
        glare = (255.0 * alpha * mask)[..., None] * tint[None, None, :]
    else:
        glare = (255.0 * alpha * mask)[..., None]

    x += glare
    #np.clip(x, 0, 255, out=x)
    return x

def glare_then_preprocess(x):
    x = _add_glare(x, p=0.3, max_alpha=0.6)
    return x/255  # your existing preprocessing

def preprocess_input(x):
    return x/255  # your existing preprocessing
