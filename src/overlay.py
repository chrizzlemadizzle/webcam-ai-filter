from __future__ import annotations
import cv2
import numpy as np

def load_bgra(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read overlay imae at: {path}")
    if img.ndim != 3 or img.shape[2] != 4:
        raise ValueError("Overlay image must be a PNG with alpha (RGBA)")
    return img

def resize_to_width(overlay_bgra: np.ndarray, target_w: int) -> np.ndarray:
    if target_w <= 1:
        return overlay_bgra
    h, w = overlay_bgra.shape[:2]
    scale = target_w / float(w)
    target_h = max(1, int(h * scale))
    return cv2.resize(overlay_bgra, (target_w, target_h), interpolation=cv2.INTER_AREA)

def alpha_blend_bgra_onto_bgr(frame_bgr: np.ndarray, overlay_bgra: np.ndarray, x: int, y: int) -> None:
    """
    Blend overlay_bgra onto frame_bgr with top-left at (x,y). Modifies frame_bgr in place.
    """
    H, W = frame_bgr.shape[:2]
    h, w = overlay_bgra.shape[:2]

    # clip to frame
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + w), min(H, y + h)
    if (x1 >= x2) or y1 >= y2:
        return
    
    ox1, oy1 = x1 - x, y1 - y
    ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 - y1)

    patch = overlay_bgra[oy1:oy2, ox1, ox2]
    overlay_bgr = patch[:, :, :3].astype(np.float32)
    alpha = (patch[:, :, 3:4].astype(np.float32)) / 255.0

    roi = frame_bgr[y1:y2, x1:x2].astype(np.float32)
    blended = alpha * overlay_bgr + (1.0 - alpha) * roi
    frame_bgr[y1:y2, x1:x2] = blended.astype(np.uint8)