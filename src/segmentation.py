from __future__ import annotations
import cv2
import numpy as np
import mediapipe as mp

class BackgroundSegmenter:
    """
    MediaPipe Selfie Segmentation wrapper.
    Produces a soft mask in [0, 1] where 1 = person (foreground).
    """

    def __init__(self, model_selection: int = 1):
        # model_selection: 0 (general) / 1 (landscape) depending on mediapipe version;
        # either is fine.
        self._mp_seg = mp.solutions.selfie_segmentation
        self._segmenter = self._mp_seg.SelfieSegmentation(model_selection=model_selection)

    def person_mask(
            self,
            frame_rgb: np.ndarray,
            downscale_width: int = 320,
            blur_ksize: int = 15,
    ) -> np.ndarray:
        """
        Returns mask of shape (H, W, 1), float32 in [0,1].
        Downscales for speed, then upscales back.
        """
        H, W = frame_rgb.shape[:2]

        if downscale_width and W > downscale_width:
            scale = downscale_width / float(W)
            small = cv2.resize(
                frame_rgb,
                (downscale_width, max(1, int(H * scale))),
                interpolation=cv2.INTER_AREA,
            )
            res = self._segmenter.process(small)
            m = res.segmentation_mask # (h_small, w_small), float
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)
        else:
            res = self._segmenter.process(frame_rgb)
            m = res.segmentation_mask # (H, W), float

        if blur_ksize and blur_ksize >= 3 and blur_ksize % 2 == 1:
            m = cv2.GaussianBlur(m, (blur_ksize, blur_ksize), 0)

        m = np.clip(m, 0.0, 1.0).astype(np.float32)
        return m[..., None] # (H, W, 1)
    
    def close(self) -> None:
        self._segmenter.close()


def apply_background_mode(
        frame_bgr: np.ndarray,
        mask01: np.ndarray,
        mode: str,
        bg_image_bgr: np.ndarray | None = None,
        solid_bgr: tuple[int, int, int] = (30, 30, 30),
        blur_ksize: int = 31,
        threshold: float = 0.5,
) -> np.ndarray:
    """
    mode: "none" | "blur" | "solid" | "image"
    mask01: (H,W,1) float mask, 1 = foreground.
    threshold: used to make the mask a bit more decisive (optional).
    """
    if mode == "none":
        return frame_bgr
    
    H, W = frame_bgr.shape[:2]

    #optional: bias the soft mask a bit (keeps edges soft but reduces bleed)
    if threshold is not None:
        m = (mask01 - threshold) / max(1e-6, (1.0 - threshold))
        m = np.clip(m, 0.0, 1.0).astype(np.float32)
    else:
        m = mask01.astype(np.float32)

    fg = frame_bgr.astype(np.float32)

    if mode == "blur":
        k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        bg = cv2.GaussianBlur(frame_bgr, (k, k), 0).astype(np.float32)
    elif mode == "solid":
        bg = np.full((H, W, 3), solid_bgr, dtype=np.float32)
    elif mode == "image":
        if bg_image_bgr is None:
            return frame_bgr
        bg = cv2.resize(bg_image_bgr, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)
    else:
        return frame_bgr
    
    out = m * fg + (1.0 - m) * bg
    return out.astype(np.uint8)