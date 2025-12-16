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