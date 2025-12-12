from __future__ import annotations
from dataclasses import dataclass
import mediapipe as mp

@dataclass
class FaceBox:
    x: int
    y: int
    w: int
    h: int

class FaceDetector:
    def __init__(self, min_confidence: float = 0.6, model_selection: int = 0):
        self._mp_fd = mp.solutions.face_detection
        self._detector = self._mp_fd.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_confidence
        )

    def detect_primary(self, frame_rgb) -> FaceBox | None:
        """Return the first face box (simple). You can upgrade to biggest/confident later."""
        res = self._detector.process(frame_rgb)
        if not res.detections:
            return None

        det = res.detections[0]
        r = det.location_data.relative_bounding_box
        H, W = frame_rgb.shape[:2]

        x = int(r.xmin * W)
        y = int(r.ymin * H)
        w = int(r.width * W)
        h = int(r.height * H)

        # clamp to image bounds
        x = max(0, x)
        y = max(0, y)
        w = max(1, min(W - x, w))
        h = max(1, min(H - y, h))

        return FaceBox(x, y, w, h)

    def close(self) -> None:
        self._detector.close()
