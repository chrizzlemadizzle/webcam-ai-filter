import numpy as np
import cv2

def inspect_frame(cap, frame: np.ndarray) -> None:
    print("cap type:", type(cap))
    print("frame type:", type(frame))
    print("frame.shape (H,W,C):", frame.shape)
    print("frame.dtype:", frame.dtype)
    print("top-left pixel (BGR):", frame[0, 0])
