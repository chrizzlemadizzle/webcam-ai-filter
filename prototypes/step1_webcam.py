import cv2
import numpy as np
from src.inspect_utils import inspect_frame

def main():
    cap = cv2.VideoCapture(0)  # 0 = default camera
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (try a different index).")

    printed = False

    while True:
        ok, frame = cap.read()
                # --- INSPECTION (prints once) ---
        
        if not ok:
            break
        if not printed:
            inspect_frame(cap, frame)
            printed = True
            
        frame = cv2.flip(frame, 1)  # mirror like a selfie
        cv2.imshow("Step 1 - Webcam", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
