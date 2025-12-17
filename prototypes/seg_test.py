import cv2
import numpy as np
import mediapipe as mp


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    mp_seg = mp.solutions.selfie_segmentation
    segmenter = mp_seg.SelfieSegmentation(model_selection=1)

    printed = False
    window_mask = "Segmentation Mask (grayscale)"
    window_comp = "Composite (solid background)"
    cv2.namedWindow(window_mask, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_comp, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            frame_bgr = cv2.flip(frame_bgr, 1)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Run segmentation
            res = segmenter.process(frame_rgb)
            m = res.segmentation_mask  # typically shape (H, W), float32-ish, values ~[0,1]

            if not printed:
                printed = True
                print("frame_bgr:", type(frame_bgr), frame_bgr.shape, frame_bgr.dtype)
                print("frame_rgb:", type(frame_rgb), frame_rgb.shape, frame_rgb.dtype)
                print("mask m:", type(m), m.shape, m.dtype)
                print("mask min/max:", float(np.min(m)), float(np.max(m)))

            # Visualize mask as an image (0..255)
            mask_vis = (np.clip(m, 0.0, 1.0) * 255).astype(np.uint8)
            cv2.imshow(window_mask, mask_vis)

            # Composite: keep foreground, replace background with a solid color
            # Make mask shape (H, W, 1) to broadcast over 3 channels
            mask01 = np.clip(m, 0.0, 1.0).astype(np.float32)[..., None]

            fg = frame_bgr.astype(np.float32)
            bg = np.full_like(fg, (30, 30, 30), dtype=np.float32)  # dark gray BGR

            comp = mask01 * fg + (1.0 - mask01) * bg
            comp = comp.astype(np.uint8)

            cv2.imshow(window_comp, comp)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if cv2.getWindowProperty(window_comp, cv2.WND_PROP_VISIBLE) < 1:
                break

    finally:
        segmenter.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
