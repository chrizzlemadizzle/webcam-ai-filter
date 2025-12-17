import cv2
from src.face_detection import FaceDetector
from src.inspect_utils import inspect_frame
from src.overlay import load_bgra, resize_to_width, alpha_blend_bgra_onto_bgr
from src.segmentation import BackgroundSegmenter, apply_background_mode

INSPECT_FRAME = True

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")
    
    overlay = load_bgra("assets/glasses.png")

    window = "Webcam AI Filter"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    face = FaceDetector(min_confidence=0.6)

    segmenter = BackgroundSegmenter(1)
    bg_mode = "none" # none | blur | solid | image
    bg_image = cv2.imread("assets/background.png")
    if bg_image is None:
        bg_image = None
            
    inspected = False

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            
            if  not inspected and INSPECT_FRAME:
                inspect_frame(cap, frame_bgr)
                inspected = True
            
            frame_bgr = cv2.flip(frame_bgr, 1)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            if bg_mode != "none":
                mask01 = segmenter.person_mask(
                    frame_rgb,
                    downscale_width=320,
                    blur_ksize=15,
                )
                frame_bgr = apply_background_mode(
                    frame_bgr,
                    mask01,
                    mode=bg_mode,
                    bg_image_bgr=bg_image,
                    solid_bgr=(30, 30, 30),
                    blur_ksize=31,
                    threshold=0.5,
                )

            box = face.detect_primary(frame_rgb)
            if box:
                # draw box
                cv2.rectangle(frame_bgr, (box.x, box.y), (box.x + box.w, box.y + box.h), (0, 255, 0), 2)

                # resize overlay to face width
                target_w = max(1, int(box.w * 1.05))
                ov = resize_to_width(overlay, target_w)

                # overlay position within box
                ox = box.x + (box.w - ov.shape[1]) // 2
                oy = box.y + int(box.h * 0.1)

                alpha_blend_bgra_onto_bgr(frame_bgr, ov, ox, oy)

            cv2.imshow(window, frame_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("b"):
                bg_mode = "none" if bg_mode != "none" else "blur"
            elif key == ord("1"):
                bg_mode = "solid"
            elif key == ord("2"):
                bg_mode = "blur"
            elif key == ord("3"):
                bg_mode = "image"
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break

    finally:
        segmenter.close()
        face.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
