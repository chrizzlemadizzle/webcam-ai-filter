import cv2
import mediapipe as mp

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    mp_fd = mp.solutions.face_detection
    detector = mp_fd.FaceDetection(model_selection=0, min_detection_confidence=0.6)

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        frame_bgr = cv2.flip(frame_bgr, 1)

        # MediaPipe expects RGB (OpenCV gives BGR)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        result = detector.process(frame_rgb)

        h, w = frame_bgr.shape[:2]

        if result.detections:
            # pick first detection (good enough for now)
            det = result.detections[0]
            r = det.location_data.relative_bounding_box

            x = int(r.xmin * w)
            y = int(r.ymin * h)
            bw = int(r.width * w)
            bh = int(r.height * h)

            x = max(0, x)
            y = max(0, y)
            bw = max(1, min(w - x, bw))
            bh = max(1, min(h - y, bh))

            cv2.rectangle(frame_bgr, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

        cv2.imshow("Step 2 - Face Detection", frame_bgr)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    detector.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
