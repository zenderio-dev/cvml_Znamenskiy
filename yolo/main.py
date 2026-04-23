import cv2
from ultralytics import YOLO


def main():
    model_path = r"C:\runs\detect\runs\cube_sphere_yolo-3\weights\best.pt"
    conf_threshold = 0.1

    model = YOLO(model_path)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            source=frame,
            conf=conf_threshold,
            verbose=False
        )

        annotated_frame = frame.copy()

        for result in results:
            boxes = result.boxes
            names = result.names

            if boxes is None:
                continue

            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                cls_id = int(box.cls[0].cpu().item())
                conf = float(box.conf[0].cpu().item())

                x1, y1, x2, y2 = xyxy
                label = f"{names[cls_id]} {conf:.2f}"

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

        cv2.imshow("YOLO Detection", annotated_frame)

        key = cv2.waitKey(1)
        if key == 27 or key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()