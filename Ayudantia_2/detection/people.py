from ultralytics import YOLO
import cv2
import torch


def run_detection(model, source, cam=False):
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if cam:
        cap = cv2.VideoCapture(0)
        # cap = cv2.VideoCapture("http://192.168.100.46:8080/video")


        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(frame, device=device, verbose=False)

            for r in results:
                img = r.orig_img.copy()

                for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                    x1, y1, x2, y2 = map(int, box)
                    label = r.names[int(cls)]
                    text = f"{label} {conf:.2f}"

                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                cv2.imshow("YOLO - Camera", img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    else:
        results = model.predict(source, device=device, verbose=False)

        for r in results:
            img = r.orig_img.copy()

            for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                x1, y1, x2, y2 = map(int, box)
                label = r.names[int(cls)]
                text = f"{label} {conf:.2f}"

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("YOLO - Image", img)
            cv2.waitKey(0)

        cv2.destroyAllWindows()

def detect_and_draw(model, frame, device):
    results = model.predict(frame, device=device, verbose=False)

    img = frame.copy()

    for r in results:
        for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
            x1, y1, x2, y2 = map(int, box)
            label = r.names[int(cls)]
            text = f"{label} {conf:.2f}"

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return img

if __name__ == "__main__":
    model = YOLO("yolo26m.pt")

    cam = True
    source = "./data/image.jpg"

    run_detection(model, source, cam=cam)
