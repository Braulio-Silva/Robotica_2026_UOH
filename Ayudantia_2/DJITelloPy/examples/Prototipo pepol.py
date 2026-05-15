from ultralytics import YOLO
import cv2
import torch


def draw_people_counter_interface(frame, person_count):
    """
    Dibuja una interfaz simple que muestra solamente
    el contador de personas detectadas.
    """

    overlay = frame.copy()

    # Panel superior izquierdo
    cv2.rectangle(
        overlay,
        (10, 10),
        (260, 75),
        (0, 0, 0),
        -1
    )

    # Transparencia del panel
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Texto del contador
    cv2.putText(
        frame,
        f"Personas detectadas: {person_count}",
        (25, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    return frame


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

            img = frame.copy()
            person_count = 0

            for r in results:
                for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                    x1, y1, x2, y2 = map(int, box)
                    label = r.names[int(cls)]

                    # Mostrar y contar solo personas
                    if label == "person":
                        person_count += 1

                        text = f"Persona {conf:.2f}"

                        cv2.rectangle(
                            img,
                            (x1, y1),
                            (x2, y2),
                            (0, 255, 0),
                            2
                        )

                        cv2.putText(
                            img,
                            text,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2
                        )

            # Interfaz con contador de personas
            img = draw_people_counter_interface(img, person_count)

            cv2.imshow("YOLO - Contador de Personas", img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    else:
        results = model.predict(source, device=device, verbose=False)

        for r in results:
            img = r.orig_img.copy()
            person_count = 0

            for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                x1, y1, x2, y2 = map(int, box)
                label = r.names[int(cls)]

                # Mostrar y contar solo personas
                if label == "person":
                    person_count += 1

                    text = f"Persona {conf:.2f}"

                    cv2.rectangle(
                        img,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        img,
                        text,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )

            # Interfaz con contador de personas
            img = draw_people_counter_interface(img, person_count)

            cv2.imshow("YOLO - Contador de Personas", img)
            cv2.waitKey(0)

        cv2.destroyAllWindows()


def detect_and_draw(model, frame, device):
    results = model.predict(frame, device=device, verbose=False)

    img = frame.copy()
    person_count = 0

    for r in results:
        for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
            x1, y1, x2, y2 = map(int, box)
            label = r.names[int(cls)]

            # Mostrar y contar solo personas
            if label == "person":
                person_count += 1

                text = f"Persona {conf:.2f}"

                cv2.rectangle(
                    img,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    img,
                    text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

    # Interfaz con contador de personas
    img = draw_people_counter_interface(img, person_count)

    return img


if __name__ == "__main__":
    model = YOLO("yolo26m.pt")

    cam = True
    source = "./data/image.jpg"

    run_detection(model, source, cam=cam)