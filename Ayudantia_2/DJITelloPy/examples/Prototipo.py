from djitellopy import Tello
import cv2, time, os, sys, signal, platform
from datetime import datetime
from pupil_apriltags import Detector
import numpy as np
from ultralytics import YOLO

# =======================
# DETECTAR OS
# =======================
OS = platform.system()
USE_PYNPUT = (OS == "Darwin")  # macOS
print(f"USE PYNPUT: {USE_PYNPUT}")

if USE_PYNPUT:
    from pynput import keyboard
    keys = set()

    def on_press(key):
        try:
            keys.add(key.char)
        except:
            if key == keyboard.Key.esc:
                keys.add('esc')

    def on_release(key):
        try:
            keys.discard(key.char)
        except:
            if key == keyboard.Key.esc:
                keys.discard('esc')

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

# =======================
# TELLO
# =======================
tello = Tello()
tello.connect()
print("Battery:", tello.get_battery())

tello.streamoff()
tello.streamon()
frame_read = tello.get_frame_read()
time.sleep(2)

# =======================
# SAVE DIR
# =======================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = os.path.join("images", timestamp)
os.makedirs(save_dir, exist_ok=True)

# =======================
# TAKEOFF
# =======================
tello.takeoff()
time.sleep(2)

# =======================
# SAFE LAND FUNCTION
# =======================
def safe_land():
    print("LANDING...")

    # Detener movimiento
    for _ in range(5):
        tello.send_rc_control(0, 0, 0, 0)
        time.sleep(0.05)

    time.sleep(0.3)

    # Intentar aterrizar varias veces
    for _ in range(3):
        try:
            tello.land()
            print("LANDED OK")
            return
        except:
            time.sleep(0.5)

    print("FORCED EMERGENCY")
    tello.emergency()

# =======================
# CTRL+C
# =======================
def handler(sig, frame):
    safe_land()
    tello.streamoff()
    tello.end()
    sys.exit(0)

signal.signal(signal.SIGINT, handler)

# =======================
# LOOP
# =======================
fps = 5
interval = 1.0 / fps
last_frame_time = time.time()
frame_id = 0

speed = 40

# RC rate limit
last_rc_time = 0
rc_interval = 0.05

# =======================
# APRILTAG DETECTOR
# =======================
at_detector = Detector(
    families="tag36h11",
    nthreads=1,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0
)

# =======================
# MODELO YOLO
# =======================
model = YOLO("yolo26m.pt")

# =======================
# INTERFAZ / TELEMETRIA
# =======================
# Se consulta la bateria y altura cada cierto tiempo
# para no saturar la comunicacion con el dron.
battery = tello.get_battery()
height_cm = 0
last_telemetry_time = 0
telemetry_interval = 1.0

def draw_interface(frame, battery, height_cm, person_count):
    """
    Dibuja una interfaz simple sobre el video:
    - Bateria
    - Altura
    - Contador de personas
    - Barra visual de bateria
    - Ayuda de controles
    """

    h, w = frame.shape[:2]

    # Panel superior izquierdo
    panel_x1, panel_y1 = 10, 10
    panel_x2, panel_y2 = 310, 135

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (panel_x1, panel_y1),
        (panel_x2, panel_y2),
        (0, 0, 0),
        -1
    )

    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Color de bateria segun nivel
    if battery >= 50:
        battery_color = (0, 255, 0)
    elif battery >= 25:
        battery_color = (0, 255, 255)
    else:
        battery_color = (0, 0, 255)

    # Titulo
    cv2.putText(
        frame,
        "INTERFAZ TELLO",
        (25, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    # Bateria
    cv2.putText(
        frame,
        f"Bateria: {battery}%",
        (25, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        battery_color,
        2
    )

    # Altura
    cv2.putText(
        frame,
        f"Altura: {height_cm} cm",
        (25, 92),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # Personas
    cv2.putText(
        frame,
        f"Personas: {person_count}",
        (25, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # Barra visual de bateria
    bar_x1, bar_y1 = 185, 52
    bar_x2, bar_y2 = 290, 70

    cv2.rectangle(
        frame,
        (bar_x1, bar_y1),
        (bar_x2, bar_y2),
        (255, 255, 255),
        2
    )

    fill_width = int(
        (bar_x2 - bar_x1 - 4) * max(0, min(battery, 100)) / 100
    )

    cv2.rectangle(
        frame,
        (bar_x1 + 2, bar_y1 + 2),
        (bar_x1 + 2 + fill_width, bar_y2 - 2),
        battery_color,
        -1
    )

    # Ayuda de controles
    help_text = "W/S adelante-atras | A/D lateral | R/F subir-bajar | Q/E girar | L/ESC aterrizar"

    cv2.putText(
        frame,
        help_text,
        (10, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1
    )

    return frame

# =======================
# LOOP PRINCIPAL
# =======================
while True:
    frame = frame_read.frame

    if frame is None:
        continue

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    cv2.imwrite(os.path.join(save_dir, f"{frame_id:06d}.png"), frame)

    # ===========================
    # DETECCION APRILTAG
    # ===========================

    # Pupil apriltag reconoce en gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resultados AprilTags
    results = at_detector.detect(gray)

    for r in results:
        corners = r.corners.astype(int)

        # Dibujar bounding box del AprilTag
        for i in range(4):
            pt1 = tuple(corners[i])
            pt2 = tuple(corners[(i + 1) % 4])
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

        # Centro del AprilTag
        center = r.center
        cx, cy = int(center[0]), int(center[1])
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        # Tamano en pixeles
        width = np.linalg.norm(r.corners[0] - r.corners[1])
        height = np.linalg.norm(r.corners[1] - r.corners[2])
        size_px = (width + height) / 2

        text = f"ID:{r.tag_id} | size:{size_px:.1f}px | x:{cx}, y:{cy}"

        cv2.putText(
            frame,
            text,
            (cx + 10, cy + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2
        )

    # ===========================
    # DETECCION PERSONAS
    # ===========================

    person_count = 0

    results2 = model.predict(
        frame,
        device='cpu',
        verbose=False
    )

    for r in results2:
        for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
            x1, y1, x2, y2 = map(int, box)
            label = r.names[int(cls)]

            # Contar solamente personas
            if label == "person":
                person_count += 1

                text = f"{label} {conf:.2f}"

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

    # =======================
    # TELEMETRIA
    # =======================
    now = time.time()

    if now - last_telemetry_time >= telemetry_interval:
        try:
            battery = tello.get_battery()
        except Exception:
            pass

        try:
            height_cm = tello.get_height()
        except Exception:
            pass

        last_telemetry_time = now

    # =======================
    # DISPLAY / INTERFAZ
    # =======================
    frame = draw_interface(
        frame,
        battery,
        height_cm,
        person_count
    )


    os.makedirs('processed',exist_ok=True)
    cv2.imwrite(os.path.join('processed', f"{frame_id:06d}.png"), frame)
    cv2.imshow("Tello - Interfaz", frame)

    # =======================
    # TECLADO
    # =======================
    if USE_PYNPUT:
        cv2.pollKey()
        pressed = keys.copy()
    else:
        key = cv2.waitKey(1) & 0xFF
        pressed = set()

        if key != 255:
            pressed.add(chr(key))

    # =======================
    # SAVE
    # =======================
    now = time.time()
    frame_id += 1
    last_frame_time = now

    # =======================
    # CONTROL
    # =======================
    lr, fb, ud, yaw = 0, 0, 0, 0

    if 'w' in pressed:
        fb = speed

    if 's' in pressed:
        fb = -speed

    if 'a' in pressed:
        lr = -speed

    if 'd' in pressed:
        lr = speed

    if 'r' in pressed:
        ud = speed

    if 'f' in pressed:
        ud = -speed

    if 'q' in pressed:
        yaw = -speed

    if 'e' in pressed:
        yaw = speed

    # =======================
    # SEND RC
    # =======================
    # Descomenta esto cuando quieras enviar movimiento real al dron.
    # Mientras esta comentado, solo muestra la camara y detecciones.
    #
    if now - last_rc_time > rc_interval:
         tello.send_rc_control(lr, fb, ud, yaw)
         last_rc_time = now

    # =======================
    # LAND
    # =======================
    if 'l' in pressed or 'esc' in pressed:
        safe_land()
        break

# =======================
# CLEANUP
# =======================
tello.streamoff()
tello.end()
cv2.destroyAllWindows()