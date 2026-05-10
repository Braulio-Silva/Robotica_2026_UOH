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
# tello.takeoff()
# time.sleep(2)

# =======================
# SAFE LAND FUNCTION
# =======================
def safe_land():
    print("LANDING...")

    # detener rc
    for _ in range(5):
        tello.send_rc_control(0,0,0,0)
        time.sleep(0.05)

    time.sleep(0.3)

    # intentar land varias veces
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

at_detector = Detector(
   families="tag36h11",
   nthreads=1,
   quad_decimate=1.0,  # downscale factor
   quad_sigma=0.0, # blur sigma 
   refine_edges=1, #refine (? )
   decode_sharpening=0.25, # 
   debug=0
)

model = YOLO("yolo26m.pt")

while True:
    frame = frame_read.frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if frame is None:
        continue

    cv2.imwrite(os.path.join(save_dir, f"{frame_id:06d}.png"), frame)


    # ===========================
    #      DETECCION APRILTAG 
    # ===========================


    # pupil apriltag reconoce en gray no en rgb
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #sacamos los resultados 
    results = at_detector.detect(gray)

    # obtenemos los resultados de todos los apriltags
    for r in results:
        corners = r.corners.astype(int)

        # dibujar bounding box
        for i in range(4):
            pt1 = tuple(corners[i])
            pt2 = tuple(corners[(i+1) % 4])
            cv2.line(frame, pt1, pt2, (0,255,0), 2)

        # centro
        center = r.center
        cx, cy = int(center[0]), int(center[1])
        cv2.circle(frame, (cx, cy), 5, (0,0,255), -1)

        # tamaño en píxeles (promedio de ancho y alto)
        width = np.linalg.norm(r.corners[0] - r.corners[1])
        height = np.linalg.norm(r.corners[1] - r.corners[2])
        size_px = (width + height) / 2

        # texto
        text = f"ID:{r.tag_id} | size:{size_px:.1f}px | x:{cx}, y:{cy}"

        cv2.putText(
            frame,
            text,
            (cx + 10, cy + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255,0,0),
            2
            )



    # ===========================
    #      DETECCION personas 
    # ===========================

    results2 = model.predict(frame, device='cpu', verbose=False)

    for r in results2:
        for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
            x1, y1, x2, y2 = map(int, box)
            label = r.names[int(cls)]
            text = f"{label} {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # =======================
    # DISPLAY
    # =======================
    cv2.imshow("Tello", frame)

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
    #if now - last_frame_time >= interval:
    frame_id += 1
    last_frame_time = now

    # =======================
    # CONTROL
    # =======================
    lr, fb, ud, yaw = 0, 0, 0, 0

    if 'w' in pressed: fb = speed
    if 's' in pressed: fb = -speed
    if 'a' in pressed: lr = -speed
    if 'd' in pressed: lr = speed
    if 'r' in pressed: ud = speed
    if 'f' in pressed: ud = -speed
    if 'q' in pressed: yaw = -speed
    if 'e' in pressed: yaw = speed

    # =======================
    # SEND RC
    # =======================
    # if now - last_rc_time > rc_interval:
    #     tello.send_rc_control(lr, fb, ud, yaw)
    #     last_rc_time = now

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
