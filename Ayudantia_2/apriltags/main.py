from pupil_apriltags import Detector
import cv2
import numpy as np

# cap = cv2.VideoCapture("data/video2.mp4")
cap = cv2.VideoCapture(0)

at_detector = Detector(
   families="tag36h11",
   nthreads=1,
   quad_decimate=1.0,  # downscale factor
   quad_sigma=0.0, # blur sigma 
   refine_edges=1, #refine (? )
   decode_sharpening=0.25, # 
   debug=0
)

while True:
   ret, frame = cap.read()
   if not ret:
      break

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
   cv2.imshow("AprilTags Video", frame)
   cv2.waitKey(1)


cap.release()
cv2.destroyAllWindows()
