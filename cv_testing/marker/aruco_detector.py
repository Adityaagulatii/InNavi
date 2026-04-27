"""
WayfinderAI — ArUco Marker Aisle Detector + Live Minimap
=========================================================
Route: ENTR -> A152 -> A100 -> A1 -> A2 -> A100 -> A34 -> A101 -> CHEC
       (milk, eggs, pasta demo route)

Hold printed ArUco markers in front of the webcam.
When A152 (marker 0) or A100 (marker 1) is detected, position
updates on the minimap.

Display:
  Left  = webcam with detection boxes + aisle banner
  Right = live minimap

Requires:  pip install opencv-contrib-python
Press Q to quit.
"""

import tkinter as tk
import csv
import cv2
import numpy as np
import time
from datetime import datetime

# ─────────────────────────────────────────────────────────────
# Route + minimap definition  (same as ocr_webcam_test.py)
# ─────────────────────────────────────────────────────────────
MINIMAP_W, MINIMAP_H = 500, 480

ROUTE_NODES = [
    {"id": "entrance_left", "label": "ENTR", "short": "Entrance",      "pos": (60,  420)},
    {"id": "152",           "label": "A152", "short": "Bakery",         "pos": (60,  310)},
    {"id": "100",           "label": "A100", "short": "Dairy",          "pos": (190, 230)},
    {"id": "1",             "label": "A1",   "short": "Dairy & Bakery", "pos": (190, 350)},
    {"id": "2",             "label": "A2",   "short": "Dry Goods",      "pos": (300, 350)},
    {"id": "34",            "label": "A34",  "short": "Yogurt",         "pos": (370, 230)},
    {"id": "101",           "label": "A101", "short": "Meat",           "pos": (450, 230)},
    {"id": "checkout_1",    "label": "CHEC", "short": "Checkout",       "pos": (460, 390)},
]

ROUTE_ORDER = [
    "entrance_left", "152", "100", "1", "2", "1", "100", "34", "101", "checkout_1"
]

MARKER_TO_NODE = {
    0: "152",
    1: "100",
}

SCAN_STOPS  = {"152", "100"}
NODE_LOOKUP = {n["id"]: n for n in ROUTE_NODES}

# ─────────────────────────────────────────────────────────────
# Minimap draw  (identical to ocr_webcam_test.py)
# ─────────────────────────────────────────────────────────────
def draw_minimap(current_node_id, last_label):
    canvas = np.ones((MINIMAP_H, MINIMAP_W, 3), dtype=np.uint8) * 22

    cv2.putText(canvas, "Milk  Eggs  Pasta  Route", (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200, 200, 200), 1)

    for i in range(len(ROUTE_ORDER) - 1):
        a = NODE_LOOKUP.get(ROUTE_ORDER[i])
        b = NODE_LOOKUP.get(ROUTE_ORDER[i + 1])
        if a and b:
            cv2.line(canvas, a["pos"], b["pos"], (70, 130, 220), 2)

    for node in ROUTE_NODES:
        x, y    = node["pos"]
        nid     = node["id"]
        is_cur  = (nid == current_node_id)
        is_scan = (nid in SCAN_STOPS)

        if is_cur:
            cv2.circle(canvas, (x, y), 22, (0, 60, 200), -1)
            cv2.circle(canvas, (x, y), 22, (80, 160, 255), 2)
            cv2.circle(canvas, (x, y), 10, (0, 0, 220), -1)
            cv2.circle(canvas, (x, y),  5, (255, 80, 80), -1)
        elif is_scan:
            cv2.circle(canvas, (x, y), 18, (200, 140, 0), -1)
            cv2.circle(canvas, (x, y), 18, (255, 200, 0), 2)
        else:
            cv2.circle(canvas, (x, y), 14, (60, 90, 160), -1)
            cv2.circle(canvas, (x, y), 14, (100, 130, 200), 1)

        label      = node["label"]
        font_scale = 0.38 if len(label) > 3 else 0.42
        ts         = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
        cv2.putText(canvas, label,
                    (x - ts[0] // 2, y + ts[1] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255) if is_cur else (220, 220, 220), 1)

        ns = cv2.getTextSize(node["short"], cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)[0]
        cv2.putText(canvas, node["short"],
                    (x - ns[0] // 2, y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                    (255, 220, 100) if is_scan else (140, 140, 140), 1)

    if last_label:
        cv2.rectangle(canvas, (0, MINIMAP_H - 38), (MINIMAP_W, MINIMAP_H), (0, 70, 0), -1)
        cv2.putText(canvas, last_label, (10, MINIMAP_H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (80, 255, 80), 1)

    cv2.circle(canvas, (12, MINIMAP_H - 90), 7, (0, 0, 220), -1)
    cv2.circle(canvas, (12, MINIMAP_H - 90), 3, (255, 80, 80), -1)
    cv2.putText(canvas, "You are here", (24, MINIMAP_H - 86),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 180, 180), 1)
    cv2.circle(canvas, (12, MINIMAP_H - 68), 7, (200, 140, 0), -1)
    cv2.putText(canvas, "Scan marker to update", (24, MINIMAP_H - 64),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 180, 180), 1)

    return canvas

# ─────────────────────────────────────────────────────────────
# ArUco setup
# ─────────────────────────────────────────────────────────────
aruco_dict     = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params   = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
print("ArUco detector ready (DICT_4X4_50)")

# ─────────────────────────────────────────────────────────────
# Webcam
# ─────────────────────────────────────────────────────────────
print("Opening webcam...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open webcam.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ─────────────────────────────────────────────────────────────
# CSV log
# ─────────────────────────────────────────────────────────────
log_file = open("aruco_detector_log.csv", "w", newline="")
logger   = csv.writer(log_file)
logger.writerow(["run_started", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "", ""])
logger.writerow(["timestamp", "aisle_detected", "detection_time_ms", "map_update_time_ms"])
log_file.flush()
print("Log reset — aruco_detector_log.csv is fresh for this run.")

current_node   = "entrance_left"
last_label     = ""
detected_flash = 0
fps_timer      = time.time()
fps_display    = 0.0
frame_count    = 0

_r = tk.Tk(); SCREEN_W, SCREEN_H = _r.winfo_screenwidth(), _r.winfo_screenheight(); _r.destroy()
WIN = "WayfinderAI — ArUco Detector + Minimap  (Q to quit)"
cv2.namedWindow(WIN, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("\n=== ARUCO DETECTOR + MINIMAP ===")
print("Hold marker_0 (A152) or marker_1 (A100) in front of the camera.")
print("Press Q to quit.\n")

# ─────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    h, w = frame.shape[:2]

    # ── ArUco detection (grayscale = fastest path) ────────────
    frame_start  = time.time()
    gray         = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco_detector.detectMarkers(gray)
    detection_ms = (time.time() - frame_start) * 1000

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        for i, marker_id in enumerate(ids.flatten()):
            mid = int(marker_id)
            if mid not in MARKER_TO_NODE:
                continue

            new_node  = MARKER_TO_NODE[mid]
            node_info = NODE_LOOKUP[new_node]

            # Label above marker
            pts = corners[i][0].astype(int)
            cx  = int(pts[:, 0].mean())
            cy  = int(pts[:, 1].min()) - 10
            chip = f"ID {mid}  {node_info['label']}  {node_info['short']}"
            (tw, th), _ = cv2.getTextSize(chip, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (cx - tw // 2 - 4, cy - th - 4),
                          (cx + tw // 2 + 4, cy + 6), (255, 255, 255), -1)
            cv2.putText(frame, chip, (cx - tw // 2, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 0), 2)

            if new_node != current_node:
                map_ms         = (time.time() - frame_start) * 1000
                current_node   = new_node
                last_label     = f"Scanned: {node_info['label']}  ->  {node_info['short']}"
                detected_flash = 50
                logger.writerow([
                    datetime.now().strftime("%H:%M:%S"),
                    node_info["label"],
                    f"{detection_ms:.0f}",
                    f"{map_ms:.0f}",
                ])
                log_file.flush()
                print(f"  + Marker {mid} -> {node_info['label']}  ({detection_ms:.0f}ms detect / {map_ms:.0f}ms total)")

    if detected_flash > 0:
        detected_flash -= 1

    # ── Top banner ────────────────────────────────────────────
    if ids is not None and any(int(m) in MARKER_TO_NODE for m in ids.flatten()):
        cv2.rectangle(frame, (0, 0), (w, 44), (0, 110, 0), -1)
        labels = [NODE_LOOKUP[MARKER_TO_NODE[int(m)]]["label"]
                  for m in ids.flatten() if int(m) in MARKER_TO_NODE]
        cv2.putText(frame, f"DETECTED:  {'  |  '.join(labels)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 255, 80), 2)
    elif detected_flash > 0:
        cv2.rectangle(frame, (0, 0), (w, 44), (0, 80, 0), -1)
        cv2.putText(frame, last_label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 255, 80), 2)
    else:
        cv2.rectangle(frame, (0, 0), (w, 44), (28, 28, 28), -1)
        cv2.putText(frame, "Hold marker_0 (A152) or marker_1 (A100) in front of camera",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (130, 130, 130), 1)

    # ── FPS ───────────────────────────────────────────────────
    if frame_count % 10 == 0:
        fps_display = 10 / (time.time() - fps_timer + 1e-6)
        fps_timer   = time.time()

    cv2.putText(frame, f"ARUCO  {fps_display:.1f} fps",
                (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)

    # ── Minimap ───────────────────────────────────────────────
    minimap  = draw_minimap(current_node, last_label if detected_flash > 0 else "")
    minimap  = cv2.resize(minimap, (MINIMAP_W, h))
    combined = np.hstack([frame, minimap])

    dw = int(SCREEN_W * 0.65)
    cv2.imshow(WIN, cv2.resize(combined, (dw, int(dw * combined.shape[0] / combined.shape[1]))))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Stopped.")
        break

cap.release()
cv2.destroyAllWindows()
log_file.close()
print("Results saved to aruco_detector_log.csv")
