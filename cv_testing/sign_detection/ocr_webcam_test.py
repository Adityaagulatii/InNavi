"""
WayfinderAI — OCR Aisle Detector + Live Minimap
=================================================
Route: ENTR → A152 → A100 → A1 → A2 → A100 → A34 → A101 → CHEC
       (milk, eggs, pasta demo route)

Hold an aisle sign in front of the camera.
When A152 or A100 is detected, position updates on the minimap.

Display:
  Left  = webcam with detection boxes + aisle banner
  Right = live minimap

Press Q to quit.
"""

import tkinter as tk
import cv2
import easyocr
import numpy as np
import csv
import re
import time
from datetime import datetime

# ─────────────────────────────────────────────────────────────
# Route + minimap definition
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

SCANNABLE = {
    "152": "152", "a152": "152",
    "100": "100", "a100": "100",
}

SCAN_STOPS  = {"152", "100"}
NODE_LOOKUP = {n["id"]: n for n in ROUTE_NODES}

def normalize(text):
    return re.sub(r'\s+', '', text.lower())

# ─────────────────────────────────────────────────────────────
# Minimap draw
# ─────────────────────────────────────────────────────────────
def draw_minimap(current_node_id, last_label):
    canvas = np.ones((MINIMAP_H, MINIMAP_W, 3), dtype=np.uint8) * 22

    cv2.putText(canvas, "Milk  Eggs  Pasta  Route", (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200, 200, 200), 1)

    # Route edges
    for i in range(len(ROUTE_ORDER) - 1):
        a = NODE_LOOKUP.get(ROUTE_ORDER[i])
        b = NODE_LOOKUP.get(ROUTE_ORDER[i + 1])
        if a and b:
            cv2.line(canvas, a["pos"], b["pos"], (70, 130, 220), 2)

    # Nodes
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

    # Last scanned label at bottom of minimap
    if last_label:
        cv2.rectangle(canvas, (0, MINIMAP_H - 38), (MINIMAP_W, MINIMAP_H), (0, 70, 0), -1)
        cv2.putText(canvas, last_label, (10, MINIMAP_H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (80, 255, 80), 1)

    # Legend
    cv2.circle(canvas, (12, MINIMAP_H - 90), 7, (0, 0, 220), -1)
    cv2.circle(canvas, (12, MINIMAP_H - 90), 3, (255, 80, 80), -1)
    cv2.putText(canvas, "You are here", (24, MINIMAP_H - 86),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 180, 180), 1)
    cv2.circle(canvas, (12, MINIMAP_H - 68), 7, (200, 140, 0), -1)
    cv2.putText(canvas, "Scan sign to update", (24, MINIMAP_H - 64),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 180, 180), 1)

    return canvas


# ─────────────────────────────────────────────────────────────
# OCR pattern — any A-number sign
# ─────────────────────────────────────────────────────────────
AISLE_PATTERN = re.compile(r'\b(A\s*\d{1,3}|Aisle\s*\d{1,3})\b', re.IGNORECASE)
SCAN_PATTERN  = re.compile(r'\b(A?\s*1[05][02])\b', re.IGNORECASE)

def extract_aisle_number(text):
    m = re.search(r'\d+', text)
    return m.group() if m else text

# ─────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────
print("Loading EasyOCR… (first run downloads ~100MB)")
reader = easyocr.Reader(['en'], gpu=True)
print("Model loaded. Opening webcam…")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

log_file = open("ocr_results_log.csv", "w", newline="")
logger   = csv.writer(log_file)
logger.writerow(["run_started", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "", ""])
logger.writerow(["timestamp", "aisle_detected", "ocr_time_ms", "map_update_time_ms"])
log_file.flush()
print(f"Log reset — ocr_results_log.csv is fresh for this run.")

OCR_INTERVAL   = 3
frame_count    = 0
last_results   = []
current_node   = "entrance_left"
last_label     = ""
detected_flash = 0

_r = tk.Tk(); SCREEN_W, SCREEN_H = _r.winfo_screenwidth(), _r.winfo_screenheight(); _r.destroy()
WIN = "WayfinderAI — OCR + Minimap  (Q to quit)"
cv2.namedWindow(WIN, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("\n=== LIVE OCR + MINIMAP ===")
print("Hold A152 or A100 sign in front of the camera.")
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

    # ── OCR every N frames ────────────────────────────────────
    if frame_count % OCR_INTERVAL == 0:
        frame_start  = time.time()

        raw = reader.readtext(frame)
        ocr_ms = (time.time() - frame_start) * 1000

        last_results = [(bbox, text, conf) for bbox, text, conf in raw if conf >= 0.45]

        for _, text, conf in last_results:
            if not AISLE_PATTERN.search(text):
                continue

            key = normalize(text)
            if key in SCANNABLE:
                new_node = SCANNABLE[key]
                map_ms   = (time.time() - frame_start) * 1000
                logger.writerow([
                    datetime.now().strftime("%H:%M:%S"),
                    text.upper(),
                    f"{ocr_ms:.0f}",
                    f"{map_ms:.0f}",
                ])
                log_file.flush()

                if new_node != current_node:
                    current_node   = new_node
                    last_label     = f"Scanned: {text.upper()}  →  {NODE_LOOKUP[new_node]['short']}"
                    detected_flash = 50
                    print(f"  ✓ {text}  ({ocr_ms:.0f}ms OCR / {map_ms:.0f}ms total)  →  {NODE_LOOKUP[new_node]['short']}")

    if detected_flash > 0:
        detected_flash -= 1

    # ── Draw detection boxes ──────────────────────────────────
    aisle_match = None
    for bbox, text, conf in last_results:
        pts      = np.array(bbox, dtype=np.int32)
        is_aisle = bool(AISLE_PATTERN.search(text))

        if is_aisle:
            aisle_match = text
            cv2.polylines(frame, [pts], True, (0, 255, 0), 3)
            cv2.putText(frame, f"{text}  {conf:.2f}",
                        (pts[0][0], pts[0][1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.polylines(frame, [pts], True, (100, 100, 180), 1)

    # ── Top banner ────────────────────────────────────────────
    if aisle_match:
        cv2.rectangle(frame, (0, 0), (w, 44), (0, 110, 0), -1)
        cv2.putText(frame,
                    f"DETECTED:  Aisle {extract_aisle_number(aisle_match)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 255, 80), 2)
    elif detected_flash > 0:
        cv2.rectangle(frame, (0, 0), (w, 44), (0, 80, 0), -1)
        cv2.putText(frame, last_label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 255, 80), 2)
    else:
        cv2.rectangle(frame, (0, 0), (w, 44), (28, 28, 28), -1)
        cv2.putText(frame, "Hold aisle sign in front of camera",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (130, 130, 130), 1)

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
print("Results saved to ocr_results_log.csv")
