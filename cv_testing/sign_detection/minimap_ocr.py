"""
Minimap + OCR Location Tracker
================================
Route: ENTR → A152 → A100 → A1 → A2 → A100 → A34 → A101 → CHEC
       (milk, eggs, pasta demo route)

Hold A152 or A100 signs in front of the webcam.
When detected, your position on the minimap updates automatically.
Press Q to quit.
"""

import tkinter as tk
import cv2
import easyocr
import numpy as np
import re
import time

# ─────────────────────────────────────────────────────────────
# ROUTE DEFINITION
# Each node: id, display label, (x, y) on minimap canvas
# ─────────────────────────────────────────────────────────────
MINIMAP_W, MINIMAP_H = 500, 420

ROUTE_NODES = [
    {"id": "entrance_left", "label": "ENTR",  "short": "Entrance",        "pos": (60,  370)},
    {"id": "152",           "label": "A152",  "short": "Bakery",           "pos": (60,  270)},
    {"id": "100",           "label": "A100",  "short": "Dairy",            "pos": (180, 200)},
    {"id": "1",             "label": "A1",    "short": "Dairy & Bakery",   "pos": (180, 300)},
    {"id": "2",             "label": "A2",    "short": "Dry Goods",        "pos": (280, 300)},
    {"id": "34",            "label": "A34",   "short": "Yogurt",           "pos": (340, 200)},
    {"id": "101",           "label": "A101",  "short": "Meat & Poultry",   "pos": (420, 200)},
    {"id": "checkout_1",    "label": "CHEC",  "short": "Checkout",         "pos": (440, 340)},
]

# Route order (sequence of node IDs the walking dot follows)
ROUTE_ORDER = [
    "entrance_left", "152", "100", "1", "2", "1", "100", "34", "101", "checkout_1"
]

# Scannable signs → which node they update position to
SCANNABLE = {
    "152": "152",
    "a152": "152",
    "100": "100",
    "a100": "100",
}

# Nodes the user will physically scan (shown differently on minimap)
SCAN_STOPS = {"152", "100"}

# ─────────────────────────────────────────────────────────────
# OCR pattern — look for A152, A100, 152, 100
# ─────────────────────────────────────────────────────────────
SCAN_PATTERN = re.compile(r'\b(A?\s*1[05][02])\b', re.IGNORECASE)

def normalize(text):
    """Strip spaces and lowercase for lookup."""
    return re.sub(r'\s+', '', text.lower())

# ─────────────────────────────────────────────────────────────
# Build a lookup: node id → node dict
# ─────────────────────────────────────────────────────────────
NODE_LOOKUP = {n["id"]: n for n in ROUTE_NODES}

# ─────────────────────────────────────────────────────────────
# Draw the minimap onto a blank canvas
# ─────────────────────────────────────────────────────────────
def draw_minimap(current_node_id):
    canvas = np.ones((MINIMAP_H, MINIMAP_W, 3), dtype=np.uint8) * 30  # dark bg

    # Title
    cv2.putText(canvas, "Milk  Eggs  Pasta  Route", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # Draw route edges
    for i in range(len(ROUTE_ORDER) - 1):
        a = NODE_LOOKUP.get(ROUTE_ORDER[i])
        b = NODE_LOOKUP.get(ROUTE_ORDER[i + 1])
        if a and b:
            cv2.line(canvas, a["pos"], b["pos"], (70, 130, 220), 2)

    # Draw nodes
    for node in ROUTE_NODES:
        x, y = node["pos"]
        nid = node["id"]
        is_current = (nid == current_node_id)
        is_scan    = (nid in SCAN_STOPS)

        if is_current:
            # Pulsing red circle for current position
            cv2.circle(canvas, (x, y), 22, (0, 60, 200), -1)
            cv2.circle(canvas, (x, y), 22, (80, 160, 255), 2)
            cv2.circle(canvas, (x, y), 10, (0, 0, 220), -1)
            cv2.circle(canvas, (x, y), 5,  (255, 80, 80), -1)
        elif is_scan:
            # Scannable stops — bright blue outlined
            cv2.circle(canvas, (x, y), 18, (200, 140, 0), -1)
            cv2.circle(canvas, (x, y), 18, (255, 200, 0), 2)
        else:
            cv2.circle(canvas, (x, y), 14, (60, 90, 160), -1)
            cv2.circle(canvas, (x, y), 14, (100, 130, 200), 1)

        # Label inside node
        label = node["label"]
        font_scale = 0.38 if len(label) > 3 else 0.42
        text_size  = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
        tx = x - text_size[0] // 2
        ty = y + text_size[1] // 2
        cv2.putText(canvas, label, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255) if is_current else (220, 220, 220), 1)

        # Short name below node
        name_size = cv2.getTextSize(node["short"], cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)[0]
        cv2.putText(canvas, node["short"],
                    (x - name_size[0] // 2, y + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                    (255, 220, 100) if is_scan else (150, 150, 150), 1)

    # Legend
    cv2.circle(canvas, (12, MINIMAP_H - 55), 7, (0, 0, 220), -1)
    cv2.circle(canvas, (12, MINIMAP_H - 55), 3, (255, 80, 80), -1)
    cv2.putText(canvas, "You are here", (24, MINIMAP_H - 51),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

    cv2.circle(canvas, (12, MINIMAP_H - 32), 7, (200, 140, 0), -1)
    cv2.putText(canvas, "Scan sign to update", (24, MINIMAP_H - 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

    cv2.line(canvas, (10, MINIMAP_H - 12), (28, MINIMAP_H - 12), (70, 130, 220), 2)
    cv2.putText(canvas, "Route", (34, MINIMAP_H - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

    return canvas

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
print("Loading EasyOCR...")
reader = easyocr.Reader(['en'], gpu=False)
print("Done. Opening webcam...")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_count   = 0
OCR_INTERVAL  = 8
last_results  = []
last_ocr_time = 0
current_node  = "entrance_left"   # start at entrance
last_detected = ""
detected_flash = 0                # countdown to clear the detection banner

_r = tk.Tk(); SCREEN_W, SCREEN_H = _r.winfo_screenwidth(), _r.winfo_screenheight(); _r.destroy()
WIN = "WayfinderAI — OCR + Minimap  (Q to quit)"
cv2.namedWindow(WIN, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("\nHold A152 or A100 in front of the webcam.")
print("Press Q to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # ── Run OCR every N frames ──────────────────────────────────
    if frame_count % OCR_INTERVAL == 0:
        t0 = time.time()
        raw = reader.readtext(frame)
        last_ocr_time = time.time() - t0

        last_results = [(bbox, text, conf) for bbox, text, conf in raw if conf >= 0.45]

        for _, text, conf in last_results:
            key = normalize(text)
            if key in SCANNABLE:
                new_node = SCANNABLE[key]
                if new_node != current_node:
                    current_node  = new_node
                    last_detected = f"Scanned: {text.upper()}  → {NODE_LOOKUP[new_node]['short']}"
                    detected_flash = 40   # show banner for 40 frames
                    print(f"  ✓ Detected '{text}' (conf {conf:.2f}) → moved to {NODE_LOOKUP[new_node]['short']}")

    if detected_flash > 0:
        detected_flash -= 1

    # ── Draw bounding boxes on webcam feed ─────────────────────
    for bbox, text, conf in last_results:
        pts = np.array(bbox, dtype=np.int32)
        is_aisle = bool(SCAN_PATTERN.search(text))
        color = (0, 255, 0) if is_aisle else (180, 100, 0)
        cv2.polylines(frame, [pts], True, color, 2)
        cv2.putText(frame, f"{text} ({conf:.2f})", (pts[0][0], pts[0][1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    # Detection flash banner
    if detected_flash > 0:
        cv2.rectangle(frame, (0, 0), (640, 36), (0, 120, 0), -1)
        cv2.putText(frame, last_detected, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 255, 80), 2)

    # HUD
    cv2.putText(frame, f"OCR every {OCR_INTERVAL}f  |  last: {last_ocr_time:.2f}s",
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)

    # ── Build minimap ───────────────────────────────────────────
    minimap = draw_minimap(current_node)

    # Resize minimap to match webcam height
    minimap_resized = cv2.resize(minimap, (MINIMAP_W, frame.shape[0]))

    # Combine webcam + minimap side by side
    combined = np.hstack([frame, minimap_resized])

    dw = int(SCREEN_W * 0.65)
    cv2.imshow(WIN, cv2.resize(combined, (dw, int(dw * combined.shape[0] / combined.shape[1]))))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Stopped.")
        break

cap.release()
cv2.destroyAllWindows()
