"""
WayfinderAI — Shelf Product Detector
======================================
Uses a trained YOLOv8 grocery model to detect products on shelves.
Splits the frame into zones (Left / Right, Top / Mid / Bottom)
and shows which products are detected on each side — matching the
store navigation data format used by the app.

Display:
  Left panel  = webcam with bounding boxes coloured by shelf side
  Right panel = zone summary (Left shelf / Right shelf × 3 levels)

Requires:
  pip install ultralytics

Set MODEL_PATH below to your trained .pt weights file.
Press Q to quit.
"""

import tkinter as tk
import cv2
import numpy as np
import time
from collections import defaultdict
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────
# CONFIG — update MODEL_PATH to your weights file
# ─────────────────────────────────────────────────────────────
MODEL_PATH  = "yolov8n.pt"   # nano — fastest, ~6MB, plenty for single-class detection
CONF_THRESH = 0.40              # minimum confidence to show a detection
IOU_THRESH  = 0.45
ONLY_CLASSES = {"bottle"}   # set to None to detect everything

# Zone split fractions
LEFT_FRAC   = 0.50              # left of this = Left shelf, right = Right shelf
TOP_FRAC    = 0.33              # top third
BOT_FRAC    = 0.66              # bottom third

# Sidebar dimensions
SIDE_W      = 420
SIDE_H      = 480

# Colours per side (BGR)
COLOR_LEFT  = (255, 140,  40)   # orange
COLOR_RIGHT = ( 40, 180, 255)   # blue
COLOR_TEXT  = (220, 220, 220)

# ─────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────
print(f"Loading YOLO model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
model.conf = CONF_THRESH
model.iou  = IOU_THRESH
print("Model loaded.")

# ─────────────────────────────────────────────────────────────
# Zone helper
# ─────────────────────────────────────────────────────────────
def get_zone(cx, cy, fw, fh):
    """Return (side, level) strings for a detection centre point."""
    side  = "Left"  if cx < fw * LEFT_FRAC else "Right"
    if cy < fh * TOP_FRAC:
        level = "Top"
    elif cy < fh * BOT_FRAC:
        level = "Mid"
    else:
        level = "Bot"
    return side, level


# ─────────────────────────────────────────────────────────────
# Sidebar builder — shows detected items per zone
# ─────────────────────────────────────────────────────────────
LEVELS = ["Top", "Mid", "Bot"]
LEVEL_LABELS = {"Top": "Top shelf", "Mid": "Mid shelf", "Bot": "Bot shelf"}

def draw_sidebar(zone_items, frame_h):
    """
    zone_items: dict  (side, level) → list of (label, conf)
    Returns a BGR image of size (SIDE_W, frame_h)
    """
    canvas = np.ones((frame_h, SIDE_W, 3), dtype=np.uint8) * 22

    cv2.putText(canvas, "Shelf Detection", (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.line(canvas, (10, 34), (SIDE_W - 10, 34), (60, 60, 60), 1)

    col_w   = (SIDE_W - 30) // 2   # width for each side column
    col_xs  = {"Left": 10, "Right": 10 + col_w + 10}
    colors  = {"Left": COLOR_LEFT, "Right": COLOR_RIGHT}

    # Column headers
    for side, cx in col_xs.items():
        cv2.rectangle(canvas, (cx, 42), (cx + col_w, 62), colors[side], -1)
        cv2.putText(canvas, f"{side} side", (cx + 4, 57),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1)

    y = 72
    row_h = (frame_h - y - 10) // len(LEVELS)

    for level in LEVELS:
        # Level header bar
        cv2.rectangle(canvas, (8, y), (SIDE_W - 8, y + 18), (45, 45, 45), -1)
        cv2.putText(canvas, LEVEL_LABELS[level], (12, y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 100), 1)
        y += 22

        max_items = (row_h - 22) // 16   # how many items fit
        for side in ("Left", "Right"):
            cx    = col_xs[side]
            items = zone_items.get((side, level), [])
            iy    = y
            for label, conf in items[:max_items]:
                short = label[:18]
                cv2.putText(canvas, f"{short}", (cx + 2, iy + 11),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.34, colors[side], 1)
                iy += 16
            if not items:
                cv2.putText(canvas, "—", (cx + 2, y + 11),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.34, (70, 70, 70), 1)

        y += row_h - 22

    # Footer: total count
    total = sum(len(v) for v in zone_items.values())
    cv2.putText(canvas, f"Detections: {total}",
                (12, frame_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (130, 130, 130), 1)

    return canvas


# ─────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────
print("Opening webcam…")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open webcam.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

fps_timer   = time.time()
fps_display = 0.0
frame_count = 0

_root = tk.Tk()
SCREEN_W, SCREEN_H = _root.winfo_screenwidth(), _root.winfo_screenheight()
_root.destroy()

WIN = "WayfinderAI — Shelf Detector  (Q to quit)"
cv2.namedWindow(WIN, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("\n=== SHELF DETECTOR LIVE ===")
print("Point camera at a shelf — products detected on Left/Right sides.")
print("Press Q to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    fh, fw = frame.shape[:2]

    # ── YOLO inference ────────────────────────────────────────
    results = model(frame, verbose=False)[0]

    zone_items = defaultdict(list)   # (side, level) → [(label, conf)]

    for box in results.boxes:
        conf  = float(box.conf[0])
        cls   = int(box.cls[0])
        label = model.names[cls]

        if ONLY_CLASSES and label not in ONLY_CLASSES:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        side, level = get_zone(cx, cy, fw, fh)
        color = COLOR_LEFT if side == "Left" else COLOR_RIGHT

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label chip
        chip = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(chip, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, chip, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

        # Side tag on box corner
        cv2.putText(frame, f"{side[0]}{level[0]}",
                    (x2 - 22, y2 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

        zone_items[(side, level)].append((label, conf))

    # ── Divider lines on webcam feed ─────────────────────────
    mid_x = int(fw * LEFT_FRAC)
    mid_y1 = int(fh * TOP_FRAC)
    mid_y2 = int(fh * BOT_FRAC)

    cv2.line(frame, (mid_x, 0), (mid_x, fh), (80, 80, 80), 1)
    cv2.line(frame, (0, mid_y1), (fw, mid_y1), (50, 50, 50), 1)
    cv2.line(frame, (0, mid_y2), (fw, mid_y2), (50, 50, 50), 1)

    # Side labels on webcam
    cv2.putText(frame, "LEFT",  (8, fh - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_LEFT, 1)
    cv2.putText(frame, "RIGHT", (mid_x + 8, fh - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_RIGHT, 1)

    # ── FPS ───────────────────────────────────────────────────
    if frame_count % 10 == 0:
        fps_display = 10 / (time.time() - fps_timer + 1e-6)
        fps_timer   = time.time()

    cv2.putText(frame, f"{fps_display:.1f} fps", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

    # ── Combine webcam + sidebar ──────────────────────────────
    sidebar  = draw_sidebar(zone_items, fh)
    combined = np.hstack([frame, sidebar])

    display_w = int(SCREEN_W * 0.65)
    display_h = int(display_w * combined.shape[0] / combined.shape[1])
    cv2.imshow(WIN, cv2.resize(combined, (display_w, display_h)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Stopped.")
        break

cap.release()
cv2.destroyWindow(WIN)
