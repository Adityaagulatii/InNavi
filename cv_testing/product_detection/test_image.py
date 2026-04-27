"""
WayfinderAI — Shelf Detector: Static Image Test
================================================
Runs the YOLO grocery model on a single image.
Divides shelf into 3 columns × 3 rows = 9 zones.

Columns:  Left | Center | Right  (each 33% of width)
Rows:     Top  | Mid    | Bot    (each 33% of height)

Usage:  python test_image.py
"""

import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

IMAGE_PATH  = "grocery_preview.png"
MODEL_PATH  = "yolov8n.pt"
OUTPUT_PATH = "result.jpg"
CONF_THRESH = 0.35

COLOR_LEFT   = (255, 140,  40)   # orange
COLOR_CENTER = (180,  40, 200)   # purple
COLOR_RIGHT  = ( 40, 180, 255)   # blue
COLORS = {"Left": COLOR_LEFT, "Center": COLOR_CENTER, "Right": COLOR_RIGHT}

SIDEBAR_W = 500

# ─────────────────────────────────────────────────────────────
def get_zone(cx, cy, fw, fh):
    if cx < fw / 3:
        col = "Left"
    elif cx < fw * 2 / 3:
        col = "Center"
    else:
        col = "Right"

    if cy < fh / 3:
        row = "Top"
    elif cy < fh * 2 / 3:
        row = "Mid"
    else:
        row = "Bot"

    return col, row

# ─────────────────────────────────────────────────────────────
def draw_sidebar(zone_items, img_h):
    canvas = np.ones((img_h, SIDEBAR_W, 3), dtype=np.uint8) * 22
    cols   = ["Left", "Center", "Right"]
    rows   = ["Top", "Mid", "Bot"]
    cw     = (SIDEBAR_W - 20) // 3   # column width

    cv2.putText(canvas, "Shelf Zones", (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.line(canvas, (8, 32), (SIDEBAR_W - 8, 32), (60, 60, 60), 1)

    # Column headers
    for i, col in enumerate(cols):
        x = 10 + i * cw
        cv2.rectangle(canvas, (x, 36), (x + cw - 4, 56), COLORS[col], -1)
        cv2.putText(canvas, col, (x + 4, 51),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 0, 0), 1)

    y      = 62
    row_h  = max(60, (img_h - y - 10) // 3)

    for row in rows:
        # Row label
        cv2.rectangle(canvas, (8, y), (SIDEBAR_W - 8, y + 18), (45, 45, 45), -1)
        cv2.putText(canvas, f"{row} shelf", (12, y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 100), 1)
        y += 20

        for i, col in enumerate(cols):
            x     = 10 + i * cw
            items = zone_items.get((col, row), [])
            iy    = y
            max_n = max(1, (row_h - 20) // 16)
            for label, conf in items[:max_n]:
                short = label.replace("-", " ")[:16]
                cv2.putText(canvas, short, (x + 2, iy + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.30, COLORS[col], 1)
                iy += 16
            if not items:
                cv2.putText(canvas, "—", (x + 2, y + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (60, 60, 60), 1)

        y += row_h - 20

    total = sum(len(v) for v in zone_items.values())
    cv2.putText(canvas, f"Detections: {total}", (12, img_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (130, 130, 130), 1)
    return canvas

# ─────────────────────────────────────────────────────────────
print(f"Loading model…")
model      = YOLO(MODEL_PATH)
model.conf = CONF_THRESH

print(f"Reading image…")
frame = cv2.imread(IMAGE_PATH)
if frame is None:
    print("ERROR: Could not load image.")
    exit()

# Fit to screen (cap width at 900, height at 800)
fh, fw = frame.shape[:2]
scale  = min(900 / fw, 800 / fh, 1.0)
if scale < 1.0:
    frame = cv2.resize(frame, (int(fw * scale), int(fh * scale)))
    fh, fw = frame.shape[:2]

print(f"Image: {fw}w x {fh}h   columns at x={fw//3}, {fw*2//3}")

results    = model(frame, verbose=False)[0]
zone_items = defaultdict(list)

for box in results.boxes:
    conf  = float(box.conf[0])
    cls   = int(box.cls[0])
    label = model.names[cls]
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    col, row = get_zone(cx, cy, fw, fh)
    color    = COLORS[col]

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    chip = f"{label.replace('-',' ')} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(chip, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
    cv2.rectangle(frame, (x1, y1 - th - 5), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, chip, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1)
    zone_items[(col, row)].append((label, conf))

# Column + row divider lines
for xi in [fw // 3, fw * 2 // 3]:
    cv2.line(frame, (xi, 0), (xi, fh), (0, 0, 200), 1)
for yi in [fh // 3, fh * 2 // 3]:
    cv2.line(frame, (0, yi), (fw, yi), (60, 60, 60), 1)

# Column labels at bottom
for i, (lbl, col) in enumerate(zip(["LEFT", "CENTER", "RIGHT"],
                                   ["Left", "Center", "Right"])):
    x = i * (fw // 3) + 4
    cv2.putText(frame, lbl, (x, fh - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS[col], 1)

sidebar  = draw_sidebar(zone_items, fh)
combined = np.hstack([frame, sidebar])

# Fit combined to screen width
MAX_W = 1400
ch, cw = combined.shape[:2]
if cw > MAX_W:
    combined = cv2.resize(combined, (MAX_W, int(ch * MAX_W / cw)))

cv2.imwrite(OUTPUT_PATH, combined)
print(f"Saved → {OUTPUT_PATH}")

print("\n── Zone Summary ──")
for (col, row), items in sorted(zone_items.items()):
    print(f"  {col:6} {row}: {', '.join(l for l,_ in items)}")

cv2.namedWindow("WayfinderAI — Shelf Test  (any key)", cv2.WINDOW_NORMAL)
cv2.imshow("WayfinderAI — Shelf Test  (any key)", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
