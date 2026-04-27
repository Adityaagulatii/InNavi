"""
WayfinderAI — ArUco Marker Generator
======================================
Generates DICT_4X4_50 marker images and saves them as PNGs.
Print them, tape to walls / shelves, then run aruco_detector.py.

Requires:  pip install opencv-contrib-python

Output:  marker_0.png … marker_9.png  (200×200 px each)
         marker_sheet.png             (all 10 on one A4-ish sheet)
"""

import cv2
import numpy as np
import os

OUTPUT_DIR  = os.path.dirname(__file__)   # save alongside this script
MARKER_SIZE = 400                         # px per marker image
NUM_MARKERS = 8                           # IDs 0-7 (route stops)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

AISLE_LABELS = {
    0: "A152  Bakery",
    1: "A100  Dairy",
    2: "A1    Dairy & Bakery",
    3: "A2    Dry Goods",
    4: "A34   Yogurt",
    5: "A101  Meat & Poultry",
    6: "ENTR  Entrance",
    7: "CHEC  Checkout",
    8: "Aisle 9  Cleaning",
    9: "Aisle 10 Personal Care",
}

marker_imgs = []

for marker_id in range(NUM_MARKERS):
    # Generate raw marker (200×200 white border included)
    img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, MARKER_SIZE)

    # Add white label strip below the marker
    label     = AISLE_LABELS.get(marker_id, f"Marker {marker_id}")
    label_h   = 60
    canvas    = np.ones((MARKER_SIZE + label_h, MARKER_SIZE), dtype=np.uint8) * 255
    canvas[:MARKER_SIZE, :] = img

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    tx = (MARKER_SIZE - tw) // 2
    ty = MARKER_SIZE + label_h - 16
    cv2.putText(canvas, label, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    path = os.path.join(OUTPUT_DIR, f"marker_{marker_id}.png")
    cv2.imwrite(path, canvas)
    marker_imgs.append(canvas)
    print(f"  Saved  marker_{marker_id}.png  ({label})")

# ── Build a single printable sheet (2 columns × 5 rows) ──────
cols, rows   = 2, 4
cell_w       = MARKER_SIZE
cell_h       = MARKER_SIZE + 30
padding      = 20
sheet_w      = cols * cell_w + (cols + 1) * padding
sheet_h      = rows * cell_h + (rows + 1) * padding
sheet        = np.ones((sheet_h, sheet_w), dtype=np.uint8) * 255

for i, img in enumerate(marker_imgs):
    row = i // cols
    col = i %  cols
    y   = padding + row * (cell_h + padding)
    x   = padding + col * (cell_w + padding)
    sheet[y:y + cell_h, x:x + cell_w] = img

sheet_path = os.path.join(OUTPUT_DIR, "marker_sheet.png")
cv2.imwrite(sheet_path, sheet)
print(f"\nAll-in-one sheet saved → marker_sheet.png")
print("Print that file, cut out markers, tape to shelves, then run aruco_detector.py")
