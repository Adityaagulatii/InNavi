import cv2
import numpy as np
from collections import deque

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
DARK_THRESHOLD  = 100
SCORE_THRESHOLD = 0.35
LEFT_END        = 0.35
RIGHT_START     = 0.65
SMOOTH          = 8

# ─────────────────────────────────────────
# DETECTION
# ─────────────────────────────────────────
def process(frame):
    fh, fw = frame.shape[:2]
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray   = clahe.apply(gray)

    lx = int(fw * LEFT_END)
    rx = int(fw * RIGHT_START)

    l_mean = np.mean(gray[:, 0:lx])
    c_mean = np.mean(gray[:, lx:rx])
    r_mean = np.mean(gray[:, rx:fw])

    l_open = l_mean < DARK_THRESHOLD
    r_open = r_mean < DARK_THRESHOLD

    med   = np.median(gray)
    edges = cv2.Canny(gray, int(max(0,0.67*med)), int(min(255,1.33*med)))
    cy    = int(fh * 0.25)
    fy    = int(fh * 0.75)

    def h_lines(zone):
        ls = cv2.HoughLinesP(zone, 1, np.pi/180, 40,
                             minLineLength=int(fw*0.25), maxLineGap=20)
        if ls is None: return []
        return [l for l in ls if
                np.degrees(np.arctan2(abs(l[0][3]-l[0][1]),
                                      abs(l[0][2]-l[0][0]))) < 20]

    ceil_lines  = h_lines(edges[0:cy, :])
    floor_lines = h_lines(edges[fy:fh, :])
    has_h_edge  = len(ceil_lines) > 0 or len(floor_lines) > 0

    score = 0.0
    if l_open or r_open:
        score = 0.5
        if has_h_edge: score += 0.3
    elif has_h_edge:
        score = 0.1

    return score, l_open, r_open, l_mean, r_mean, \
           ceil_lines, floor_lines, cy, fy, lx, rx, fh, fw

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        exit()

    print("Intersection Detection — press Q to quit")
    history = deque(maxlen=SMOOTH)

    while True:
        ret, frame = cap.read()
        if not ret: break

        score, l_open, r_open, l_mean, r_mean, \
        ceil_l, floor_l, cy, fy, lx, rx, fh, fw = process(frame)

        history.append(1 if score >= SCORE_THRESHOLD else 0)
        confidence = sum(history) / len(history)
        detected   = confidence >= 0.5

        display = frame.copy()

        # Tint zones
        def tint(img, x1, x2, active):
            ov = img.copy()
            cv2.rectangle(ov,(x1,0),(x2,fh),(0,0,180) if active else (0,120,0),-1)
            cv2.addWeighted(ov,0.2,img,0.8,0,img)

        tint(display, 0,  lx, l_open)
        tint(display, rx, fw, r_open)

        # Zone lines
        cv2.line(display,(lx,0),(lx,fh),(0,200,255),1)
        cv2.line(display,(rx,0),(rx,fh),(0,200,255),1)
        cv2.line(display,(0,cy),(fw,cy),(255,165,0),1)
        cv2.line(display,(0,fy),(fw,fy),(255,165,0),1)

        # Floor/ceiling lines
        for l in ceil_l:
            x1,y1,x2,y2=l[0]
            cv2.line(display,(x1,y1),(x2,y2),(0,255,255),2)
        for l in floor_l:
            x1,y1,x2,y2=l[0]
            cv2.line(display,(x1,y1+fy),(x2,y2+fy),(0,255,255),2)

        # Result
        if detected:
            label = "INTERSECTION DETECTED"
            color = (0,255,0)
            bg    = (0,80,0)
        else:
            label = "CORRIDOR"
            color = (150,150,150)
            bg    = (0,0,0)

        cv2.rectangle(display,(0,0),(fw,60),bg,-1)
        cv2.putText(display, label, (15,42),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)

        # Confidence bar
        cv2.rectangle(display,(0,58),(fw,65),(40,40,40),-1)
        cv2.rectangle(display,(0,58),(int(fw*confidence),65),(0,200,255),-1)

        cv2.imshow("Intersection Detection", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()