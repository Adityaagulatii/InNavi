import streamlit as st
import networkx as nx
import json
import math
import re
import time
import numpy as np
import cv2
from PIL import Image, ImageDraw
from streamlit_drawable_canvas import st_canvas
import easyocr
import urllib.request
import threading

st.set_page_config(page_title="Navi Grid", layout="wide")

# ── Session State ────────────────────────────────────────────────────────────
for key, default in {
    "nodes": {}, "edges": [], "edge_start": None,
    "path": [], "pending_click": None, "directions": [],
    "mode": "Place Node", "rotation": 0,
    "current_step": 0,   # which checkpoint we're at
    "ocr_status": None,  # last OCR result message
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Direction Logic ──────────────────────────────────────────────────────────
def pixel_distance(a, b):
    return int(((a["x"]-b["x"])**2 + (a["y"]-b["y"])**2)**0.5)

def get_reference_node(nodes):
    return list(nodes.keys())[0]

def get_turn_instruction(facing_vec, curr_node, next_node):
    dx = next_node["x"] - curr_node["x"]
    dy = next_node["y"] - curr_node["y"]
    cross = facing_vec[0]*dy - facing_vec[1]*dx
    dot   = facing_vec[0]*dx + facing_vec[1]*dy
    angle = math.degrees(math.atan2(cross, dot))
    if   -45 <= angle <= 45:  return "↑ Go straight"
    elif angle > 45:          return "← Turn left"
    elif angle < -45:         return "→ Turn right"
    else:                     return "↩ Turn around"

def get_landmark_hint(direction, next_label):
    mapping = {
        "↑ Go straight": f"Go straight — look for **{next_label}** ahead of you",
        "← Turn left":   f"Turn left — look for **{next_label}** on your left",
        "→ Turn right":  f"Turn right — look for **{next_label}** on your right",
        "↩ Turn around": f"Turn around — look for **{next_label}** behind you",
    }
    return mapping.get(direction, f"Head towards **{next_label}**")

def compute_directions(path, nodes):
    results = []
    ref_node  = nodes[get_reference_node(nodes)]
    first_node = nodes[path[0]]
    dx = first_node["x"] - ref_node["x"]
    dy = first_node["y"] - ref_node["y"]
    mag = max((dx**2+dy**2)**0.5, 1)
    facing = (dx/mag, dy/mag)
    for i in range(len(path)-1):
        a, b   = path[i], path[i+1]
        na, nb = nodes[a], nodes[b]
        direction = get_turn_instruction(facing, na, nb)
        hint      = get_landmark_hint(direction, b)
        dist      = pixel_distance(na, nb)
        results.append((i+1, a, b, direction, hint, dist))
        dx = nb["x"] - na["x"]
        dy = nb["y"] - na["y"]
        mag = max((dx**2+dy**2)**0.5, 1)
        facing = (dx/mag, dy/mag)
    return results

def nearest_node(x, y, scale, threshold=15):
    best, best_d = None, threshold
    for lbl, pos in st.session_state.nodes.items():
        px, py = int(pos["x"]*scale), int(pos["y"]*scale)
        d = ((px-x)**2+(py-y)**2)**0.5
        if d < best_d:
            best, best_d = lbl, d
    return best

# ── OCR Logic ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_ocr():
    return easyocr.Reader(["en"], gpu=False)

def normalize(text):
    return re.sub(r"[\s\-\.]", "", text).upper()

def get_substrings(text, min_len=3):
    """Generate all substrings of length >= min_len from a normalized text."""
    text = normalize(text)
    subs = set()
    for i in range(len(text)):
        for j in range(i + min_len, len(text) + 1):
            subs.add(text[i:j])
    return subs

def match_ocr_to_node(ocr_texts, node_labels):
    """
    Aggressively match OCR results against node labels using substrings.
    e.g. '32-G415' detected as '415', '15', 'G41' all match node 'G415'
    """
    # Build substring sets for each node label
    node_substrings = {}
    for lbl in node_labels:
        node_substrings[lbl] = get_substrings(lbl, min_len=3)

    for text in ocr_texts:
        norm_text = normalize(text)
        ocr_subs  = get_substrings(norm_text, min_len=3)

        for lbl, lbl_subs in node_substrings.items():
            # Check if any OCR substring matches any label substring
            if ocr_subs & lbl_subs:  # intersection
                return lbl

    return None

def grab_frame(ip_url):
    """Grab a single frame from IP Webcam."""
    try:
        url = f"http://{ip_url}/shot.jpg"
        img_resp = urllib.request.urlopen(url, timeout=3)
        img_array = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        frame = cv2.imdecode(img_array, -1)
        return frame
    except Exception as e:
        return None

def run_ocr_on_frame(frame, target_nodes):
    """Run EasyOCR on a frame and match against target nodes."""
    reader = load_ocr()
    small  = cv2.resize(frame, (640, 480))
    rgb    = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    results = reader.readtext(rgb)
    texts   = [r[1] for r in results]
    matched = match_ocr_to_node(texts, target_nodes)
    # Draw boxes on frame for visual feedback
    for (bbox, text, conf) in results:
        pts = np.array(bbox, dtype=np.int32)
        color = (0,255,0) if matched and normalize(text) in normalize(matched) else (255,165,0)
        cv2.polylines(frame, [pts], True, color, 2)
        cv2.putText(frame, text, tuple(pts[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return matched, texts, frame

# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🗺️ Navi Grid")

    # ── 1. Upload Image ──────────────────────────────────────────────────────
    st.markdown("### 1️⃣ Upload Floor Plan")
    uploaded = st.file_uploader("Floor plan image", type=["png","jpg","jpeg"],
                                label_visibility="collapsed")
    st.divider()

    # ── 2. Two Pathways ──────────────────────────────────────────────────────
    st.markdown("### 2️⃣ Load Map Data")
    pathway = st.radio("pathway", ["📂 Load existing JSON", "✏️ Annotate manually"],
                       label_visibility="collapsed")

    # ── PATHWAY A: Load JSON ─────────────────────────────────────────────────
    if pathway == "📂 Load existing JSON":
        imported = st.file_uploader("Upload floor_graph.json", type="json",
                                    label_visibility="collapsed")
        if imported:
            data = json.load(imported)
            new_nodes = data.get("nodes", {})
            new_edges = data.get("edges", [])
            # Only rerun if data is actually different — prevents flicker loop
            if new_nodes != st.session_state.nodes or new_edges != st.session_state.edges:
                st.session_state.nodes = new_nodes
                st.session_state.edges = new_edges
                st.session_state.path = []
                st.session_state.directions = []
                st.session_state.mode = "Navigate"
                st.rerun()
            else:
                st.success(f"✅ Loaded {len(st.session_state.nodes)} nodes, {len(st.session_state.edges)} edges!")

    # ── PATHWAY B: Annotate ──────────────────────────────────────────────────
    else:
        st.divider()
        st.markdown("### 3️⃣ Annotate")

        # Rotation (only once, with unique keys)
        st.markdown("**🔄 Rotate**")
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            if st.button("↺ 90°", key="rot_left", use_container_width=True):
                st.session_state.rotation = (st.session_state.rotation - 90) % 360
                st.rerun()
        with rc2:
            st.caption(f"{st.session_state.rotation}°")
        with rc3:
            if st.button("↻ 90°", key="rot_right", use_container_width=True):
                st.session_state.rotation = (st.session_state.rotation + 90) % 360
                st.rerun()

        # Mode
        st.markdown("**✏️ Mode**")
        st.session_state.mode = st.radio("annotate_mode", ["Place Node", "Draw Edge"],
                                         label_visibility="collapsed")

        if st.session_state.mode == "Place Node" and st.session_state.pending_click:
            x, y = st.session_state.pending_click
            st.success(f"Clicked at ({x}, {y})")
            label = st.text_input("Node name", placeholder="e.g. Room 101")
            if st.button("✅ Add Node", key="add_node_btn", use_container_width=True):
                if not label:
                    st.warning("Enter a name first.")
                elif label in st.session_state.nodes:
                    st.warning("Name already used.")
                else:
                    st.session_state.nodes[label] = {"x": x, "y": y}
                    st.session_state.pending_click = None
                    st.rerun()
            if st.button("❌ Cancel", key="cancel_node_btn", use_container_width=True):
                st.session_state.pending_click = None
                st.rerun()

        if st.session_state.mode == "Draw Edge":
            if st.session_state.edge_start:
                st.info(f"Start: **{st.session_state.edge_start}** — click another node")
                if st.button("Cancel edge", key="cancel_edge_btn"):
                    st.session_state.edge_start = None
                    st.rerun()
            else:
                st.info("Click a node to start an edge")

        # Node list
        if st.session_state.nodes:
            st.divider()
            st.markdown("**Nodes**")
            for lbl in list(st.session_state.nodes.keys()):
                c1, c2 = st.columns([3,1])
                c1.markdown(f"• {lbl}")
                if c2.button("✕", key=f"del_{lbl}"):
                    del st.session_state.nodes[lbl]
                    st.session_state.edges = [e for e in st.session_state.edges if lbl not in e]
                    st.session_state.path = []
                    st.session_state.directions = []
                    st.rerun()

        # Save
        st.divider()
        st.markdown("**💾 Save when done**")
        n_nodes = len(st.session_state.nodes)
        n_edges = len(st.session_state.edges)
        connected = set()
        for a, b in st.session_state.edges:
            connected.add(a); connected.add(b)
        orphans = [n for n in st.session_state.nodes if n not in connected]
        if n_nodes == 0:
            st.warning("⚠️ No nodes placed yet.")
        elif n_edges == 0:
            st.warning("⚠️ No edges drawn.")
        elif orphans:
            st.warning(f"⚠️ Unconnected nodes: {', '.join(orphans)}")
        else:
            st.success(f"✅ {n_nodes} nodes · {n_edges} edges — looks good!")

        st.download_button("💾 Download floor_graph.json", json.dumps(
            {"nodes": st.session_state.nodes, "edges": st.session_state.edges}, indent=2),
            file_name="floor_graph.json", mime="application/json", use_container_width=True)

        if st.button("➡️ Done — Go to Navigate", key="go_navigate",
                     use_container_width=True, type="primary"):
            st.session_state.mode = "Navigate"
            st.rerun()

    # ── Navigate (shown for both pathways) ───────────────────────────────────
    if st.session_state.mode == "Navigate" and st.session_state.nodes:
        st.divider()
        st.markdown("### 🧭 Navigate")
        node_names = list(st.session_state.nodes.keys())
        ref_label = get_reference_node(st.session_state.nodes)
        st.info(f"📍 Reference node: **{ref_label}**")

        # IP Webcam URL input
        st.markdown("**📱 IP Webcam**")
        ip_url = st.text_input("Phone camera IP", 
                               value=st.session_state.get("ip_url", "10.235.152.48:8080"),
                               placeholder="e.g. 10.235.152.48:8080",
                               key="ip_url_input")
        st.session_state["ip_url"] = ip_url

        # Test connection
        if st.button("Test Camera 🔗", key="test_cam"):
            frame = grab_frame(ip_url)
            if frame is not None:
                st.success("✅ Camera connected!")
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st.image(rgb, caption="Live preview", width=200)
            else:
                st.error("❌ Can't reach camera — check IP and WiFi")

        if len(node_names) >= 2:
            nav_nodes = [n for n in node_names if n != ref_label]
            st.markdown(f"🚪 **Start:** {ref_label} *(fixed)*")
            entry = ref_label
            dst   = st.selectbox("🏁 Destination", nav_nodes, key="nav_dst")

            if st.button("Find Path 🔍", key="find_path", use_container_width=True, type="primary"):
                G = nx.Graph()
                G.add_nodes_from(st.session_state.nodes.keys())
                for a, b in st.session_state.edges:
                    G.add_edge(a, b, weight=pixel_distance(
                        st.session_state.nodes[a], st.session_state.nodes[b]))
                try:
                    st.session_state.path = nx.astar_path(G, entry, dst, weight="weight")
                    st.session_state.directions = compute_directions(
                        st.session_state.path, st.session_state.nodes)
                    st.session_state.current_step = 0
                    st.session_state.ocr_status = None
                except nx.NetworkXNoPath:
                    st.error("No path found — check edges.")
                except nx.NodeNotFound as e:
                    st.error(f"Node not found: {e}")
        else:
            st.warning("Add at least 2 nodes first.")

    st.divider()
    if st.button("🗑️ Clear Everything", key="clear_all", use_container_width=True):
        for k, v in {"nodes":{}, "edges":[], "path":[], "edge_start":None,
                     "pending_click":None, "directions":[], "mode":"Place Node", "rotation":0}.items():
            st.session_state[k] = v
        st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# MAIN CANVAS
# ════════════════════════════════════════════════════════════════════════════
if not uploaded:
    st.markdown("## 👈 Upload a floor plan to get started")
    st.stop()

img = Image.open(uploaded).convert("RGB")
if st.session_state.rotation != 0:
    img = img.rotate(-st.session_state.rotation, expand=True)
W, H = img.size
scale = min(1.0, 900/W)
dw, dh = int(W*scale), int(H*scale)
bg = img.resize((dw, dh), Image.LANCZOS).copy()
draw = ImageDraw.Draw(bg)

PATH_PAIRS = (set(zip(st.session_state.path, st.session_state.path[1:])) |
              set(zip(st.session_state.path[1:], st.session_state.path))) \
              if st.session_state.path else set()

# Draw edges
for edge in st.session_state.edges:
    a, b = edge[0], edge[1]
    if a in st.session_state.nodes and b in st.session_state.nodes:
        ax = int(st.session_state.nodes[a]["x"] * scale)
        ay = int(st.session_state.nodes[a]["y"] * scale)
        bx = int(st.session_state.nodes[b]["x"] * scale)
        by = int(st.session_state.nodes[b]["y"] * scale)
        on_path = (a,b) in PATH_PAIRS or (b,a) in PATH_PAIRS
        draw.line([(ax,ay),(bx,by)],
                  fill="#00FF88" if on_path else "#888888",
                  width=4 if on_path else 2)

# Draw nodes
for label, pos in st.session_state.nodes.items():
    px = int(pos["x"] * scale)
    py = int(pos["y"] * scale)
    is_ref   = label == get_reference_node(st.session_state.nodes)
    is_entry = st.session_state.path and label == st.session_state.path[0]
    is_dest  = st.session_state.path and label == st.session_state.path[-1]
    is_path  = label in st.session_state.path
    color = ("#FF6B6B" if is_entry else
             "#FFD700" if is_dest else
             "#00FF88" if is_path else
             "#FF8C00" if is_ref else
             "#4A90D9")
    draw.ellipse([(px-10,py-10),(px+10,py+10)], fill=color, outline="white", width=2)
    bbox = draw.textbbox((0,0), label)
    tw = bbox[2]-bbox[0]
    draw.rectangle([(px-tw//2-2, py+11),(px+tw//2+2, py+23)], fill="black")
    draw.text((px-tw//2, py+12), label, fill="white")
    if is_path:
        draw.text((px+12, py-8), str(st.session_state.path.index(label)+1), fill="#00FF88")

canvas_result = st_canvas(
    background_image=bg, height=dh, width=dw,
    drawing_mode="point", point_display_radius=1,
    stroke_color="#FF000000",
    key=f"canvas_{st.session_state.mode}",
    update_streamlit=True,
)

if canvas_result.json_data:
    objects = canvas_result.json_data.get("objects", [])
    if objects:
        obj = objects[-1]
        cx, cy = int(obj["left"]), int(obj["top"])
        if st.session_state.mode == "Place Node":
            ox, oy = int(cx/scale), int(cy/scale)
            if st.session_state.pending_click != (ox, oy):
                st.session_state.pending_click = (ox, oy)
                st.rerun()
        elif st.session_state.mode == "Draw Edge":
            hit = nearest_node(cx, cy, scale)
            if hit:
                if st.session_state.edge_start is None:
                    st.session_state.edge_start = hit
                    st.rerun()
                elif hit != st.session_state.edge_start:
                    edge = sorted([st.session_state.edge_start, hit])
                    if edge not in [sorted(e) for e in st.session_state.edges]:
                        st.session_state.edges.append(edge)
                    st.session_state.edge_start = None
                    st.rerun()

# ── Step-by-step Directions + OCR Checkpoints ───────────────────────────────
if st.session_state.directions:
    st.divider()
    st.markdown("### 🧭 Navigation")

    total_steps = len(st.session_state.directions)
    current_step = st.session_state.current_step

    # ── Progress bar ──────────────────────────────────────────────────────────
    st.progress(current_step / total_steps, text=f"Step {current_step} of {total_steps}")

    # ── Destination reached ───────────────────────────────────────────────────
    if current_step >= total_steps:
        st.success(f"🎉 You've arrived at **{st.session_state.path[-1]}**!")
        if st.button("🔄 Start Over", key="restart"):
            st.session_state.current_step = 0
            st.session_state.path = []
            st.session_state.directions = []
            st.session_state.ocr_status = None
            st.rerun()
    else:
        # ── Current instruction ───────────────────────────────────────────────
        step_num, frm, to, direction, hint, dist = st.session_state.directions[current_step]
        st.markdown("#### Current Step")
        c1, c2 = st.columns([1, 5])
        c1.markdown(f"## {direction}")
        c2.markdown(f"**Step {step_num}:** {hint}")
        c2.caption(f"From {frm} · {dist}px away")

        # ── Remaining steps (collapsed) ───────────────────────────────────────
        if current_step + 1 < total_steps:
            with st.expander("See remaining steps"):
                for s, f, t, d, h, dist2 in st.session_state.directions[current_step+1:]:
                    st.markdown(f"**Step {s}:** {h}")
                    st.caption(f"From {f} · {dist2}px away")
                    st.divider()

        st.divider()

        # ── Live IP Webcam OCR ────────────────────────────────────────────────
        st.divider()
        ip_url = st.session_state.get("ip_url", "")
        remaining_nodes = st.session_state.path[current_step+1:]

        if not ip_url:
            st.warning("⚠️ Set your IP Webcam address in the sidebar first")
        else:
            st.markdown(f"#### 📷 Looking for **{to}** — point camera at room sign")

            # Live feed placeholder
            frame_placeholder = st.empty()
            status_placeholder = st.empty()
            scan_btn_col, stop_col = st.columns(2)

            scanning = st.session_state.get("scanning", False)

            with scan_btn_col:
                if st.button("▶️ Start Scanning", key="start_scan", 
                             disabled=scanning, use_container_width=True):
                    st.session_state.scanning = True
                    st.rerun()
            with stop_col:
                if st.button("⏹ Stop", key="stop_scan",
                             disabled=not scanning, use_container_width=True):
                    st.session_state.scanning = False
                    st.rerun()

            if scanning:
                matched = None
                attempt = 0
                while True:
                    attempt += 1
                    frame = grab_frame(ip_url)
                    if frame is None:
                        status_placeholder.error("❌ Lost camera connection — check IP Webcam is running")
                        st.session_state.scanning = False
                        break

                    matched, texts, annotated = run_ocr_on_frame(frame.copy(), remaining_nodes)
                    rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(rgb, caption=f"Scan {attempt} — detected: {texts}",
                                            width=600)

                    if matched:
                        st.session_state.scanning = False
                        if matched == to:
                            status_placeholder.success(f"✅ Found **{matched}** — moving to next step!")
                            st.session_state.current_step += 1
                        else:
                            new_idx = max(0, st.session_state.path.index(matched) - 1)
                            status_placeholder.warning(f"⚠️ Detected **{matched}** — updating position")
                            st.session_state.current_step = new_idx
                        time.sleep(1)
                        st.rerun()
                        break
                    else:
                        status_placeholder.info(f"🔄 Scan {attempt} — still looking for **{to}**...")
                        time.sleep(0.5)

        # Manual override — in case OCR fails
        st.caption("Or confirm manually:")
        if st.button(f"✅ I'm at {to} — next step", key="manual_confirm"):
            st.session_state.current_step += 1
            st.rerun()