import streamlit as st
import numpy as np
import networkx as nx
import json
from PIL import Image, ImageDraw
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Navi Grid", layout="wide")

# ── Session State ────────────────────────────────────────────────────────────
for key, default in {
    "nodes": {},       # {label: {"x": int, "y": int}}
    "edges": [],       # [[a, b], ...]
    "edge_start": None,
    "path": [],
    "pending_click": None,  # (x, y) waiting for a label
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🗺️ Navi Grid")
    uploaded = st.file_uploader("Upload Floor Plan", type=["png", "jpg", "jpeg"])

    st.divider()
    mode = st.radio("Mode", ["Place Node", "Draw Edge", "Navigate"])

    # --- Place Node: show label input if a click is pending ---
    if mode == "Place Node" and st.session_state.pending_click:
        x, y = st.session_state.pending_click
        st.success(f"Clicked at ({x}, {y})")
        label = st.text_input("Node name", placeholder="e.g. Room 101")
        if st.button("✅ Add Node", use_container_width=True):
            if not label:
                st.warning("Enter a name first.")
            elif label in st.session_state.nodes:
                st.warning("Name already used.")
            else:
                st.session_state.nodes[label] = {"x": x, "y": y}
                st.session_state.pending_click = None
                st.rerun()
        if st.button("❌ Cancel", use_container_width=True):
            st.session_state.pending_click = None
            st.rerun()

    # --- Draw Edge ---
    if mode == "Draw Edge":
        if st.session_state.edge_start:
            st.info(f"Start: **{st.session_state.edge_start}** — click another node")
            if st.button("Cancel edge"):
                st.session_state.edge_start = None
                st.rerun()
        else:
            st.info("Click a node dot to start an edge")

    # --- Navigate ---
    if mode == "Navigate":
        node_names = list(st.session_state.nodes.keys())
        if len(node_names) >= 2:
            src = st.selectbox("From", node_names)
            dst = st.selectbox("To", node_names, index=1)
            if st.button("Find Path 🔍", use_container_width=True):
                G = nx.Graph()
                G.add_nodes_from(st.session_state.nodes.keys())
                for a, b in st.session_state.edges:
                    na, nb = st.session_state.nodes[a], st.session_state.nodes[b]
                    dist = ((na["x"]-nb["x"])**2 + (na["y"]-nb["y"])**2)**0.5
                    G.add_edge(a, b, weight=dist)
                try:
                    st.session_state.path = nx.astar_path(G, src, dst, weight="weight")
                    st.success(" → ".join(st.session_state.path))
                except nx.NetworkXNoPath:
                    st.error("No path found — check edges.")
        else:
            st.warning("Add at least 2 nodes first.")

    st.divider()

    # Export / Clear
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("💾 Export", json.dumps(
            {"nodes": st.session_state.nodes, "edges": st.session_state.edges}, indent=2),
            file_name="floor_graph.json", mime="application/json", use_container_width=True)
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.nodes = {}
            st.session_state.edges = []
            st.session_state.path = []
            st.session_state.edge_start = None
            st.session_state.pending_click = None
            st.rerun()

    # Node list with delete
    if st.session_state.nodes:
        st.divider()
        st.markdown("**Nodes**")
        for lbl in list(st.session_state.nodes.keys()):
            c1, c2 = st.columns([3, 1])
            c1.markdown(f"• {lbl}")
            if c2.button("✕", key=f"del_{lbl}"):
                del st.session_state.nodes[lbl]
                st.session_state.edges = [e for e in st.session_state.edges if lbl not in e]
                st.session_state.path = []
                st.rerun()

# ── Main Canvas ──────────────────────────────────────────────────────────────
if not uploaded:
    st.info("👈 Upload a floor plan to get started.")
    st.stop()

img = Image.open(uploaded).convert("RGB")
W, H = img.size
MAX_W = 900
scale = min(1.0, MAX_W / W)
dw, dh = int(W * scale), int(H * scale)
display_img = img.resize((dw, dh), Image.LANCZOS)

# Draw existing nodes + edges + path onto the background
bg = display_img.copy()
draw = ImageDraw.Draw(bg)

PATH_PAIRS = set(zip(st.session_state.path, st.session_state.path[1:])) | \
             set(zip(st.session_state.path[1:], st.session_state.path)) \
             if st.session_state.path else set()

# Edges
for a, b in st.session_state.edges:
    if a in st.session_state.nodes and b in st.session_state.nodes:
        ax, ay = int(st.session_state.nodes[a]["x"] * scale), int(st.session_state.nodes[a]["y"] * scale)
        bx, by = int(st.session_state.nodes[b]["x"] * scale), int(st.session_state.nodes[b]["y"] * scale)
        color = "#00FF88" if (a, b) in PATH_PAIRS or (b, a) in PATH_PAIRS else "#AAAAAA"
        draw.line([(ax, ay), (bx, by)], fill=color, width=3)

# Nodes
for i, (label, pos) in enumerate(st.session_state.nodes.items()):
    px, py = int(pos["x"] * scale), int(pos["y"] * scale)
    is_path = label in st.session_state.path
    is_start = label == st.session_state.edge_start
    color = "#00FF88" if is_path else ("#FFD700" if is_start else "#4A90D9")
    r = 10
    draw.ellipse([(px-r, py-r), (px+r, py+r)], fill=color, outline="white", width=2)
    bbox = draw.textbbox((0, 0), label)
    tw = bbox[2] - bbox[0]
    draw.rectangle([(px - tw//2 - 2, py + r + 1), (px + tw//2 + 2, py + r + 13)], fill="black")
    draw.text((px - tw//2, py + r + 2), label, fill="white")
    if is_path:
        idx = st.session_state.path.index(label)
        draw.text((px + r + 2, py - r), str(idx + 1), fill="#00FF88")

# ── Canvas (captures clicks) ─────────────────────────────────────────────────
st.markdown(f"**Mode: `{mode}`** — {'Click anywhere to place a node' if mode == 'Place Node' else 'Click a node to select it' if mode == 'Draw Edge' else 'Set From/To in sidebar'}")

canvas_result = st_canvas(
    background_image=bg,
    height=dh,
    width=dw,
    drawing_mode="point",        # single click = a point object
    point_display_radius=1,      # nearly invisible — our PIL dots are the UI
    stroke_color="#FF000000",    # transparent
    key=f"canvas_{mode}",        # reset canvas when mode changes
    update_streamlit=True,
)

# ── Handle click ─────────────────────────────────────────────────────────────
def nearest_node(x, y, threshold=15):
    """Return label of nearest node within threshold px (display coords)."""
    best, best_d = None, threshold
    for lbl, pos in st.session_state.nodes.items():
        px, py = int(pos["x"] * scale), int(pos["y"] * scale)
        d = ((px - x)**2 + (py - y)**2)**0.5
        if d < best_d:
            best, best_d = lbl, d
    return best

if canvas_result.json_data:
    objects = canvas_result.json_data.get("objects", [])
    if objects:
        # Take the latest point
        obj = objects[-1]
        cx, cy = int(obj["left"]), int(obj["top"])

        if mode == "Place Node":
            # Convert back to original image coords and store pending
            orig_x, orig_y = int(cx / scale), int(cy / scale)
            if st.session_state.pending_click != (orig_x, orig_y):
                st.session_state.pending_click = (orig_x, orig_y)
                st.rerun()

        elif mode == "Draw Edge":
            hit = nearest_node(cx, cy)
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