import { useState, useEffect, useRef } from "react";
import { mockMapData, MOCK_STORE_NAME, MOCK_AISLE_INFO } from "./mockData";

const API = import.meta.env.VITE_API_URL || "";
const SCALE_X = 82, SCALE_Y = 72, OFFSET_X = 50, OFFSET_Y = 110;
const SVG_W = 1160, SVG_H = 600;

function toSVG(x, y) {
  return [x * SCALE_X + OFFSET_X, SVG_H - (y * SCALE_Y + OFFSET_Y)];
}

function nodeColor(id, name = "") {
  const n = name.toLowerCase();
  if (id === "entrance")  return "#f59e0b";
  if (id === "exit")      return "#ef4444";
  if (id === "checkout" || id.startsWith("checkout")) return "#8b5cf6";
  if (id === "100" || n.includes("dairy"))   return "#3b82f6";
  if (id === "101" || n.includes("meat"))    return "#ef4444";
  if (id === "152" || n.includes("bakery"))  return "#f97316";
  if (id === "34"  || n.includes("yogurt"))  return "#06b6d4";
  if (["105","351","352"].includes(id) || n.includes("produce") || n.includes("fruit") || n.includes("veg") || n.includes("green")) return "#22c55e";
  if (["8","40","42"].includes(id) || n.includes("frozen"))  return "#67e8f9";
  if (id === "pharmacy")  return "#10b981";
  if (id === "cleaning")  return "#94a3b8";
  if (id === "vitamins")  return "#a78bfa";
  if (id === "447" || n.includes("deli"))    return "#fb923c";
  if (n.includes("snack") || n.includes("popcorn") || n.includes("chip")) return "#fbbf24";
  if (n.includes("beverage") || n.includes("water") || n.includes("soda")) return "#38bdf8";
  if (n.includes("baking") || n.includes("coffee") || n.includes("flour")) return "#d97706";
  if (n.includes("breakfast") || n.includes("cereal") || n.includes("oat")) return "#fde68a";
  if (n.includes("canned") || n.includes("sauce") || n.includes("bean"))   return "#f87171";
  if (n.includes("international") || n.includes("bread") || n.includes("tortilla")) return "#c084fc";
  if (n.includes("personal") || n.includes("care") || n.includes("soap"))  return "#f9a8d4";
  if (n.includes("oil") || n.includes("condiment"))  return "#86efac";
  if (n.includes("granola") || n.includes("syrup"))  return "#fcd34d";
  if (n.includes("pasta") || n.includes("dry") || n.includes("rice"))      return "#fed7aa";
  if (n.includes("front") || n.includes("perimeter")) return "#64748b";
  return "#6366f1";
}

// Parse CSV: id,name,x,y  (edges optional: from_id,to_id on rows after a blank line)
function parseCSV(text) {
  const lines = text.trim().split(/\r?\n/).map(l => l.trim()).filter(Boolean);
  const nodes = [], edges = [];
  let inEdges = false;

  for (const line of lines) {
    if (line.startsWith("#")) continue;
    const parts = line.split(",").map(s => s.trim());
    if (parts.length === 2) { inEdges = true; }
    if (inEdges && parts.length >= 2) {
      edges.push({ from: parts[0], to: parts[1] });
    } else if (!inEdges && parts.length >= 4) {
      const [id, name, x, y] = parts;
      if (id === "id") continue; // header row
      nodes.push({ id, name, x: parseFloat(x), y: parseFloat(y) });
    }
  }
  return { nodes, edges };
}

// ── Upload Screen ──────────────────────────────────────────────────────────────
function UploadScreen({ onLoaded }) {
  const [dragging, setDragging]   = useState(false);
  const [status, setStatus]       = useState("idle"); // idle | loading | error
  const [message, setMessage]     = useState("");
  const [fileName, setFileName]   = useState("");
  const inputRef = useRef();

  function handleFile(file) {
    if (!file) return;
    if (!file.name.endsWith(".csv")) {
      setStatus("error");
      setMessage("Please upload a .csv file.");
      return;
    }
    setFileName(file.name);
    setStatus("loading");
    setMessage("");
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const parsed = parseCSV(e.target.result);
        if (parsed.nodes.length === 0) {
          setStatus("error");
          setMessage("No nodes found. Check your CSV format.");
          return;
        }
        setStatus("idle");
        onLoaded(parsed, file.name);
      } catch {
        setStatus("error");
        setMessage("Failed to parse CSV. Check the format.");
      }
    };
    reader.readAsText(file);
  }

  function onDrop(e) {
    e.preventDefault();
    setDragging(false);
    handleFile(e.dataTransfer.files[0]);
  }

  function useDefault() {
    if (!API) {
      setStatus("error");
      setMessage("No backend configured. Set VITE_API_URL to connect.");
      return;
    }
    setStatus("loading");
    setMessage("Loading live store data...");
    fetch(`${API}/map`)
      .then(r => r.json())
      .then(d => {
        setStatus("idle");
        onLoaded({ nodes: d.nodes, edges: d.edges }, d.store);
      })
      .catch(() => {
        setStatus("error");
        setMessage("Could not connect to backend. Is it running?");
      });
  }

  return (
    <div style={{
      minHeight: "100vh", background: "#f1f4f8", display: "flex",
      flexDirection: "column", alignItems: "center", justifyContent: "center",
      fontFamily: "'Inter', sans-serif", color: "#1e293b", padding: 40,
    }}>
      {/* Header */}
      <div style={{ textAlign: "center", marginBottom: 40 }}>
        <div style={{ fontSize: 48, marginBottom: 12 }}>🏪</div>
        <h1 style={{ color: "#2563eb", margin: "0 0 8px", fontSize: 28, fontWeight: 800 }}>Store Layout Admin</h1>
        <p style={{ color: "#94a3b8", margin: 0, fontSize: 14 }}>Agent 1 — Upload your store blueprint to get started</p>
      </div>

      {/* Drop Zone */}
      <div
        onDragOver={e => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current.click()}
        style={{
          width: 420, padding: "40px 32px", borderRadius: 16, cursor: "pointer",
          border: `2px dashed ${dragging ? "#2563eb" : "#cbd5e1"}`,
          background: dragging ? "#eff6ff" : "#fff",
          textAlign: "center", transition: "all 0.2s",
        }}
      >
        <input ref={inputRef} type="file" accept=".csv" style={{ display: "none" }}
          onChange={e => handleFile(e.target.files[0])} />

        <div style={{ fontSize: 36, marginBottom: 12 }}>
          {status === "loading" ? "⏳" : dragging ? "📂" : "📄"}
        </div>
        <p style={{ color: "#1e293b", margin: "0 0 6px", fontSize: 15, fontWeight: 600 }}>
          {status === "loading" ? "Processing..." : "Drop your CSV file here"}
        </p>
        <p style={{ color: "#64748b", margin: 0, fontSize: 13 }}>
          or click to browse
        </p>

        {fileName && status !== "error" && (
          <div style={{ marginTop: 14, color: "#2563eb", fontSize: 13 }}>✓ {fileName}</div>
        )}
      </div>

      {/* Status message */}
      <div style={{ marginTop: 16, minHeight: 24, textAlign: "center" }}>
        {status === "error" && (
          <p style={{ color: "#ef4444", fontSize: 13, margin: 0 }}>⚠ {message}</p>
        )}
        {status === "loading" && message && (
          <p style={{ color: "#94a3b8", fontSize: 13, margin: 0 }}>{message}</p>
        )}
      </div>

      {/* Divider */}
      <div style={{ display: "flex", alignItems: "center", gap: 12, margin: "28px 0", width: 420 }}>
        <div style={{ flex: 1, height: 1, background: "#e2e6ea" }} />
        <span style={{ color: "#475569", fontSize: 12 }}>or</span>
        <div style={{ flex: 1, height: 1, background: "#e2e6ea" }} />
      </div>

      {/* Use live data button */}
      <button onClick={useDefault} style={{
        padding: "12px 32px", borderRadius: 10, border: "1px solid #cbd5e1",
        background: "transparent", color: "#94a3b8", fontSize: 14, cursor: "pointer",
        transition: "all 0.2s",
      }}
        onMouseEnter={e => { e.target.style.borderColor = "#2563eb"; e.target.style.color = "#2563eb"; }}
        onMouseLeave={e => { e.target.style.borderColor = "#cbd5e1"; e.target.style.color = "#94a3b8"; }}
      >
        Connect to backend
      </button>

      {/* CSV format hint */}
      <div style={{
        marginTop: 40, padding: "16px 20px", background: "#fff",
        borderRadius: 10, border: "1px solid #e2e6ea", width: 420, boxSizing: "border-box",
      }}>
        <p style={{ color: "#64748b", fontSize: 12, margin: "0 0 8px", fontWeight: 600, letterSpacing: 0.5 }}>
          CSV FORMAT
        </p>
        <pre style={{ color: "#475569", fontSize: 11, margin: 0, lineHeight: 1.7 }}>{`id,name,x,y
entrance,Main Entrance,4.5,-0.8
1,Dairy & Bakery,1.5,2.5
2,Dry Goods,2.5,2.5
checkout_1,Checkout Lane 1,11.0,2.0
exit,Exit,11.0,5.0`}</pre>
      </div>
    </div>
  );
}

// ── Map View ───────────────────────────────────────────────────────────────────
function MapView({ mapData, sourceName, onReset }) {
  const [selected, setSelected]   = useState(null);
  const [aisleInfo, setAisleInfo] = useState(null);
  const [loading, setLoading]     = useState(false);

  const { nodes = [], edges = [] } = mapData;

  async function selectNode(node) {
    if (selected?.id === node.id) { setSelected(null); setAisleInfo(null); return; }
    setSelected(node);
    setLoading(true);
    setAisleInfo(null);
    if (!API) {
      setAisleInfo(MOCK_AISLE_INFO[node.id] ?? { name: node.name, audio: "", items: [] });
      setLoading(false);
      return;
    }
    try {
      const res = await fetch(`${API}/aisle/${node.id}`);
      setAisleInfo(await res.json());
    } catch {
      setAisleInfo(MOCK_AISLE_INFO[node.id] ?? { name: node.name, audio: "", items: [] });
    }
    setLoading(false);
  }

  const groupedItems = aisleInfo?.items?.reduce((acc, item) => {
    if (!acc[item.side]) acc[item.side] = [];
    acc[item.side].push(item);
    return acc;
  }, {}) ?? {};

  return (
    <div style={{ display: "flex", minHeight: "100vh", background: "#f1f4f8", color: "#1e293b", fontFamily: "sans-serif" }}>

      {/* Map panel */}
      <div style={{ flex: 1, padding: 24 }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 4 }}>
          <h2 style={{ color: "#2563eb", margin: 0 }}>Store Layout</h2>
          <button onClick={onReset} style={{
            padding: "6px 14px", borderRadius: 8, border: "1px solid #cbd5e1",
            background: "#fff", color: "#64748b", fontSize: 12, cursor: "pointer",
          }}>
            ↩ Upload / Connect
          </button>
        </div>
        <p style={{ color: "#64748b", margin: "0 0 14px", fontSize: 13 }}>
          {sourceName} · {nodes.length} sections · Click any section to explore
        </p>

        {/* Legend */}
        <div style={{ display: "flex", flexWrap: "wrap", gap: "6px 14px", marginBottom: 14 }}>
          {[
            ["Entrance", "#f59e0b"],
            ["Checkout", "#8b5cf6"],
            ["Dairy", "#3b82f6"],
            ["Meat", "#ef4444"],
            ["Produce", "#22c55e"],
            ["Frozen", "#67e8f9"],
            ["Bakery", "#f97316"],
            ["Pharmacy", "#10b981"],
            ["Snacks", "#fbbf24"],
            ["Other", "#6366f1"],
          ].map(([label, color]) => (
            <div key={label} style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 11, color: "#64748b" }}>
              <div style={{ width: 9, height: 9, borderRadius: 2, background: color }} />
              {label}
            </div>
          ))}
        </div>

        <svg width="100%" viewBox={`0 0 ${SVG_W} ${SVG_H + 40}`}
          style={{ background: "#f0f2f5", borderRadius: 14, border: "1px solid #d1d5db" }}>

          {edges.map((e, i) => {
            const from = nodes.find(n => n.id === e.from);
            const to   = nodes.find(n => n.id === e.to);
            if (!from || !to) return null;
            const [x1, y1] = toSVG(from.x, from.y);
            const [x2, y2] = toSVG(to.x, to.y);
            return <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} stroke="#cbd5e1" strokeWidth={1.5} />;
          })}

          {nodes.map(n => {
            const [cx, cy] = toSVG(n.x, n.y);
            const isSelected = selected?.id === n.id;
            const color = nodeColor(n.id, n.name);
            const r = isSelected ? 22 : 16;

            return (
              <g key={n.id} onClick={() => selectNode(n)} style={{ cursor: "pointer" }}>
                {isSelected && (
                  <circle cx={cx} cy={cy} r={r + 8} fill="none" stroke={color} strokeWidth={2} opacity={0.35} />
                )}
                <rect x={cx - r} y={cy - r} width={r * 2} height={r * 2} rx={5}
                  fill={color}
                  stroke={isSelected ? "#2563eb" : "#f1f4f8"}
                  strokeWidth={isSelected ? 2 : 1}
                  opacity={selected && !isSelected ? 0.4 : 1}
                />
                {/* Aisle code (A1, A2 …) */}
                <text x={cx} y={!isNaN(n.id) ? cy - 3 : cy + 1}
                  textAnchor="middle" dominantBaseline="middle"
                  fontSize="9" fill="#fff" fontWeight="800"
                  style={{ pointerEvents: "none", userSelect: "none" }}>
                  {!isNaN(n.id) ? `A${n.id}` : n.id === "entrance" ? "ENT" : n.id === "exit" ? "EXIT" : n.id === "checkout" ? "CHK" : n.id.substring(0, 4).toUpperCase()}
                </text>
                {/* Short dept name under code */}
                {!isNaN(n.id) && (
                  <text x={cx} y={cy + 7} textAnchor="middle" dominantBaseline="middle"
                    fontSize="6" fill="rgba(255,255,255,0.75)"
                    style={{ pointerEvents: "none", userSelect: "none" }}>
                    {n.name.replace(/^Aisle \d+ - /, "").split(" ").slice(0, 2).join(" ").substring(0, 10)}
                  </text>
                )}
              </g>
            );
          })}
        </svg>
      </div>

      {/* Detail panel */}
      <div style={{
        width: 300, minHeight: "100vh", background: "#fff",
        borderLeft: "1px solid #e2e6ea", padding: 24, boxSizing: "border-box",
      }}>
        {!selected ? (
          <div style={{ marginTop: 80, textAlign: "center", color: "#334155" }}>
            <div style={{ fontSize: 36, marginBottom: 10 }}>👈</div>
            <p style={{ fontSize: 13 }}>Click any section on the map to see its contents</p>
          </div>
        ) : loading ? (
          <p style={{ color: "#64748b", marginTop: 40 }}>Loading...</p>
        ) : aisleInfo ? (
          <>
            <div style={{
              background: `${nodeColor(selected.id, selected.name)}18`,
              border: `1px solid ${nodeColor(selected.id, selected.name)}`,
              borderRadius: 10, padding: 14, marginBottom: 16,
            }}>
              <h3 style={{ margin: "0 0 6px", color: nodeColor(selected.id, selected.name), fontSize: 15 }}>
                {aisleInfo.name}
              </h3>
              {aisleInfo.audio && (
                <p style={{ margin: 0, color: "#64748b", fontSize: 12, fontStyle: "italic", lineHeight: 1.5 }}>
                  "{aisleInfo.audio}"
                </p>
              )}
            </div>

            {aisleInfo.items.length === 0 ? (
              <p style={{ color: "#475569", fontSize: 13 }}>No product data for this section.</p>
            ) : (
              <>
                <p style={{ color: "#64748b", fontSize: 12, margin: "0 0 12px" }}>{aisleInfo.items.length} products</p>
                {Object.entries(groupedItems).map(([side, items]) => (
                  <div key={side} style={{ marginBottom: 14 }}>
                    <div style={{
                      fontSize: 11, color: "#2563eb", fontWeight: "bold",
                      textTransform: "uppercase", letterSpacing: 1,
                      marginBottom: 5, borderBottom: "1px solid #e2e6ea", paddingBottom: 3,
                    }}>
                      {side}
                    </div>
                    {items.map((item, i) => (
                      <div key={i} style={{
                        display: "flex", justifyContent: "space-between",
                        padding: "5px 0", borderBottom: "1px solid #e2e6ea",
                      }}>
                        <span style={{ color: "#1e293b", fontSize: 13, textTransform: "capitalize" }}>
                          {item.product}
                        </span>
                        <span style={{ fontSize: 11, color: "#475569", background: "#f1f4f8", padding: "2px 6px", borderRadius: 4 }}>
                          {item.shelf}
                        </span>
                      </div>
                    ))}
                  </div>
                ))}
              </>
            )}
          </>
        ) : (
          <p style={{ color: "#475569", marginTop: 40, fontSize: 13 }}>No data available for this section.</p>
        )}
      </div>
    </div>
  );
}

// ── Root ───────────────────────────────────────────────────────────────────────
export default function StoreLayout() {
  const [mapData, setMapData]       = useState(mockMapData);
  const [sourceName, setSourceName] = useState(MOCK_STORE_NAME);

  if (!mapData) {
    return (
      <UploadScreen onLoaded={(data, name) => {
        setMapData(data);
        setSourceName(name);
      }} />
    );
  }

  return (
    <MapView
      mapData={mapData}
      sourceName={sourceName}
      onReset={() => { setMapData(mockMapData); setSourceName(MOCK_STORE_NAME); }}
    />
  );
}
