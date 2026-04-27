import { useState, useEffect, useRef } from "react";
import StoreLayout from "./StoreLayout";
import { mockMapData } from "./mockData";
import { demoNavigate } from "./demoNavigate";

const API = import.meta.env.VITE_API_URL || "";
const SCALE_X = 82, SCALE_Y = 72, OFFSET_X = 50, OFFSET_Y = 110;
const SVG_W = 1160, SVG_H = 600;

function toSVG(x, y) {
  return [x * SCALE_X + OFFSET_X, SVG_H - (y * SCALE_Y + OFFSET_Y)];
}

function shortLabel(id) {
  if (id === "entrance") return "ENT";
  if (id === "exit")     return "EXIT";
  if (id === "checkout") return "CHK";
  if (!isNaN(id)) return `A${id}`;
  return id.slice(0, 4).toUpperCase();
}

function subLabel(id, name) {
  if (["entrance","exit","checkout"].includes(id)) return null;
  if (!isNaN(id)) return name.replace(/^Aisle \d+ - /, "").split(" ").slice(0, 2).join(" ");
  return null;
}

function aisleColor(id, name = "") {
  const n = name.toLowerCase();
  if (id === "entrance")              return { fill: "#f59e0b", stroke: "#d97706" };
  if (id === "exit")                  return { fill: "#ef4444", stroke: "#dc2626" };
  if (id === "checkout")              return { fill: "#8b5cf6", stroke: "#7c3aed" };
  if (id === "100" || n.includes("dairy"))    return { fill: "#3b82f6", stroke: "#2563eb" };
  if (id === "101" || n.includes("meat"))     return { fill: "#ef4444", stroke: "#dc2626" };
  if (id === "152" || n.includes("bakery"))   return { fill: "#f97316", stroke: "#ea580c" };
  if (id === "34"  || n.includes("yogurt"))   return { fill: "#06b6d4", stroke: "#0891b2" };
  if (["105","351","352"].includes(id) || n.includes("produce") || n.includes("fruit") || n.includes("veg") || n.includes("green")) return { fill: "#22c55e", stroke: "#16a34a" };
  if (["8","40","42"].includes(id) || n.includes("frozen"))     return { fill: "#67e8f9", stroke: "#06b6d4" };
  if (id === "cleaning")              return { fill: "#94a3b8", stroke: "#64748b" };
  if (id === "vitamins")              return { fill: "#a78bfa", stroke: "#7c3aed" };
  if (id === "pharmacy")              return { fill: "#10b981", stroke: "#059669" };
  if (id === "447" || n.includes("deli")) return { fill: "#fb923c", stroke: "#ea580c" };
  if (n.includes("snack") || n.includes("chip")) return { fill: "#fbbf24", stroke: "#f59e0b" };
  if (n.includes("beverage") || n.includes("water") || n.includes("soda")) return { fill: "#38bdf8", stroke: "#0284c7" };
  if (n.includes("baking") || n.includes("coffee")) return { fill: "#d97706", stroke: "#b45309" };
  if (n.includes("breakfast") || n.includes("cereal")) return { fill: "#fde68a", stroke: "#f59e0b" };
  if (n.includes("canned") || n.includes("sauce"))   return { fill: "#f87171", stroke: "#ef4444" };
  if (n.includes("personal") || n.includes("care"))  return { fill: "#f9a8d4", stroke: "#ec4899" };
  if (n.includes("oil") || n.includes("condiment"))  return { fill: "#86efac", stroke: "#22c55e" };
  return { fill: "#6366f1", stroke: "#4f46e5" };
}

function speak(text) {
  if (!window.speechSynthesis) return;
  window.speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.rate = 0.95; u.pitch = 1;
  window.speechSynthesis.speak(u);
}

// Animated moving dot along the route path
function AnimatedDot({ nodes, route, currentStep }) {
  const [t, setT] = useState(0);
  const rafRef = useRef();

  useEffect(() => {
    let start = null;
    const duration = 1800;
    function animate(ts) {
      if (!start) start = ts;
      const progress = Math.min((ts - start) / duration, 1);
      // ease-in-out
      const eased = progress < 0.5 ? 2 * progress * progress : -1 + (4 - 2 * progress) * progress;
      setT(eased);
      if (progress < 1) rafRef.current = requestAnimationFrame(animate);
    }
    rafRef.current = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(rafRef.current);
  }, [currentStep]);

  if (!route.length || !nodes.length) return null;

  const idx = Math.min(currentStep, route.length - 1);
  const nextIdx = Math.min(idx + 1, route.length - 1);
  const curr = nodes.find(n => n.id === route[idx]);
  const next = nodes.find(n => n.id === route[nextIdx]);
  if (!curr || !next) return null;

  const [x1, y1] = toSVG(curr.x, curr.y);
  const [x2, y2] = toSVG(next.x, next.y);
  const cx = x1 + (x2 - x1) * t;
  const cy = y1 + (y2 - y1) * t;

  return (
    <g>
      <circle cx={cx} cy={cy} r={14} fill="#ef4444" opacity={0.18} />
      <circle cx={cx} cy={cy} r={9}  fill="#ef4444" />
      <circle cx={cx} cy={cy} r={4}  fill="#fff" />
    </g>
  );
}

function StoreMap({ nodes, edges, route, currentNode, currentStep }) {
  const [hovered, setHovered] = useState(null);
  if (!nodes.length) return null;

  const hasRoute     = route.length > 0;
  const routeSet     = new Set(route);
  const routeEdgeSet = new Set();
  for (let i = 0; i < route.length - 1; i++) {
    routeEdgeSet.add(`${route[i]}-${route[i+1]}`);
    routeEdgeSet.add(`${route[i+1]}-${route[i]}`);
  }

  return (
    <svg width="100%" viewBox={`0 0 ${SVG_W} ${SVG_H}`}
      style={{ background: "#f0f2f5", borderRadius: 12, display: "block", border: "1px solid #d1d5db" }}>

      {/* Grid lines for store floor feel */}
      {[1,2,3,4].map(i => (
        <line key={`h${i}`} x1={0} y1={i * SVG_H/5} x2={SVG_W} y2={i * SVG_H/5}
          stroke="#e8eaed" strokeWidth={1} strokeDasharray="4 8" />
      ))}

      {/* Edges */}
      {edges.map((e, i) => {
        const from = nodes.find(n => n.id === e.from);
        const to   = nodes.find(n => n.id === e.to);
        if (!from || !to) return null;
        const [x1, y1] = toSVG(from.x, from.y);
        const [x2, y2] = toSVG(to.x,   to.y);
        const isRoute  = routeEdgeSet.has(`${e.from}-${e.to}`);
        return (
          <line key={i} x1={x1} y1={y1} x2={x2} y2={y2}
            stroke={isRoute ? "#1d4ed8" : "#b0b8c4"}
            strokeWidth={isRoute ? 3.5 : 1.5}
            strokeDasharray={isRoute ? "none" : "none"}
            opacity={hasRoute && !isRoute ? 0.4 : 1} />
        );
      })}

      {/* Nodes */}
      {nodes.map(n => {
        const [cx, cy] = toSVG(n.x, n.y);
        const isRoute   = hasRoute && routeSet.has(n.id);
        const isCurrent = hasRoute && n.id === currentNode;
        const isDimmed  = hasRoute && !routeSet.has(n.id);
        const isHovered = hovered === n.id;
        const r = isCurrent ? 20 : 14;
        const label  = shortLabel(n.id);
        const colors = aisleColor(n.id, n.name);
        const fillColor   = isCurrent ? "#ef4444" : isRoute ? "#1d4ed8" : colors.fill;
        const strokeColor = isCurrent ? "#dc2626" : isRoute ? "#1e40af" : colors.stroke;

        return (
          <g key={n.id}
            onMouseEnter={() => setHovered(n.id)}
            onMouseLeave={() => setHovered(null)}
            style={{ cursor: "default" }}>

            {isCurrent && (
              <circle cx={cx} cy={cy} r={r + 12} fill="none" stroke="#ef4444" strokeWidth={2} opacity={0.2} />
            )}
            {isHovered && (
              <circle cx={cx} cy={cy} r={r + 7} fill="none" stroke={strokeColor} strokeWidth={2} opacity={0.3} />
            )}

            <circle cx={cx} cy={cy} r={r}
              fill={fillColor} stroke={strokeColor} strokeWidth={2}
              opacity={isDimmed ? 0.25 : 1} />

            <text x={cx} y={subLabel(n.id, n.name) ? cy - 3 : cy + 1}
              textAnchor="middle" dominantBaseline="middle"
              fontSize="10" fontWeight="800" fill="#fff"
              opacity={isDimmed ? 0.35 : 1}
              style={{ pointerEvents: "none", userSelect: "none" }}>
              {label}
            </text>
            {subLabel(n.id, n.name) && (
              <text x={cx} y={cy + 8} textAnchor="middle" dominantBaseline="middle"
                fontSize="7" fill="rgba(255,255,255,0.75)"
                opacity={isDimmed ? 0.35 : 1}
                style={{ pointerEvents: "none", userSelect: "none" }}>
                {subLabel(n.id, n.name)}
              </text>
            )}

            {isHovered && (() => {
              const tipW = Math.max(120, n.name.replace(/^Aisle \d+ - /, "").length * 7 + 20);
              return (
                <g>
                  <rect x={cx - tipW/2} y={cy - r - 36} width={tipW} height={24} rx={6} fill="#1e293b" opacity={0.92} />
                  <text x={cx} y={cy - r - 20} textAnchor="middle"
                    fontSize="11" fill="#fff" fontWeight="500"
                    style={{ pointerEvents: "none", userSelect: "none" }}>
                    {n.name.replace(/^Aisle \d+ - /, "")}
                  </text>
                </g>
              );
            })()}
          </g>
        );
      })}

      {/* Animated position dot */}
      {hasRoute && <AnimatedDot nodes={nodes} route={route} currentStep={currentStep} />}

      {/* Legend */}
      <g transform={`translate(16, ${SVG_H - 24})`}>
        <circle cx={8}   cy={0} r={7} fill="#ef4444" />
        <text x={20} y={4} fontSize="11" fill="#94a3b8">You are here</text>
        <circle cx={105} cy={0} r={7} fill="#1d4ed8" />
        <text x={117} y={4} fontSize="11" fill="#94a3b8">Your route</text>
        <circle cx={195} cy={0} r={7} fill="#e8ecf0" stroke="#c8cdd5" strokeWidth={1.5} />
        <text x={207} y={4} fontSize="11" fill="#94a3b8">Aisle</text>
      </g>
    </svg>
  );
}

// Direction arrow icon
function DirIcon({ dir }) {
  const arrows = { ST: "↑", TL: "↰", TR: "↱" };
  const colors  = { ST: "#64748b", TL: "#d97706", TR: "#2563eb" };
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", justifyContent: "center",
      width: 28, height: 28, borderRadius: 8, fontSize: 16,
      background: colors[dir] + "15", color: colors[dir], fontWeight: 700,
    }}>
      {arrows[dir] || "↑"}
    </span>
  );
}

export default function App() {
  const [tab, setTab]                 = useState("nav");
  const [mapData, setMapData]         = useState({ nodes: [], edges: [], store: "", address: "" });
  const [items, setItems]             = useState("");
  const [result, setResult]           = useState(null);
  const [loading, setLoading]         = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [apiStatus, setApiStatus]     = useState("checking");
  const activeRef                     = useRef(null);

  useEffect(() => {
    fetch(`${API}/map`)
      .then(r => r.json())
      .then(d => { setMapData(d); setApiStatus("online"); })
      .catch(() => setApiStatus("offline"));
  }, []);

  useEffect(() => {
    if (activeRef.current) activeRef.current.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [currentStep]);

  async function navigate() {
    setLoading(true); setResult(null); setCurrentStep(0);
    const itemList = items.split(",").map(s => s.trim()).filter(Boolean);

    if (!API) {
      // Demo mode: run client-side navigation on the mock store graph
      const data = demoNavigate(itemList);
      setResult(data);
      if (data.directions?.[0]) {
        speak(`Demo route ready. ${data.directions.length} stops. Starting at ${data.directions[0].name}.`);
      }
      setLoading(false);
      return;
    }

    try {
      const res = await fetch(`${API}/navigate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ items: itemList }),
      });
      const data = await res.json();
      setResult(data);
      if (data.directions?.[0]) {
        speak(`Route ready. ${data.directions.length} stops. Starting at ${data.directions[0].name}.`);
      }
    } catch {
      speak("Failed to connect to the server.");
    }
    setLoading(false);
  }

  function goToStep(idx) {
    setCurrentStep(idx);
    const dir = result?.directions?.find(d => result.route.indexOf(d.target) === idx);
    if (dir) speak(`Step ${dir.step}. ${dir.name}. ${dir.dir_label}. ${dir.audio || ""}`);
  }

  const currentNode = result?.route?.[currentStep] ?? "entrance";
  const directions  = result?.directions ?? [];
  const totalStops  = directions.length;

  const EXAMPLES = ["milk, eggs, pasta", "chips, salsa, guacamole", "coffee, oatmeal, bananas", "chicken, broccoli, rice"];

  return (
    <div style={{ fontFamily: "'Inter', -apple-system, sans-serif", background: "#f1f4f8", minHeight: "100vh", color: "#1e293b" }}>

      {/* ── Header ── */}
      <div style={{
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "0 28px", height: 56,
        borderBottom: "1px solid #e2e6ea",
        background: "#fff", boxShadow: "0 1px 4px rgba(0,0,0,0.06)",
      }}>
        {/* Logo */}
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{
            width: 32, height: 32, borderRadius: 8, background: "linear-gradient(135deg,#2563eb,#7c3aed)",
            display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0,
          }}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M3 11l19-9-9 19-2-8-8-2z"/>
            </svg>
          </div>
          <div>
            <span style={{ fontSize: 17, fontWeight: 800, color: "#2563eb", letterSpacing: -0.5 }}>Wayfinder</span>
            <span style={{ fontSize: 17, fontWeight: 300, color: "#64748b", letterSpacing: -0.5 }}>AI</span>
          </div>
        </div>

        {/* Store badge */}
        {mapData.store && (
          <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
            <div style={{
              width: 7, height: 7, borderRadius: "50%",
              background: apiStatus === "online" ? "#22c55e" : "#f59e0b",
            }} />
            <div>
              <div style={{ fontSize: 12, fontWeight: 600, color: "#1e293b", lineHeight: 1.2 }}>{mapData.store}</div>
              {mapData.address && <div style={{ fontSize: 10, color: "#94a3b8" }}>{mapData.address}</div>}
            </div>
          </div>
        )}

        {/* Tabs */}
        <div style={{ display: "flex", gap: 2, alignItems: "center" }}>
          {[["nav", "Navigate"], ["map", "Store Map"]].map(([t, label]) => (
            <button key={t} onClick={() => setTab(t)} style={{
              padding: "6px 16px", borderRadius: 8, fontSize: 13, cursor: "pointer",
              border: tab === t ? "1px solid #2563eb" : "1px solid transparent",
              background: tab === t ? "#eff6ff" : "transparent",
              color: tab === t ? "#2563eb" : "#94a3b8",
              fontWeight: tab === t ? 600 : 400,
            }}>{label}</button>
          ))}
        </div>
      </div>


      {tab === "map" && <StoreLayout />}

      {tab === "nav" && (
        <div style={{ display: "flex", flexDirection: "column", height: "calc(100vh - 56px)" }}>
        <div style={{ display: "flex", flex: 1, overflow: "hidden" }}>

          {/* ── Sidebar ── */}
          <div style={{
            width: 360, borderRight: "1px solid #e2e6ea", background: "#fff",
            display: "flex", flexDirection: "column", flexShrink: 0,
            boxShadow: "2px 0 8px rgba(0,0,0,0.04)",
          }}>

            {/* Search */}
            <div style={{ padding: "20px 20px 14px", borderBottom: "1px solid #f1f5f9" }}>
              <p style={{ color: "#94a3b8", fontSize: 11, margin: "0 0 8px", textTransform: "uppercase", letterSpacing: 0.8, fontWeight: 600 }}>
                Shopping List
              </p>
              <div style={{ display: "flex", gap: 8 }}>
                <input
                  value={items}
                  onChange={e => setItems(e.target.value)}
                  onKeyDown={e => e.key === "Enter" && navigate()}
                  placeholder="milk, pasta, chips..."
                  style={{
                    flex: 1, padding: "10px 14px", borderRadius: 8,
                    border: "1px solid #e2e6ea", background: "#f8fafc",
                    color: "#1e293b", fontSize: 14, outline: "none", fontFamily: "inherit",
                  }}
                />
              </div>

              {/* Example chips */}
              <div style={{ display: "flex", flexWrap: "wrap", gap: 5, marginTop: 8 }}>
                {EXAMPLES.map(ex => (
                  <button key={ex} onClick={() => setItems(ex)} style={{
                    padding: "3px 10px", borderRadius: 20, border: "1px solid #e2e6ea",
                    background: "#f8fafc", color: "#475569", fontSize: 11, cursor: "pointer",
                    fontFamily: "inherit",
                  }}>
                    {ex}
                  </button>
                ))}
              </div>

              <button onClick={navigate} disabled={loading || !items.trim()} style={{
                width: "100%", marginTop: 10, padding: "10px 0", borderRadius: 8,
                border: "none",
                background: loading || !items.trim()
                  ? "#e2e6ea"
                  : "linear-gradient(135deg, #2563eb, #7c3aed)",
                color: loading || !items.trim() ? "#94a3b8" : "#fff",
                fontSize: 14, fontWeight: 600,
                cursor: loading || !items.trim() ? "not-allowed" : "pointer",
                fontFamily: "inherit",
              }}>
                {loading ? "Finding best route..." : "Get Directions →"}
              </button>

              {result?.not_found?.length > 0 && (
                <div style={{ marginTop: 8, padding: "8px 12px", background: "#fef2f2", borderRadius: 6, border: "1px solid #fecaca" }}>
                  <p style={{ color: "#dc2626", fontSize: 12, margin: 0 }}>
                    Not found: {result.not_found.join(", ")}
                  </p>
                </div>
              )}

              {/* Route summary */}
              {result && (
                <div style={{ marginTop: 10, display: "flex", gap: 8 }}>
                  <div style={{ flex: 1, padding: "8px 12px", background: "#f0fdf4", borderRadius: 8, border: "1px solid #bbf7d0", textAlign: "center" }}>
                    <div style={{ fontSize: 18, fontWeight: 800, color: "#16a34a" }}>{totalStops}</div>
                    <div style={{ fontSize: 10, color: "#94a3b8" }}>stops</div>
                  </div>
                  <div style={{ flex: 1, padding: "8px 12px", background: "#eff6ff", borderRadius: 8, border: "1px solid #bfdbfe", textAlign: "center" }}>
                    <div style={{ fontSize: 18, fontWeight: 800, color: "#2563eb" }}>{result.route.length}</div>
                    <div style={{ fontSize: 10, color: "#94a3b8" }}>waypoints</div>
                  </div>
                  <div style={{ flex: 1, padding: "8px 12px", background: "#fdf4ff", borderRadius: 8, border: "1px solid #e9d5ff", textAlign: "center" }}>
                    <div style={{ fontSize: 18, fontWeight: 800, color: "#7c3aed" }}>~{Math.ceil(result.route.length * 0.5)}</div>
                    <div style={{ fontSize: 10, color: "#94a3b8" }}>min est.</div>
                  </div>
                </div>
              )}
            </div>

            {/* Directions list */}
            <div style={{ flex: 1, overflowY: "auto", padding: "8px 0" }}>
              {directions.length === 0 && !loading && (
                <div style={{ padding: "40px 24px", textAlign: "center" }}>
                  <div style={{ fontSize: 40, marginBottom: 12 }}>🛒</div>
                  <p style={{ color: "#cbd5e1", fontSize: 13, margin: 0 }}>
                    Enter items above to get<br />step-by-step directions
                  </p>
                </div>
              )}

              {loading && (
                <div style={{ padding: "40px 24px", textAlign: "center" }}>
                  <div style={{ width: 36, height: 36, margin: "0 auto 12px", border: "3px solid #e2e6ea", borderTop: "3px solid #2563eb", borderRadius: "50%", animation: "spin 0.8s linear infinite" }} />
                  <p style={{ color: "#94a3b8", fontSize: 13, margin: 0 }}>Optimizing route...</p>
                </div>
              )}

              {directions.map((d, i) => {
                const isActive = result?.route?.indexOf(d.target) === currentStep;
                const dirColors = { TL: "#d97706", TR: "#2563eb", ST: "#64748b" };
                const dirColor  = dirColors[d.direction] || "#64748b";

                return (
                  <div key={i}
                    ref={isActive ? activeRef : null}
                    onClick={() => goToStep(result.route.indexOf(d.target))}
                    style={{
                      margin: "0 12px 4px",
                      padding: "12px 14px",
                      borderRadius: 10,
                      cursor: "pointer",
                      background: isActive ? "#eff6ff" : "#fafbfc",
                      border: `1px solid ${isActive ? "#bfdbfe" : "#e2e8f0"}`,
                      transition: "all 0.15s",
                      boxShadow: isActive ? "0 2px 8px rgba(37,99,235,0.1)" : "none",
                    }}
                  >
                    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
                      <DirIcon dir={d.direction} />
                      <div style={{ flex: 1 }}>
                        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                          <span style={{ color: isActive ? "#1e40af" : "#334155", fontWeight: 600, fontSize: 13 }}>
                            {d.name}
                          </span>
                          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                            <span style={{ fontSize: 10, color: "#94a3b8" }}>{d.step}/{d.total}</span>
                            <button
                              onClick={e => { e.stopPropagation(); speak(`${d.name}. ${d.dir_label}. ${d.audio || ""}`); }}
                              style={{
                                padding: "2px 6px", borderRadius: 5, border: "1px solid #e2e6ea",
                                background: "#f8fafc", color: "#64748b", fontSize: 11, cursor: "pointer",
                              }}
                              title="Read aloud"
                            >🔊</button>
                          </div>
                        </div>
                        <span style={{ fontSize: 11, color: dirColor, fontWeight: 600 }}>{d.dir_label}</span>
                      </div>
                    </div>

                    {d.walk.length > 0 && (
                      <div style={{ fontSize: 11, color: "#64748b", marginBottom: 5, lineHeight: 1.5 }}>
                        {d.walk.map((s, j) => (
                          <span key={j}>
                            <span style={{ color: isActive ? "#475569" : "#94a3b8" }}>{s.replace(/^Aisle \d+ - /, "")}</span>
                            {j < d.walk.length - 1 && <span style={{ color: "#cbd5e1", margin: "0 3px" }}>→</span>}
                          </span>
                        ))}
                      </div>
                    )}

                    {d.audio && isActive && (
                      <div style={{ fontSize: 11, color: "#64748b", fontStyle: "italic", marginBottom: 5, lineHeight: 1.4 }}>
                        {d.audio.split(".")[0]}.
                      </div>
                    )}

                    {d.items.map((item, j) => (
                      <div key={j} style={{ fontSize: 12, color: isActive ? "#334155" : "#64748b", paddingLeft: 4, lineHeight: 1.7 }}>
                        · {item}
                      </div>
                    ))}
                  </div>
                );
              })}
            </div>

            {/* Prev / Next */}
            {result && (
              <div style={{ padding: "12px 16px", borderTop: "1px solid #e2e6ea" }}>
                {/* Progress bar */}
                <div style={{ height: 4, background: "#e2e6ea", borderRadius: 2, marginBottom: 10, overflow: "hidden" }}>
                  <div style={{
                    height: "100%", borderRadius: 2,
                    background: "linear-gradient(90deg,#2563eb,#7c3aed)",
                    width: `${(currentStep / Math.max(result.route.length - 1, 1)) * 100}%`,
                    transition: "width 0.3s ease",
                  }} />
                </div>
                <div style={{ display: "flex", gap: 8 }}>
                  <button onClick={() => goToStep(Math.max(0, currentStep - 1))}
                    style={{ flex: 1, padding: "9px 0", borderRadius: 8, border: "1px solid #e2e6ea", background: "#f8fafc", color: "#475569", cursor: "pointer", fontSize: 13, fontFamily: "inherit" }}>
                    ← Prev
                  </button>
                  <button onClick={() => goToStep(Math.min(result.route.length - 1, currentStep + 1))}
                    style={{ flex: 1, padding: "9px 0", borderRadius: 8, border: "none", background: "linear-gradient(135deg,#2563eb,#7c3aed)", color: "#fff", cursor: "pointer", fontSize: 13, fontWeight: 600, fontFamily: "inherit" }}>
                    Next →
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* ── Map ── */}
          <div style={{ flex: 1, padding: 24, overflowY: "auto", display: "flex", flexDirection: "column", background: "#f1f4f8" }}>
            {/* Map header */}
            {result && (
              <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14 }}>
                <div style={{ flex: 1, height: 1, background: "#e2e6ea" }} />
                <span style={{ fontSize: 12, color: "#94a3b8", fontWeight: 500 }}>
                  Step {currentStep + 1} of {result.route.length} · {result.store}
                </span>
                <div style={{ flex: 1, height: 1, background: "#e2e6ea" }} />
              </div>
            )}
            <StoreMap
              nodes={mapData.nodes?.length ? mapData.nodes : mockMapData.nodes}
              edges={mapData.edges?.length ? mapData.edges : mockMapData.edges}
              route={result?.route || []}
              currentNode={currentNode}
              currentStep={currentStep}
            />
            {!result && (
              <div style={{ textAlign: "center", padding: "40px 0", color: "#94a3b8" }}>
                <p style={{ fontSize: 13 }}>Your route will appear here after getting directions.</p>
              </div>
            )}
          </div>
        </div>
        </div>
      )}

      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}
