import { useState, useRef, useEffect } from "react";

const API = import.meta.env.VITE_API_URL || "";

const SCALE_X = 82, SCALE_Y = 72, OFFSET_X = 50, OFFSET_Y = 110;
const SVG_W = 1160, SVG_H = 600;
function toSVG(x, y) { return [x * SCALE_X + OFFSET_X, SVG_H - (y * SCALE_Y + OFFSET_Y)]; }

function nodeColor(id, name = "") {
  const n = name.toLowerCase();
  if (id === "entrance")  return "#f59e0b";
  if (id === "exit")      return "#ef4444";
  if (id === "checkout")  return "#8b5cf6";
  if (id === "100" || n.includes("dairy"))   return "#3b82f6";
  if (id === "101" || n.includes("meat"))    return "#ef4444";
  if (id === "152" || n.includes("bakery"))  return "#f97316";
  if (id === "34"  || n.includes("yogurt"))  return "#06b6d4";
  if (["105","351","352"].includes(id) || n.includes("produce") || n.includes("fruit") || n.includes("veg")) return "#22c55e";
  if (["8","40","42"].includes(id) || n.includes("frozen")) return "#67e8f9";
  if (id === "pharmacy")  return "#10b981";
  if (id === "cleaning")  return "#94a3b8";
  if (id === "vitamins")  return "#a78bfa";
  if (id === "447" || n.includes("deli")) return "#fb923c";
  if (n.includes("snack") || n.includes("popcorn")) return "#fbbf24";
  if (n.includes("beverage") || n.includes("water")) return "#38bdf8";
  if (n.includes("baking") || n.includes("coffee"))  return "#d97706";
  if (n.includes("canned") || n.includes("sauce"))   return "#f87171";
  if (n.includes("personal") || n.includes("care"))  return "#f9a8d4";
  if (n.includes("front") || n.includes("perimeter")) return "#64748b";
  return "#6366f1";
}

// ── Route map ──────────────────────────────────────────────────────────────────
function RouteMap({ nodes, edges, route, currentNode }) {
  const [hovered, setHovered] = useState(null);
  if (!nodes.length) return null;
  const routeSet = new Set(route);
  const routeEdgeSet = new Set();
  for (let i = 0; i < route.length - 1; i++) {
    routeEdgeSet.add(`${route[i]}-${route[i+1]}`);
    routeEdgeSet.add(`${route[i+1]}-${route[i]}`);
  }
  return (
    <svg width="100%" viewBox={`0 0 ${SVG_W} ${SVG_H}`}
      style={{ background: "#f0f2f5", borderRadius: 10, border: "1px solid #d1d5db", display: "block" }}>
      {edges.map((e, i) => {
        const f = nodes.find(n => n.id === e.from), t = nodes.find(n => n.id === e.to);
        if (!f || !t) return null;
        const [x1,y1] = toSVG(f.x,f.y), [x2,y2] = toSVG(t.x,t.y);
        const isR = routeEdgeSet.has(`${e.from}-${e.to}`);
        return <line key={i} x1={x1} y1={y1} x2={x2} y2={y2}
          stroke={isR ? "#1d4ed8" : "#b0b8c4"} strokeWidth={isR ? 3 : 1.5} opacity={route.length && !isR ? 0.4 : 1}/>;
      })}
      {nodes.map(n => {
        const [cx,cy] = toSVG(n.x,n.y);
        const isCurr = n.id === currentNode;
        const isR    = routeSet.has(n.id);
        const isDim  = route.length > 0 && !isR;
        const r      = isCurr ? 20 : 13;
        const fill   = isCurr ? "#ef4444" : isR ? "#1d4ed8" : nodeColor(n.id, n.name);
        const label  = !isNaN(n.id) ? `A${n.id}` : n.id === "entrance" ? "ENT" : n.id === "exit" ? "EXIT" : n.id === "checkout" ? "CHK" : n.id.slice(0,4).toUpperCase();
        return (
          <g key={n.id} onMouseEnter={() => setHovered(n.id)} onMouseLeave={() => setHovered(null)}>
            {isCurr && <circle cx={cx} cy={cy} r={r+10} fill="none" stroke="#ef4444" strokeWidth={2} opacity={0.25}/>}
            <circle cx={cx} cy={cy} r={r} fill={fill} stroke={fill} strokeWidth={1.5} opacity={isDim ? 0.25 : 1}/>
            <text x={cx} y={cy+1} textAnchor="middle" dominantBaseline="middle" fontSize="10" fontWeight="800"
              fill="#fff" opacity={isDim ? 0.35 : 1} style={{pointerEvents:"none",userSelect:"none"}}>{label}</text>
            {hovered === n.id && (
              <g>
                <rect x={cx-60} y={cy-r-28} width={120} height={20} rx={4} fill="#1e293b" opacity={0.92}/>
                <text x={cx} y={cy-r-14} textAnchor="middle" fontSize="10" fill="#fff" style={{pointerEvents:"none",userSelect:"none"}}>
                  {n.name.replace(/^Aisle \d+ - /,"")}
                </text>
              </g>
            )}
          </g>
        );
      })}
    </svg>
  );
}

// ── STEP 0: Chatbot ────────────────────────────────────────────────────────────
function ChatStep({ onDone, voiceMode, onVoiceMode }) {
  const [input, setInput]         = useState("");
  const [loading, setLoading]     = useState(false);
  const [result, setResult]       = useState(null);
  const [listening, setListening] = useState(false);
  const [listenError, setListenError] = useState("");
  const [status, setStatus]       = useState("");
  const recognitionRef            = useRef(null);
  const inputRef                  = useRef("");  // mirror for async callbacks

  useEffect(() => { inputRef.current = input; }, [input]);

  function speak(text, onEnd) {
    if (!window.speechSynthesis) { onEnd?.(); return; }
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.lang = "en-US"; u.rate = 1.05; u.pitch = 1.0;
    if (onEnd) u.onend = onEnd;
    window.speechSynthesis.speak(u);
  }

  function startListen() {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) { setListenError("Use Chrome for voice input."); return; }
    if (recognitionRef.current) { recognitionRef.current.stop(); return; }
    const r = new SR();
    r.lang = "en-US"; r.interimResults = true; r.continuous = false;
    r.onresult = e => {
      const t = Array.from(e.results).map(x => x[0].transcript).join("");
      setInput(t); inputRef.current = t; setListenError("");
    };
    r.onerror = e => {
      setListenError({ "not-allowed": "Mic denied.", "network": "Use Chrome.", "no-speech": "No speech detected." }[e.error] || `Error: ${e.error}`);
      setListening(false); recognitionRef.current = null;
    };
    r.onend = () => {
      setListening(false); recognitionRef.current = null;
      if (inputRef.current.trim()) extractRef.current();  // auto-submit on mic stop
    };
    recognitionRef.current = r; r.start(); setListening(true); setListenError("");
    setStatus("Listening..."); onVoiceMode();
  }

  async function extract() {
    const query = inputRef.current || input;
    if (!query.trim()) return;
    setLoading(true); setResult(null); setStatus("Finding ingredients...");
    try {
      const res  = await fetch(`${API}/extract`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text: query }) });
      const data = await res.json();
      setResult(data);
      if (data.ingredients?.length) {
        const ings = data.ingredients;
        if (voiceMode) {
          const msg = (data.intro ? data.intro + "  " : "") +
            `I found ${ings.length} ingredients: ${ings.join(", ")}. Say yes to build your route.`;
          setStatus("Say \"yes\" to build your route...");
          speak(msg, () => {
            const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
            if (!SR) { onDone(ings); return; }
            const r = new SR();
            r.lang = "en-US"; r.interimResults = false; r.continuous = false;
            r.onresult = e => {
              const said = e.results[0][0].transcript.toLowerCase();
              if (/yes|yeah|yep|ok|okay|sure|go|start|build|navigate/.test(said)) onDone(ings);
              else { setStatus("Say yes to continue, or type a new recipe."); setResult(null); }
            };
            r.onerror = () => onDone(ings);
            r.onend = () => setListening(false);
            r.start(); setListening(true);
          });
        }
        // if not voice mode, just show result + button — no auto-advance
      }
    } catch { setResult({ error: "Could not reach backend." }); setStatus(""); }
    setLoading(false);
  }

  const extractRef = useRef(extract);
  useEffect(() => { extractRef.current = extract; });

  return (
    <div style={{ maxWidth: 560, margin: "0 auto", padding: "48px 24px" }}>
      {/* Logo / greeting */}
      <div style={{ textAlign: "center", marginBottom: 32 }}>
        <div style={{ width: 56, height: 56, borderRadius: "50%", background: "#eff6ff", border: "2px solid #bfdbfe", display: "flex", alignItems: "center", justifyContent: "center", margin: "0 auto 12px", fontSize: 24 }}>🛒</div>
        <h2 style={{ color: "#1e293b", fontSize: 22, fontWeight: 700, margin: "0 0 6px" }}>What would you like to make?</h2>
        {status && <p style={{ color: "#2563eb", fontSize: 14, margin: 0, fontWeight: 500 }}>{status}</p>}
      </div>

      {/* Input */}
      <div style={{ display: "flex", gap: 8, marginBottom: 8 }}>
        <input value={input} onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === "Enter" && extract()}
          placeholder="e.g. carbonara for 4, vegan dinner..."
          style={{ flex: 1, padding: "13px 16px", borderRadius: 12, border: "1px solid #e2e6ea", background: "#f8fafc", color: "#1e293b", fontSize: 14, outline: "none", fontFamily: "inherit" }}/>
        <button onClick={startListen} title={listening ? "Stop" : "Speak"} style={{
          padding: "0 16px", borderRadius: 12, border: `1px solid ${listening ? "#2563eb" : "#e2e6ea"}`,
          background: listening ? "#eff6ff" : "#f8fafc", color: listening ? "#2563eb" : "#94a3b8", cursor: "pointer", fontSize: 20,
        }}>{listening ? "⏹" : "🎤"}</button>
      </div>
      {listening   && <p style={{ color: "#2563eb", fontSize: 12, margin: "0 0 8px" }}>🔴 Listening — speak now</p>}
      {listenError && <p style={{ color: "#dc2626", fontSize: 12, margin: "0 0 8px" }}>{listenError}</p>}

      <button onClick={extract} disabled={loading || !input.trim()} style={{
        width: "100%", padding: "13px 0", borderRadius: 12, border: "none",
        background: loading || !input.trim() ? "#e2e6ea" : "#2563eb",
        color: loading || !input.trim() ? "#94a3b8" : "#fff",
        fontSize: 15, fontWeight: 600, cursor: loading || !input.trim() ? "not-allowed" : "pointer", fontFamily: "inherit",
      }}>{loading ? "Finding ingredients..." : "Find Ingredients"}</button>

      {/* Result */}
      {result && !result.error && (
        <div style={{ marginTop: 24 }}>
          {result.intro && (
            <div style={{ padding: "14px 16px", background: "#f0fdf4", border: "1px solid #bbf7d0", borderRadius: 12, marginBottom: 16 }}>
              <p style={{ color: "#166534", fontSize: 14, margin: 0, fontStyle: "italic" }}>"{result.intro}"</p>
            </div>
          )}
          <div style={{ padding: 16, background: "#fff", border: "1px solid #e2e6ea", borderRadius: 12 }}>
            <p style={{ color: "#94a3b8", fontSize: 11, fontWeight: 700, textTransform: "uppercase", letterSpacing: 0.8, margin: "0 0 12px" }}>
              {result.ingredients.length} ingredients found
            </p>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginBottom: voiceMode ? 0 : 20 }}>
              {result.ingredients.map((ing, i) => (
                <span key={i} style={{ padding: "5px 14px", background: "#eff6ff", color: "#2563eb", borderRadius: 20, fontSize: 13, fontWeight: 500, border: "1px solid #bfdbfe" }}>{ing}</span>
              ))}
            </div>
            {!voiceMode && (
              <button onClick={() => onDone(result.ingredients)} style={{
                width: "100%", marginTop: 16, padding: "13px 0", borderRadius: 12, border: "none",
                background: "#2563eb", color: "#fff", fontSize: 15, fontWeight: 600, cursor: "pointer", fontFamily: "inherit",
              }}>Get Navigation Route →</button>
            )}
          </div>
        </div>
      )}
      {result?.error && (
        <div style={{ marginTop: 16, padding: 14, background: "#fef2f2", border: "1px solid #fecaca", borderRadius: 12 }}>
          <p style={{ color: "#dc2626", fontSize: 13, margin: 0 }}>Error: {result.error}</p>
        </div>
      )}
    </div>
  );
}

// ── STEP 1: Route preview + confirm ───────────────────────────────────────────
function RoutePreview({ ingredients, mapData, onStart, voiceMode }) {
  const [result, setResult]     = useState(null);
  const [loading, setLoading]   = useState(true);
  const [error, setError]       = useState(null);
  const [listening, setListening] = useState(false);
  const resultRef               = useRef(null);

  useEffect(() => { loadRoute(); }, []);

  async function loadRoute() {
    setLoading(true); setError(null);
    try {
      const res  = await fetch(`${API}/navigate`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ items: ingredients }) });
      if (!res.ok) throw new Error(`Server error ${res.status}`);
      const data = await res.json();
      setResult(data); resultRef.current = data;
      const suffix = voiceMode ? " Tap the mic and say yes to start navigation." : "";
      const msg = data.total_price != null
        ? `Your route is ready. Estimated total: $${data.total_price.toFixed(2)} for ${ingredients.length} ingredients.${suffix}`
        : `Your route is ready for ${ingredients.length} ingredients.${suffix}`;
      window.speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance(msg);
      u.lang = "en-US"; u.rate = 1.05;
      window.speechSynthesis.speak(u);
    } catch (e) { setError(e.message); }
    setLoading(false);
  }

  function startConfirmListen() {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR || !resultRef.current) return;
    window.speechSynthesis.cancel();
    const r = new SR();
    r.lang = "en-US"; r.interimResults = false; r.continuous = false;
    r.onresult = e => {
      const said = e.results[0][0].transcript.toLowerCase();
      if (/yes|yeah|yep|ok|okay|sure|go|start|navigate/.test(said)) {
        onStart(resultRef.current);
      } else {
        const u = new SpeechSynthesisUtterance("Say yes to start navigation.");
        u.lang = "en-US"; window.speechSynthesis.speak(u);
      }
    };
    r.onerror = () => setListening(false);
    r.onend   = () => setListening(false);
    r.start(); setListening(true);
  }

  if (loading) return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "60vh" }}>
      <p style={{ color: "#94a3b8", fontSize: 14 }}>Finding best route...</p>
    </div>
  );

  if (error) return (
    <div style={{ maxWidth: 400, margin: "60px auto", padding: 24, background: "#fef2f2", border: "1px solid #fecaca", borderRadius: 12 }}>
      <p style={{ color: "#dc2626", fontSize: 14, margin: "0 0 12px" }}>Failed: {error}</p>
      <button onClick={loadRoute} style={{ padding: "8px 20px", borderRadius: 8, border: "none", background: "#dc2626", color: "#fff", cursor: "pointer", fontFamily: "inherit", fontWeight: 600 }}>Retry</button>
    </div>
  );

  const dirs = result?.directions ?? [];
  return (
    <div style={{ display: "flex", height: "calc(100vh - 57px)" }}>
      {/* Sidebar */}
      <div style={{ width: 320, borderRight: "1px solid #e2e6ea", background: "#fff", display: "flex", flexDirection: "column", flexShrink: 0 }}>
        <div style={{ padding: "20px", borderBottom: "1px solid #f1f5f9" }}>
          <p style={{ color: "#94a3b8", fontSize: 11, fontWeight: 700, textTransform: "uppercase", letterSpacing: 0.8, margin: "0 0 10px" }}>Your List</p>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 12 }}>
            {ingredients.map((ing, i) => (
              <span key={i} style={{ padding: "3px 10px", background: "#eff6ff", color: "#2563eb", borderRadius: 20, fontSize: 12, border: "1px solid #bfdbfe" }}>{ing}</span>
            ))}
          </div>
          {result?.not_found?.length > 0 && (
            <p style={{ color: "#dc2626", fontSize: 12, margin: 0 }}>Not found: {result.not_found.join(", ")}</p>
          )}
        </div>

        <div style={{ flex: 1, overflowY: "auto", padding: "8px 0" }}>
          {dirs.map((d, i) => (
            <div key={i} style={{ margin: "0 12px 4px", padding: "10px 12px", borderRadius: 10, background: "#fafbfc", border: "1px solid #e2e8f0" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                <span style={{ width: 20, height: 20, borderRadius: "50%", fontSize: 10, fontWeight: 700, background: "#e2e8f0", color: "#94a3b8", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>{d.step}</span>
                <span style={{ fontWeight: 600, fontSize: 13, color: "#334155" }}>{d.name}</span>
                <span style={{ marginLeft: "auto", fontSize: 16 }}>{d.dir_arrow}</span>
              </div>
              {d.items.map((item, j) => {
                const priceEntry = d.prices?.find(([name]) => item.toLowerCase().includes(name.toLowerCase()));
                return (
                  <div key={j} style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "#64748b", paddingLeft: 28 }}>
                    <span>· {item}</span>
                    {priceEntry && <span style={{ color: "#16a34a", fontWeight: 600 }}>${priceEntry[1].toFixed(2)}</span>}
                  </div>
                );
              })}
            </div>
          ))}
        </div>

        {result?.total_price != null && (
          <div style={{ padding: "10px 16px", background: "#f0fdf4", borderTop: "1px solid #bbf7d0" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span style={{ fontSize: 12, fontWeight: 700, color: "#166534" }}>Estimated Total</span>
              <span style={{ fontSize: 16, fontWeight: 800, color: "#16a34a" }}>${result.total_price.toFixed(2)}</span>
            </div>
          </div>
        )}

        {voiceMode ? (
          <div style={{ padding: "14px 16px", borderTop: "1px solid #e2e6ea", background: "#f0fdf4", display: "flex", alignItems: "center", gap: 10 }}>
            <button onClick={startConfirmListen} style={{
              width: 44, height: 44, borderRadius: "50%", border: `2px solid ${listening ? "#2563eb" : "#bbf7d0"}`,
              background: listening ? "#eff6ff" : "#fff", fontSize: 20, cursor: "pointer", flexShrink: 0,
            }}>{listening ? "⏹" : "🎤"}</button>
            <p style={{ fontSize: 12, color: "#166534", fontWeight: 600, margin: 0 }}>
              {listening ? "🔴 Listening — say yes" : "Tap mic and say \"yes\" to start"}
            </p>
          </div>
        ) : (
          <div style={{ padding: "16px" }}>
            <button onClick={() => result && onStart(result)} style={{
              width: "100%", padding: "13px 0", borderRadius: 10, border: "none",
              background: "#2563eb", color: "#fff", fontSize: 14, fontWeight: 600, cursor: "pointer", fontFamily: "inherit",
            }}>Start Navigation →</button>
          </div>
        )}
      </div>

      {/* Map */}
      <div style={{ flex: 1, padding: 24, background: "#f1f4f8" }}>
        <p style={{ color: "#64748b", fontSize: 12, fontWeight: 600, margin: "0 0 12px", textTransform: "uppercase", letterSpacing: 0.8 }}>Your Route</p>
        <RouteMap nodes={mapData.nodes||[]} edges={mapData.edges||[]} route={result?.route||[]} currentNode="entrance" />
      </div>
    </div>
  );
}

// ── STEP 2: Live navigation driven by ocr_agent.py ────────────────────────────
function LiveNav({ navResult, mapData }) {
  const [currentStep, setCurrentStep] = useState(0);
  const [detected, setDetected]       = useState(null);
  const [scanResult, setScanResult]   = useState(null);
  const [scanning, setScanning]       = useState(false);

  const dirs  = navResult?.directions ?? [];
  const route = navResult?.route      ?? [];
  const dir   = dirs[currentStep]     ?? null;
  const currentNode = route[Math.min(currentStep, route.length - 1)] ?? "entrance";

  // Speak only the first step when live nav loads
  useEffect(() => {
    if (!dir) return;
    const items = dir.items?.join(", ") ?? "";
    const msg = `Step ${dir.step} of ${dir.total}. ${dir.dir_label} to ${dir.name}. Grab: ${items}.`;
    if (window.speechSynthesis) {
      window.speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance(msg);
      u.lang = "en-US"; u.rate = 1.05;
      window.speechSynthesis.speak(u);
    }
  }, []);  // only on mount


  const done = currentStep >= dirs.length;

  useEffect(() => {
    if (done && window.speechSynthesis) {
      window.speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance("All items collected! Head to checkout. You are done. Great job!");
      u.lang = "en-US"; u.rate = 1.0;
      window.speechSynthesis.speak(u);
    }
  }, [done]);

  async function scanShelf() {
    if (!dir || scanning) return;
    const itemName = dir.items?.[0]?.split(" ")[0] ?? "product";
    setScanning(true); setScanResult(null);
    try {
      const res  = await fetch(`${API}/scan/demo?product=${encodeURIComponent(itemName)}`);
      const data = await res.json();
      setScanResult(data);
      window.speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance(data.spoken);
      u.lang = "en-US"; u.rate = 1.05;
      window.speechSynthesis.speak(u);
    } catch { setScanResult({ found: false, spoken: "Scan failed." }); }
    setScanning(false);
  }

  if (done) return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "calc(100vh - 57px)", background: "#f1f4f8" }}>
      <div style={{ textAlign: "center", padding: 40 }}>
        <div style={{ fontSize: 56, marginBottom: 16 }}>🎉</div>
        <h2 style={{ color: "#166534", fontSize: 24, fontWeight: 700, margin: "0 0 8px" }}>All items collected!</h2>
        <p style={{ color: "#64748b", fontSize: 15, margin: 0 }}>Head to checkout and you're done.</p>
      </div>
    </div>
  );

  return (
    <div style={{ display: "flex", height: "calc(100vh - 57px)" }}>

      {/* Left — instructions + step list */}
      <div style={{ width: 320, borderRight: "1px solid #e2e6ea", background: "#fff", display: "flex", flexDirection: "column", flexShrink: 0 }}>

        {/* Current step */}
        {dir && (
          <div style={{ padding: "20px", background: "#eff6ff", borderBottom: "1px solid #bfdbfe" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 8 }}>
              <span style={{ fontSize: 11, fontWeight: 700, color: "#2563eb", textTransform: "uppercase", letterSpacing: 0.8 }}>Step {dir.step} of {dir.total}</span>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 8 }}>
              <span style={{ fontSize: 28 }}>{dir.dir_arrow}</span>
              <div>
                <p style={{ fontSize: 15, fontWeight: 700, color: "#1e40af", margin: "0 0 2px" }}>{dir.dir_label}</p>
                <p style={{ fontSize: 13, color: "#3b82f6", margin: 0 }}>→ {dir.name}</p>
              </div>
            </div>
            {dir.walk?.length > 0 && (
              <p style={{ fontSize: 11, color: "#64748b", margin: 0 }}>
                Walk: {dir.walk.map(w => w.replace(/^Aisle \d+ - /,"")).join(" → ")}
              </p>
            )}
            <div style={{ marginTop: 10 }}>
              {dir.items.map((item, j) => (
                <div key={j} style={{ fontSize: 12, color: "#1e40af", padding: "3px 0" }}>📦 {item}</div>
              ))}
            </div>

            {/* Demo controls */}
            <div style={{ display: "flex", gap: 8, marginTop: 14 }}>
              <button onClick={scanShelf} disabled={scanning} style={{
                flex: 1, padding: "9px 0", borderRadius: 8, border: "none",
                background: scanning ? "#e2e8f0" : "#7c3aed", color: scanning ? "#94a3b8" : "#fff",
                fontSize: 12, fontWeight: 700, cursor: scanning ? "not-allowed" : "pointer", fontFamily: "inherit",
              }}>{scanning ? "Scanning..." : "📷 Scan Shelf (YOLO)"}</button>
              <button onClick={() => setCurrentStep(s => Math.min(s + 1, dirs.length))} style={{
                flex: 1, padding: "9px 0", borderRadius: 8, border: "none",
                background: "#0f172a", color: "#fff",
                fontSize: 12, fontWeight: 700, cursor: "pointer", fontFamily: "inherit",
              }}>Next Step →</button>
            </div>
          </div>
        )}

        {/* YOLO scan result */}
        {scanResult && (
          <div style={{ margin: "12px", padding: "12px 14px", borderRadius: 10, background: scanResult.found ? "#faf5ff" : "#fef2f2", border: `1px solid ${scanResult.found ? "#c4b5fd" : "#fecaca"}` }}>
            <p style={{ fontSize: 11, fontWeight: 700, color: scanResult.found ? "#7c3aed" : "#dc2626", textTransform: "uppercase", letterSpacing: 0.8, margin: "0 0 4px" }}>YOLO Shelf Scan</p>
            <p style={{ fontSize: 14, fontWeight: 700, color: scanResult.found ? "#6d28d9" : "#b91c1c", margin: "0 0 2px" }}>{scanResult.spoken}</p>
            {scanResult.found && <p style={{ fontSize: 11, color: "#8b5cf6", margin: 0 }}>Confidence: {Math.round((scanResult.confidence ?? 0) * 100)}%</p>}
          </div>
        )}

        {/* Scan status — only shown when ocr_agent detects something */}
        {detected && (
          <div style={{ margin: "12px", padding: "12px 14px", borderRadius: 10, background: "#f0fdf4", border: "1px solid #86efac" }}>
            <p style={{ fontSize: 11, fontWeight: 700, color: "#166534", textTransform: "uppercase", letterSpacing: 0.8, margin: "0 0 4px" }}>Aisle Detected</p>
            <p style={{ fontSize: 15, fontWeight: 700, color: "#15803d", margin: "0 0 2px" }}>{detected.code} — {detected.name}</p>
            <p style={{ fontSize: 11, color: "#4ade80", margin: 0 }}>
              {detected.node_id === dir?.target ? "✓ Correct aisle! Moving to next step..." : `Confidence: ${Math.round((detected.confidence ?? 0) * 100)}%`}
            </p>
          </div>
        )}

        {/* All steps list */}
        <div style={{ flex: 1, overflowY: "auto", padding: "8px 0" }}>
          {dirs.map((d, i) => (
            <div key={i} style={{ margin: "0 12px 3px", padding: "8px 12px", borderRadius: 8, background: i === currentStep ? "#eff6ff" : i < currentStep ? "#f0fdf4" : "#fafbfc", border: `1px solid ${i === currentStep ? "#bfdbfe" : i < currentStep ? "#bbf7d0" : "#e2e8f0"}` }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span style={{ fontSize: 13 }}>{i < currentStep ? "✓" : i === currentStep ? "▶" : "○"}</span>
                <span style={{ fontSize: 12, fontWeight: 600, color: i < currentStep ? "#166534" : i === currentStep ? "#1e40af" : "#94a3b8" }}>{d.name}</span>
                <span style={{ marginLeft: "auto", fontSize: 14 }}>{d.dir_arrow}</span>
              </div>
            </div>
          ))}
        </div>

        {/* Agent status */}
        <div style={{ padding: "12px 16px", borderTop: "1px solid #e2e6ea", background: "#f8fafc" }}>
          <p style={{ fontSize: 11, color: "#94a3b8", margin: 0 }}>
            Run <code style={{ background: "#e2e8f0", padding: "1px 5px", borderRadius: 4 }}>python agents/ocr_agent.py</code> in terminal
          </p>
        </div>
      </div>

      {/* Right — map + live camera feed */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", background: "#f1f4f8" }}>

        {/* Map */}
        <div style={{ flex: 1, padding: "16px 16px 8px" }}>
          <p style={{ color: "#64748b", fontSize: 12, fontWeight: 600, margin: "0 0 8px", textTransform: "uppercase", letterSpacing: 0.8 }}>
            Live Route — scanning for <strong style={{ color: "#2563eb" }}>
              {dir ? (!isNaN(dir.target) ? `A${dir.target}` : dir.target.toUpperCase()) : "—"}
            </strong>
          </p>
          <RouteMap nodes={mapData.nodes||[]} edges={mapData.edges||[]} route={route} currentNode={currentNode} />
        </div>

        {/* MJPEG stream from ocr_agent.py */}
        <div style={{ height: 240, margin: "0 16px 16px", borderRadius: 12, overflow: "hidden", background: "#0f172a", position: "relative", flexShrink: 0 }}>
          <img
            src="http://localhost:8004/video_feed"
            style={{ width: "100%", height: "100%", objectFit: "cover", display: "block" }}
            onLoad={e  => { e.target.style.opacity = 1; }}
            onError={e => { e.target.style.opacity = 0; }}
          />
          {/* Waiting overlay — shown until stream loads */}
          <div style={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", pointerEvents: "none" }}>
            <div style={{ width: 10, height: 10, borderRadius: "50%", background: "#ef4444", marginBottom: 10, animation: "pulse 1.5s infinite" }}/>
            <span style={{ color: "#475569", fontSize: 13 }}>Waiting for <code style={{ background: "#1e293b", padding: "1px 6px", borderRadius: 4, color: "#94a3b8" }}>ocr_agent.py</code></span>
          </div>
          {/* Overlay label */}
          <div style={{ position: "absolute", top: 10, left: 12, background: "rgba(0,0,0,0.55)", padding: "3px 10px", borderRadius: 6 }}>
            <span style={{ color: "#94a3b8", fontSize: 11, fontWeight: 600 }}>OCR Agent — Live</span>
          </div>
          {/* Green border when detected */}
          {detected && (
            <div style={{ position: "absolute", inset: 0, border: "3px solid #4ade80", borderRadius: 12, pointerEvents: "none" }}/>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Root ───────────────────────────────────────────────────────────────────────
export default function Pipeline() {
  const [step, setStep]               = useState(0);
  const [ingredients, setIngredients] = useState([]);
  const [navResult, setNavResult]     = useState(null);
  const [mapData, setMapData]         = useState({ nodes: [], edges: [] });
  const [voiceMode, setVoiceMode]     = useState(false); // true once mic is first used

  useEffect(() => {
    fetch(`${API}/map`).then(r => r.json()).then(setMapData).catch(() => {});
  }, []);

  return (
    <div style={{ minHeight: "calc(100vh - 57px)", background: "#f1f4f8", fontFamily: "'Inter', sans-serif" }}>
      {step === 0 && (
        <ChatStep
          voiceMode={voiceMode}
          onVoiceMode={() => setVoiceMode(true)}
          onDone={ings => { setIngredients(ings); setStep(1); }}
        />
      )}
      {step === 1 && (
        <RoutePreview
          ingredients={ingredients}
          mapData={mapData}
          voiceMode={voiceMode}
          onStart={result => { setNavResult(result); setStep(2); }}
        />
      )}
      {step === 2 && (
        <LiveNav navResult={navResult} mapData={mapData} />
      )}
    </div>
  );
}
