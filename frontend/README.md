# WayfinderAI — React Frontend

The web interface for WayfinderAI. Two views: an interactive store map and a step-by-step navigation assistant.

## What's inside

### Navigate tab (`/`)
- Enter a shopping list (e.g. `milk, pasta, chips`) or pick from example chips
- Optimal route calculated via Dijkstra + Nearest-Neighbor comparison
- Step-by-step directions with turn indicators (↑ ↰ ↱)
- Animated "you are here" dot moves along the route on the map
- Per-step 🔊 voice readout via Web Speech API
- Progress bar + route summary (stops, waypoints, estimated time)

### Store Map tab
- Full interactive SVG map of the store
- Color-coded by department (dairy=blue, produce=green, frozen=cyan, etc.)
- Click any aisle to see its products, shelf positions, and audio description
- Upload a custom store CSV or auto-load live Kroger data

### Full Pipeline (`localhost:3001`)
End-to-end demo: Agent 0 → Agent 2 → Agent 3 → Agent 4 in a single browser flow.

## Setup

**Requires:** Backend running at `http://localhost:8002`

```bash
# Start backend first
cd ..
python api.py

# Then start frontend
npm install
npm run dev
```

Open `http://localhost:5173`

## API endpoints used

| Endpoint | Used by |
|----------|---------|
| `GET /map` | Store map, route visualization |
| `POST /navigate` | Route calculation from shopping list |
| `POST /extract` | Natural language → ingredient list (Agent 0) |
| `POST /ocr` | Aisle sign photo → aisle code (Agent 3) |
| `POST /scan` | Shelf photo → product position (Agent 4) |
| `GET /aisle/:id` | Aisle detail panel (Store Map tab) |

## Project structure

```
frontend/
├── src/
│   ├── App.jsx          # Main app — Navigate + Store Map tabs
│   ├── Pipeline.jsx     # Full 4-agent pipeline demo
│   ├── StoreLayout.jsx  # Interactive store map with aisle detail panel
│   ├── main.jsx         # Entry point
│   └── index.css        # Global styles
├── index.html
├── package.json
└── vite.config.js
```

## Voice features

- **Navigate tab:** click 🔊 next to any direction step to hear it read aloud
- **Pipeline → Agent 0:** microphone button for voice input (Chrome required)
- **Pipeline → Agent 3:** camera capture + OCR for aisle sign confirmation
- **Pipeline → Agent 4:** camera capture + YOLO shelf scan

Built with React 18 + Vite. No external UI library.
