// Client-side navigation for demo / offline mode.
// Runs Dijkstra + nearest-neighbor on the pre-loaded store graph.

import { mockMapData } from "./mockData";

// Keyword → node ID  (case-insensitive partial match used as fallback)
const KEYWORD_TO_NODE = {
  // Dairy / refrigerated
  milk: "100", eggs: "100", butter: "1", cheese: "1",
  cream: "100", "sour cream": "34", "cream cheese": "34",
  // Bakery
  bread: "1", bagels: "152", sourdough: "152", donuts: "152",
  croissants: "152", muffins: "152",
  // Dry goods
  pasta: "2", rice: "2", quinoa: "2", couscous: "2", noodles: "2",
  // Baking & coffee
  flour: "3", sugar: "3", coffee: "3", tea: "3", soup: "3",
  // Breakfast
  oatmeal: "4", cereal: "4", crackers: "4", pretzels: "4",
  // Snacks
  chips: "5", "sports drinks": "5",
  popcorn: "9", nuts: "9",
  // Beverages
  water: "6", soda: "6", "sparkling water": "6",
  // Personal care
  toothpaste: "7", soap: "7", deodorant: "7", shampoo: "7",
  // Frozen
  "frozen pizza": "8", "ice cream": "8", pizza: "8",
  "frozen meals": "40", "pot pies": "40",
  "frozen yogurt": "42", gelato: "42",
  // Canned goods
  tuna: "11", corn: "11", broth: "11",
  beans: "18", "tomato sauce": "18", "pasta sauce": "18", mustard: "18",
  // Oils & condiments
  "olive oil": "12", ketchup: "12", vinegar: "12",
  // International
  tortillas: "16", "soy sauce": "16", "taco shells": "16",
  // Granola & syrup
  "maple syrup": "22", honey: "22", syrup: "22", granola: "22",
  // Yogurt
  yogurt: "34", kefir: "34",
  // Produce
  lettuce: "105", spinach: "105", broccoli: "105",
  apples: "351", bananas: "351", grapes: "351", strawberries: "351",
  tomatoes: "352", onions: "352", potatoes: "352",
  // Meat & deli
  chicken: "101", beef: "101", pork: "101", sausage: "101", turkey: "101",
  salsa: "447", hummus: "447", guacamole: "447", "deli meat": "447",
  // Wall sections
  vitamins: "vitamins", supplements: "vitamins",
  medicine: "pharmacy", allergy: "pharmacy",
  "dish soap": "cleaning", detergent: "cleaning", "paper towels": "cleaning",
};

function buildGraph(nodes, edges) {
  const g = {};
  for (const n of nodes) g[n.id] = [];
  for (const e of edges) {
    if (g[e.from]) g[e.from].push(e.to);
    if (g[e.to])   g[e.to].push(e.from);   // treat as undirected
  }
  return g;
}

// Returns { dist, prev } via BFS (all edge weights = 1)
function bfs(graph, source) {
  const dist = {}, prev = {};
  for (const n of Object.keys(graph)) { dist[n] = Infinity; prev[n] = null; }
  dist[source] = 0;
  const queue = [source];
  while (queue.length) {
    const cur = queue.shift();
    for (const nb of (graph[cur] || [])) {
      if (dist[nb] === Infinity) {
        dist[nb] = dist[cur] + 1;
        prev[nb] = cur;
        queue.push(nb);
      }
    }
  }
  return { dist, prev };
}

function getPath(prev, target) {
  const path = [];
  for (let cur = target; cur !== null; cur = prev[cur]) path.unshift(cur);
  return path;
}

function findNodeForItem(item) {
  const lower = item.toLowerCase().trim();
  if (KEYWORD_TO_NODE[lower]) return KEYWORD_TO_NODE[lower];
  for (const [kw, nid] of Object.entries(KEYWORD_TO_NODE)) {
    if (lower.includes(kw) || kw.includes(lower)) return nid;
  }
  return null;
}

export function demoNavigate(itemList) {
  const { nodes, edges } = mockMapData;
  const graph  = buildGraph(nodes, edges);
  const nodeMap = Object.fromEntries(nodes.map(n => [n.id, n]));

  // Map items → nodes
  const stopMap = {}, notFound = [];
  for (const item of itemList) {
    const nid = findNodeForItem(item);
    if (nid) {
      if (!stopMap[nid]) stopMap[nid] = { nodeId: nid, items: [] };
      stopMap[nid].items.push(item.charAt(0).toUpperCase() + item.slice(1));
    } else {
      notFound.push(item);
    }
  }

  const uniqueStops = Object.values(stopMap);
  if (uniqueStops.length === 0) {
    return { route: [], directions: [], not_found: notFound, store: "Kroger – Demo Store" };
  }

  // Nearest-neighbor ordering starting from entrance
  const START = "entrance_left";
  const END   = "checkout_1";
  const ordered = [];
  let current = START;
  const remaining = [...uniqueStops];

  while (remaining.length) {
    const { dist } = bfs(graph, current);
    remaining.sort((a, b) => (dist[a.nodeId] ?? Infinity) - (dist[b.nodeId] ?? Infinity));
    const next = remaining.shift();
    ordered.push(next);
    current = next.nodeId;
  }

  // Chain BFS paths into a full route
  let fullRoute = [START];
  current = START;
  for (const stop of ordered) {
    const { prev } = bfs(graph, current);
    const path = getPath(prev, stop.nodeId);
    fullRoute = [...fullRoute, ...path.slice(1)];
    current = stop.nodeId;
  }
  const { prev: prevEnd } = bfs(graph, current);
  const toCheckout = getPath(prevEnd, END);
  fullRoute = [...fullRoute, ...toCheckout.slice(1)];

  // Build directions (one per stop + checkout)
  const directions = ordered.map((stop, i) => {
    const node = nodeMap[stop.nodeId];
    return {
      step: i + 1,
      total: ordered.length + 1,
      target: stop.nodeId,
      name: node?.name ?? stop.nodeId,
      dir_label: `Stop ${i + 1} of ${ordered.length}`,
      direction: "ST",
      audio: node?.name ? `Heading to ${node.name}.` : "",
      items: stop.items,
      walk: [],
    };
  });
  directions.push({
    step: ordered.length + 1,
    total: ordered.length + 1,
    target: END,
    name: nodeMap[END]?.name ?? "Checkout",
    dir_label: "Final stop",
    direction: "ST",
    audio: "Proceed to checkout. You're all done!",
    items: [],
    walk: [],
  });

  return { route: fullRoute, directions, not_found: notFound, store: "Kroger – Demo Store" };
}
