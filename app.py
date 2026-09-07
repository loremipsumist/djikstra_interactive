"""
Dijkstra's Algorithm — Interactive Visualizer (Streamlit)
-----------------------------------------------------------
Run locally:      streamlit run app.py
Deploy free:       push this repo to GitHub, then deploy on
                    https://share.streamlit.io (Streamlit Community Cloud)
"""

import time
import heapq

import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt

st.set_page_config(page_title="Dijkstra's Algorithm Visualizer", layout="wide")

# ----------------------------------------------------------------------
# Default graph
# ----------------------------------------------------------------------
DEFAULT_EDGES_TEXT = """A,B,4
A,C,2
B,C,1
B,D,5
C,D,8
C,E,10
D,E,2
D,F,6
E,F,3
E,G,5
F,G,1
F,H,7
G,H,4"""

DEFAULT_POS = {
    "A": (0.0, 1.0), "B": (1.3, 2.0), "C": (1.3, 0.0),
    "D": (2.8, 1.6), "E": (2.8, 0.4), "F": (4.3, 1.4),
    "G": (4.3, -0.2), "H": (5.8, 0.8),
}


def parse_edges(text):
    """Parse 'node1,node2,weight' lines into a weighted edge list."""
    edges = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Bad line (need 'node1,node2,weight'): {line!r}")
        u, v, w = parts
        w = float(w)
        w = int(w) if w.is_integer() else w
        if w < 0:
            raise ValueError(f"Negative weight not allowed: {line!r}")
        edges.append((u, v, w))
    if not edges:
        raise ValueError("No edges given.")
    return edges


def build_graph(edges):
    graph = nx.Graph()
    graph.add_weighted_edges_from(edges)
    return graph


def layout_for(graph):
    """Use the hand-placed layout if the node set matches; otherwise compute one."""
    if set(graph.nodes) == set(DEFAULT_POS.keys()):
        return DEFAULT_POS
    return nx.spring_layout(graph, seed=7)


# ----------------------------------------------------------------------
# Dijkstra step generator (same logic as the desktop version)
# ----------------------------------------------------------------------
def dijkstra_steps(graph, source, target):
    dist = {node: float("inf") for node in graph.nodes}
    prev = {node: None for node in graph.nodes}
    dist[source] = 0
    visited = set()
    heap = [(0, source)]

    yield {
        "type": "init", "dist": dict(dist), "prev": dict(prev),
        "visited": set(visited), "current": None,
        "relaxed": [], "frontier": {source},
    }

    while heap:
        d, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)

        relaxed = []
        for v in graph.neighbors(u):
            if v in visited:
                continue
            weight = graph[u][v]["weight"]
            new_dist = dist[u] + weight
            if new_dist < dist[v]:
                dist[v] = new_dist
                prev[v] = u
                heapq.heappush(heap, (new_dist, v))
                relaxed.append(v)

        frontier = {n for n in graph.nodes if n not in visited and dist[n] < float("inf")}

        yield {
            "type": "visit", "dist": dict(dist), "prev": dict(prev),
            "visited": set(visited), "current": u,
            "relaxed": relaxed, "frontier": frontier,
        }

        if u == target:
            break

    path = []
    node = target
    if dist[target] < float("inf"):
        while node is not None:
            path.append(node)
            node = prev[node]
        path.reverse()

    yield {
        "type": "done", "dist": dict(dist), "prev": dict(prev),
        "visited": set(visited), "current": None,
        "relaxed": [], "frontier": set(), "path": path,
    }


# ----------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------
def draw_step(graph, pos, step, source, target):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.set_axis_off()

    visited = step["visited"]
    frontier = step["frontier"]
    current = step["current"]
    prev = step["prev"]
    dist = step["dist"]
    path = list(step.get("path", []))
    path_set = set(path)

    tree_edges = {(prev[n], n) for n in prev if prev[n] is not None}
    path_edges = set(zip(path, path[1:]))

    def edge_style(u, v):
        if (u, v) in path_edges or (v, u) in path_edges:
            return "gold", 4.0, 1.0
        if (u, v) in tree_edges or (v, u) in tree_edges:
            return "#3B7DD8", 2.6, 0.95
        return "#B9C2CC", 1.2, 0.6

    for u, v, data in graph.edges(data=True):
        color, width, alpha = edge_style(u, v)
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=width, alpha=alpha, zorder=1)
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my, str(data["weight"]), fontsize=9, color="#4A4A4A",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85), zorder=2)

    for n in graph.nodes:
        x, y = pos[n]
        if n == current:
            face, edge, ring = "#E4572E", "#8C2E14", 3.0
        elif n in path_set and step["type"] == "done":
            face, edge, ring = "#F2C94C", "#8C6D1F", 2.6
        elif n in visited:
            face, edge, ring = "#4CAF7D", "#256B45", 2.0
        elif n in frontier:
            face, edge, ring = "#F0A75E", "#8C5A22", 2.0
        else:
            face, edge, ring = "#D9DEE3", "#8A9199", 1.4
        ax.scatter([x], [y], s=1350, facecolor=face, edgecolor=edge, linewidth=ring, zorder=3)
        label = n if dist[n] == float("inf") else f"{n}\n{dist[n]}"
        ax.text(x, y, label, fontsize=11, fontweight="bold", ha="center", va="center",
                color="#1A1A1A", zorder=4)

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    pad_x = (max(xs) - min(xs)) * 0.2 + 0.5
    pad_y = (max(ys) - min(ys)) * 0.2 + 0.5
    ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
    ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)

    fig.tight_layout()
    return fig


def status_line(step, source, target):
    if step["type"] == "init":
        return f"Start: distance to source **{source}** set to 0, all others ∞."
    if step["type"] == "visit":
        relaxed = ", ".join(step["relaxed"]) if step["relaxed"] else "none"
        return f"Visited **{step['current']}** (now finalized). Relaxed neighbors: {relaxed}."
    dist = step["dist"]
    path = step.get("path", [])
    if path:
        return f"**Done.** Shortest path {source} → {target}: " + " → ".join(path) + f"  (total = **{dist[target]}**)"
    return f"**Done.** No path exists from {source} to {target}."


# ----------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------
st.title("Dijkstra's Algorithm — Interactive Visualizer")
st.caption("Step through the algorithm one node at a time, or hit Auto Play and watch it run.")

with st.sidebar:
    st.header("Graph")
    edges_text = st.text_area(
        "Edges — one per line: `node,node,weight`",
        value=DEFAULT_EDGES_TEXT, height=260,
    )
    try:
        edges = parse_edges(edges_text)
        graph = build_graph(edges)
        graph_error = None
    except ValueError as e:
        graph_error = str(e)
        graph = None

    if graph_error:
        st.error(graph_error)
        st.stop()

    nodes = sorted(graph.nodes)
    default_source = "A" if "A" in nodes else nodes[0]
    default_target = "H" if "H" in nodes else nodes[-1]
    source = st.selectbox("Source node", nodes, index=nodes.index(default_source))
    target = st.selectbox("Target node", nodes, index=nodes.index(default_target))

    st.divider()
    play_speed = st.slider("Auto play speed (seconds/step)", 0.2, 2.0, 0.9, 0.1)

# Rebuild the step list whenever the graph or endpoints change
graph_key = (edges_text, source, target)
if st.session_state.get("graph_key") != graph_key:
    st.session_state.graph_key = graph_key
    st.session_state.steps = list(dijkstra_steps(graph, source, target))
    st.session_state.i = 0
    st.session_state.playing = False

steps = st.session_state.steps
max_i = len(steps) - 1
pos = layout_for(graph)

col_graph, col_info = st.columns([2, 1])

with col_info:
    st.subheader("Distance table")
    step = steps[st.session_state.i]
    dist = step["dist"]
    visited = step["visited"]
    current = step["current"]
    rows = []
    for n in nodes:
        d = "∞" if dist[n] == float("inf") else str(dist[n])
        state = "finalized" if n in visited else ("frontier" if dist[n] < float("inf") else "—")
        if n == current:
            state = "**visiting now**"
        rows.append({"Node": n, "Distance": d, "Status": state})
    st.table(rows)
    st.markdown(
        "**Legend**  \n"
        "🔴 visiting now &nbsp; 🟢 finalized &nbsp; 🟠 frontier &nbsp; ⚪ unvisited &nbsp; 🟡 final path"
    )

with col_graph:
    fig = draw_step(graph, pos, step, source, target)
    st.pyplot(fig, width='stretch')
    plt.close(fig)
    st.info(status_line(step, source, target))

    st.write(f"Step **{st.session_state.i} / {max_i}**")
    st.progress(0 if max_i == 0 else st.session_state.i / max_i)

    b1, b2, b3, b4 = st.columns(4)
    if b1.button("⏮ Reset", width='stretch'):
        st.session_state.i = 0
        st.session_state.playing = False
        st.rerun()
    if b2.button("◀ Prev", width='stretch', disabled=st.session_state.i == 0):
        st.session_state.i -= 1
        st.session_state.playing = False
        st.rerun()
    if b3.button("Next ▶", width='stretch', disabled=st.session_state.i >= max_i):
        st.session_state.i += 1
        st.session_state.playing = False
        st.rerun()
    play_label = "⏸ Pause" if st.session_state.playing else "▶ Auto Play"
    if b4.button(play_label, width='stretch', disabled=st.session_state.i >= max_i and not st.session_state.playing):
        st.session_state.playing = not st.session_state.playing
        st.rerun()

    st.session_state.step_slider = st.session_state.i
    new_i = st.slider("Jump to step", 0, max_i, key="step_slider")
    if new_i != st.session_state.i:
        st.session_state.i = new_i
        st.session_state.playing = False
        st.rerun()

# Auto-play: advance one step, pause, then rerun. Streamlit reruns the whole
# script on every interaction, so this creates a step-by-step animation loop.
if st.session_state.playing:
    if st.session_state.i >= max_i:
        st.session_state.playing = False
    else:
        time.sleep(play_speed)
        st.session_state.i += 1
        st.rerun()
