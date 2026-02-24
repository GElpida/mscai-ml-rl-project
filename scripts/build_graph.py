from __future__ import annotations

import argparse
import sys
sys.dont_write_bytecode = True
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import networkx as nx

from game.coordination_repeated_game import CoordinationGame, make_default_game_7


def _labels_from_types(agent_types) -> Dict[int, str]:
    counters = {"X": 1, "Y": 1}
    labels: Dict[int, str] = {}
    for agent_id, agent_type in enumerate(agent_types):
        labels[agent_id] = f"{agent_type}{counters[agent_type]}"
        counters[agent_type] += 1
    return labels


def build_graph(game: CoordinationGame) -> nx.Graph:
    g = nx.Graph()
    labels = _labels_from_types(game.agent_types)

    for agent_id in range(game.n_agents):
        g.add_node(
            agent_id,
            agent_type=game.agent_types[agent_id],
            label=labels[agent_id],
        )

    for i in range(game.n_agents):
        for j in game.adjacency[i]:
            g.add_edge(i, j)
    return g


def default_positions(labels: Dict[int, str]) -> Dict[int, Tuple[float, float]]:
    # Roughly matches the assignment diagram.
    by_name = {name: agent_id for agent_id, name in labels.items()}
    needed = {"X1", "X2", "Y1", "Y2", "X3", "X4", "X5"}
    if not needed.issubset(by_name.keys()):
        return {}

    return {
        by_name["X1"]: (-1.0, 1.0),
        by_name["Y1"]: (-1.0, 0.0),
        by_name["X4"]: (-1.0, -1.0),
        by_name["X3"]: (0.0, 0.0),
        by_name["X2"]: (1.0, 1.0),
        by_name["Y2"]: (1.0, 0.0),
        by_name["X5"]: (1.0, -1.0),
    }


def draw_graph(g: nx.Graph, out_path: str | None, show: bool) -> None:
    labels = {n: g.nodes[n]["label"] for n in g.nodes}
    pos = default_positions(labels) or nx.spring_layout(g, seed=0)

    colors = [
        ("#2ca02c" if g.nodes[n]["agent_type"] == "X" else "#d6279a") for n in g.nodes
    ]

    plt.figure(figsize=(6, 3))
    nx.draw_networkx_edges(g, pos, width=2.0, edge_color="#777777")
    nx.draw_networkx_nodes(g, pos, node_color=colors, node_size=550, linewidths=1.5, edgecolors="black")
    nx.draw_networkx_labels(g, pos, labels=labels, font_size=10, font_color="black")
    plt.axis("off")

    if out_path:
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
    if show:
        plt.show()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build and visualize the 7-agent coordination game graph."
    )
    parser.add_argument("--out", help="Save plot to this path (e.g. graph.png).")
    parser.add_argument("--show", action="store_true", help="Show the plot window.")
    args = parser.parse_args()

    game = make_default_game_7()
    g = build_graph(game)

    labels = {n: g.nodes[n]["label"] for n in g.nodes}
    edges = sorted(tuple(sorted((labels[u], labels[v]))) for u, v in g.edges)
    print("Nodes:", {labels[n]: g.nodes[n]["agent_type"] for n in sorted(g.nodes)})
    print("Edges:", edges)

    if args.out or args.show:
        draw_graph(g, args.out, args.show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
