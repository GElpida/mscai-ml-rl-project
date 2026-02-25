from __future__ import annotations

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # allow imports from parent dir if needed

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

from agent_blind_fixed import QLearningAgent
from coordination_repeated_game import make_default_game_7

Action = int
# IMPORTANT: this main assumes your blind state is (last_action, agent_type)
# If your agent uses a different state tuple, adjust build_state() below accordingly.
State = Tuple


@dataclass
class RunConfig:
    episodes: int = 5000
    horizon: int = 40  # rounds per episode

    alpha: float = 0.1
    gamma: float = 0.95

    # ε starts at 1, drops by 0.01 every 40 episodes
    eps_start: float = 1.0
    eps_drop: float = 0.01
    eps_drop_every: int = 40

    # outputs
    out_prefix: str = "blind"
    out_dir: str = r"C:\Users\nchar\OneDrive\Desktop\MSc_AI\2.giannakopoulos\ergasia-vouros\results\Q_blind"

    # GIF
    gif_every: int = 50  # frame every N episodes


def ensure_outdir(cfg: RunConfig) -> Path:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_fig(out_dir: Path, filename: str) -> None:
    plt.tight_layout()
    plt.savefig(out_dir / filename)
    plt.close()


def epsilon_schedule(cfg: RunConfig, ep: int) -> float:
    eps = cfg.eps_start - (ep // cfg.eps_drop_every) * cfg.eps_drop
    return max(0.0, eps)


# -------------------------
# Metric helpers
# -------------------------
def realized_reward_for_agent(game, i: int, ai: Action, actions: List[Action]) -> float:
    """Reward agent i would get if it played ai, given others' realized actions."""
    r = 0.0
    for j in game.adjacency[i]:
        ri, _ = game._pair_reward(i, j, ai, actions[j])
        r += ri
    return r


def expected_reward_for_agent_action(game, i: int, ai: Action, p_action1: List[float]) -> float:
    """
    Expected reward for agent i if it deterministically plays ai,
    and each neighbor j plays action 0 with prob p_action1[j], action 1 otherwise.
    Independence across neighbors.
    """
    r = 0.0
    for j in game.adjacency[i]:
        pj0 = p_action1[j]
        pj1 = 1.0 - pj0
        ri0, _ = game._pair_reward(i, j, ai, 0)
        ri1, _ = game._pair_reward(i, j, ai, 1)
        r += pj0 * ri0 + pj1 * ri1
    return r


def expected_reward_for_agent_mixed(game, i: int, pi0: float, p_action1: List[float]) -> float:
    """Expected reward for agent i if it plays action0 with prob pi0."""
    r0 = expected_reward_for_agent_action(game, i, 0, p_action1)
    r1 = expected_reward_for_agent_action(game, i, 1, p_action1)
    return pi0 * r0 + (1.0 - pi0) * r1


# -------------------------
# State builder (OWN-ONLY info)
# -------------------------
def build_state(last_action_i: Action, agent_type_i: str) -> State:
    # Blind to others, but not blind to self: own last action + own type
    return (last_action_i, agent_type_i)


# -------------------------
# Training loop
# -------------------------
def run_blind(cfg: RunConfig):
    game = make_default_game_7()
    out_dir = ensure_outdir(cfg)

    agents: List[QLearningAgent] = [
        QLearningAgent(i, game.agent_types[i], alpha=cfg.alpha, gamma=cfg.gamma)
        for i in range(game.n_agents)
    ]

    # own-only memory
    last_action: List[Action] = [0] * game.n_agents

    # Track a few agents for Q plots (purely for reporting/plots)
    tracked_ids = {"X1": 0, "Y1": 2, "X3": 4}
    q_hist: Dict[str, List[Tuple[float, float]]] = {k: [] for k in tracked_ids}

    # counts by type
    cnt_by_type = {"X": 0, "Y": 0}
    for typ in game.agent_types:
        cnt_by_type[typ] += 1

    # --- Episode metrics ---
    avg_reward_ep_all: List[float] = []
    avg_reward_ep_type: Dict[str, List[float]] = {"X": [], "Y": []}

    regret_ep_all: List[float] = []
    regret_ep_type: Dict[str, List[float]] = {"X": [], "Y": []}

    exploit_ep_mean_all: List[float] = []
    exploit_ep_mean_type: Dict[str, List[float]] = {"X": [], "Y": []}

    nashconv_ep: List[float] = []

    disc_return_by_type: Dict[str, List[float]] = {"X": [], "Y": []}

    # per-timestep mean reward by type
    sum_r_by_type_t = {"X": [0.0] * cfg.horizon, "Y": [0.0] * cfg.horizon}

    # for exploitability: empirical mixed strategy each episode
    p_action2_hist: List[List[float]] = []

    # GIF frames: (episode, p_action2 list)
    gif_frames_data: List[Tuple[int, List[float]]] = []

    for ep in range(cfg.episodes):
        eps = epsilon_schedule(cfg, ep)

        ep_total_reward = 0.0
        ep_reward_by_type = {"X": 0.0, "Y": 0.0}
        ep_regret_by_agent = [0.0] * game.n_agents
        disc_returns = [0.0 for _ in range(game.n_agents)]
        action2_counts = [0] * game.n_agents

        # OPTION 2: store each tracked agent's *actual* state at the start of episode
        start_state_for_tracked: Dict[str, State] = {}

        for t in range(cfg.horizon):
            # build current states (own-only)
            states: List[State] = [
                build_state(last_action[i], game.agent_types[i])
                for i in range(game.n_agents)
            ]

            # capture states at t==0 for Q plotting (Option 2)
            if t == 0:
                for name, aid in tracked_ids.items():
                    start_state_for_tracked[name] = states[aid]

            # actions
            actions: List[Action] = [agents[i].select_action(states[i], eps) for i in range(game.n_agents)]
            rewards: List[float] = game.step(actions)

            # count action2
            for i, a in enumerate(actions):
                if a == 1:
                    action2_counts[i] += 1

            # Q update + discounted returns
            discount = (cfg.gamma ** t)
            for i, ag in enumerate(agents):
                next_state = build_state(actions[i], game.agent_types[i])
                ag.update(states[i], actions[i], rewards[i], next_state)

                disc_returns[i] += discount * rewards[i]
                last_action[i] = actions[i]

            # average reward accumulators
            ep_total_reward += sum(rewards)
            for i, r in enumerate(rewards):
                typ = game.agent_types[i]
                ep_reward_by_type[typ] += r
                sum_r_by_type_t[typ][t] += r

            # regret (external regret vs best fixed action in hindsight)
            for i in range(game.n_agents):
                r_actual = rewards[i]
                r0 = realized_reward_for_agent(game, i, 0, actions)
                r1 = realized_reward_for_agent(game, i, 1, actions)
                ep_regret_by_agent[i] += max(r0, r1) - r_actual

        # store policy frequencies
        p_action2 = [c / cfg.horizon for c in action2_counts]
        p_action2_hist.append(p_action2)

        # avg reward per episode (mean per agent per step)
        avg_all = ep_total_reward / (game.n_agents * cfg.horizon)
        avg_reward_ep_all.append(avg_all)
        for typ in ("X", "Y"):
            avg_reward_ep_type[typ].append(ep_reward_by_type[typ] / (cnt_by_type[typ] * cfg.horizon))

        # regret per episode (mean per agent per step)
        regret_all = sum(ep_regret_by_agent) / (game.n_agents * cfg.horizon)
        regret_ep_all.append(regret_all)
        for typ in ("X", "Y"):
            typ_ids = [i for i, t in enumerate(game.agent_types) if t == typ]
            regret_ep_type[typ].append(sum(ep_regret_by_agent[i] for i in typ_ids) / (cnt_by_type[typ] * cfg.horizon))

        # exploitability + NashConv from empirical mixed strategies
        p_action1 = [1.0 - p2 for p2 in p_action2]
        exploit_i = [0.0] * game.n_agents
        for i in range(game.n_agents):
            pi0 = p_action1[i]
            v_i = expected_reward_for_agent_mixed(game, i, pi0, p_action1)
            br_i = max(
                expected_reward_for_agent_action(game, i, 0, p_action1),
                expected_reward_for_agent_action(game, i, 1, p_action1),
            )
            exploit_i[i] = max(0.0, br_i - v_i)

        nashconv_ep.append(sum(exploit_i))
        exploit_ep_mean_all.append(sum(exploit_i) / game.n_agents)
        for typ in ("X", "Y"):
            typ_ids = [i for i, t in enumerate(game.agent_types) if t == typ]
            exploit_ep_mean_type[typ].append(sum(exploit_i[i] for i in typ_ids) / cnt_by_type[typ])

        # discounted return per episode per type
        disc_return_by_type["X"].append(
            sum(disc_returns[i] for i, t in enumerate(game.agent_types) if t == "X") / cnt_by_type["X"]
        )
        disc_return_by_type["Y"].append(
            sum(disc_returns[i] for i, t in enumerate(game.agent_types) if t == "Y") / cnt_by_type["Y"]
        )

        # Q snapshot (Option 2): use the REAL start-of-episode state for each tracked agent
        for name, aid in tracked_ids.items():
            s_plot = start_state_for_tracked.get(name)
            q = agents[aid].Q.get(s_plot, [0.0, 0.0])
            q_hist[name].append((q[0], q[1]))

        # GIF frame
        if (ep % cfg.gif_every) == 0:
            gif_frames_data.append((ep, p_action2.copy()))

        if (ep + 1) % 500 == 0:
            print(
                f"episode {ep+1}/{cfg.episodes} | eps={eps:.2f} | "
                f"avgR={avg_all:.3f} | nashconv={nashconv_ep[-1]:.3f}"
            )
            print("Q-table size agent 0:", len(agents[0].Q))

    mean_r_by_type_t = {
        typ: [v / cfg.episodes / cnt_by_type[typ] for v in vals]
        for typ, vals in sum_r_by_type_t.items()
    }

    return {
        "q_hist": q_hist,
        "mean_r_by_type_t": mean_r_by_type_t,
        "disc_return_by_type": disc_return_by_type,
        "avg_reward_ep_all": avg_reward_ep_all,
        "avg_reward_ep_type": avg_reward_ep_type,
        "regret_ep_all": regret_ep_all,
        "regret_ep_type": regret_ep_type,
        "exploit_ep_mean_all": exploit_ep_mean_all,
        "exploit_ep_mean_type": exploit_ep_mean_type,
        "nashconv_ep": nashconv_ep,
        "p_action2_hist": p_action2_hist,
        "gif_frames_data": gif_frames_data,
        "agent_types": list(game.agent_types),
        "adjacency": game.adjacency,
        "out_dir": str(out_dir),
    }


# -------------------------
# Plots + Network GIF
# -------------------------
def plot_results(results: dict, cfg: RunConfig):
    out_dir = ensure_outdir(cfg)
    p = cfg.out_prefix

    # Q-values per episode for tracked agents (now meaningful)
    for name, series in results["q_hist"].items():
        q0 = [x[0] for x in series]
        q1 = [x[1] for x in series]
        plt.figure()
        plt.plot(q0, label="Q(action 1)")
        plt.plot(q1, label="Q(action 2)")
        plt.title(f"{name}: Q-values per episode (blind, actual state)")
        plt.xlabel("Episode")
        plt.ylabel("Q")
        plt.legend()
        save_fig(out_dir, f"{p}_{name}_qvalues.png")

    # Mean reward per timestep by type
    for typ, vals in results["mean_r_by_type_t"].items():
        plt.figure()
        plt.plot(vals)
        plt.title(f"Mean reward per timestep for type {typ} (blind)")
        plt.xlabel("Timestep")
        plt.ylabel("Mean reward")
        save_fig(out_dir, f"{p}_mean_reward_type_{typ}.png")

    # Discounted return per episode by type
    for typ, vals in results["disc_return_by_type"].items():
        plt.figure()
        plt.plot(vals)
        plt.title(f"Mean discounted return per episode for type {typ} (blind)")
        plt.xlabel("Episode")
        plt.ylabel("Discounted return")
        save_fig(out_dir, f"{p}_disc_return_type_{typ}.png")

    # Avg reward per episode (overall + by type)
    plt.figure()
    plt.plot(results["avg_reward_ep_all"], label="All agents")
    plt.plot(results["avg_reward_ep_type"]["X"], label="Type X")
    plt.plot(results["avg_reward_ep_type"]["Y"], label="Type Y")
    plt.title("Average reward per episode (mean per agent per step)")
    plt.xlabel("Episode")
    plt.ylabel("Avg reward")
    plt.legend()
    save_fig(out_dir, f"{p}_avg_reward_per_episode.png")

    # Regret per episode
    plt.figure()
    plt.plot(results["regret_ep_all"], label="All agents")
    plt.plot(results["regret_ep_type"]["X"], label="Type X")
    plt.plot(results["regret_ep_type"]["Y"], label="Type Y")
    plt.title("External regret per episode (mean per agent per step)")
    plt.xlabel("Episode")
    plt.ylabel("Regret")
    plt.legend()
    save_fig(out_dir, f"{p}_regret_per_episode.png")

    # Exploitability per episode
    plt.figure()
    plt.plot(results["exploit_ep_mean_all"], label="All agents")
    plt.plot(results["exploit_ep_mean_type"]["X"], label="Type X")
    plt.plot(results["exploit_ep_mean_type"]["Y"], label="Type Y")
    plt.title("Exploitability (mean BR gain) per episode")
    plt.xlabel("Episode")
    plt.ylabel("Exploitability")
    plt.legend()
    save_fig(out_dir, f"{p}_exploitability_per_episode.png")

    # NashConv per episode
    plt.figure()
    plt.plot(results["nashconv_ep"])
    plt.title("NashConv per episode (sum of best-response gains)")
    plt.xlabel("Episode")
    plt.ylabel("NashConv")
    save_fig(out_dir, f"{p}_nashconv_per_episode.png")


def make_network_gif(results: dict, cfg: RunConfig):
    out_dir = ensure_outdir(cfg)
    gif_path = out_dir / f"{cfg.out_prefix}_network.gif"

    try:
        import networkx as nx
        import imageio.v2 as imageio
    except Exception:
        print("Missing dependency: install networkx + imageio to make the GIF.")
        return

    adjacency = results["adjacency"]
    agent_types = results["agent_types"]
    frames_data = results["gif_frames_data"]

    G = nx.Graph()
    n = len(agent_types)
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in adjacency[i]:
            if i < j:
                G.add_edge(i, j)

    pos = nx.spring_layout(G, seed=7)

    images = []
    for ep, p_action2 in frames_data:
        plt.figure()
        nx.draw_networkx_edges(G, pos, alpha=0.5)
        nx.draw_networkx_nodes(G, pos, node_size=900, node_color=p_action2, vmin=0.0, vmax=1.0)
        labels = {i: f"{i}\n{agent_types[i]}" for i in range(n)}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
        plt.title(f"Blind learning: node shade = P(action 2) | episode {ep}")
        plt.axis("off")
        plt.tight_layout()

        import io
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        images.append(imageio.imread(buf))
        buf.close()

    imageio.mimsave(gif_path, images, duration=0.25)
    print("Saved GIF:", gif_path)


def main():
    cfg = RunConfig()
    results = run_blind(cfg)
    plot_results(results, cfg)
    make_network_gif(results, cfg)
    print("Done. Saved all outputs in:", cfg.out_dir)


if __name__ == "__main__":
    main()
