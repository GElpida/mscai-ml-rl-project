# Coordination repeated game environment (stage game + sparse communication graph)
# Only game rules + reward computation.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


# -----------------------------
# Types
# -----------------------------
AgentId = int
Action = int  # 0 -> action 1, 1 -> action 2
AgentType = str  # "X" (column player) or "Y" (row player)


# -----------------------------
# Payoff matrices (row, col)
# We encode actions as 0=1 and 1=2
#
# X-Y interactions (Y is row, X is col):
#         Col1   Col2
# Row1    1,2    1,1
# Row2    1,1    2,1
#
# X-X interactions:
#         Col1   Col2
# Row1    2,2    1,1
# Row2    1,1    1,1
#
# Y-Y interactions:
#         Col1   Col2
# Row1    1,1    1,1
# Row2    1,1    2,2
# -----------------------------
def payoff(row_action: Action, col_action: Action) -> Tuple[float, float]:
    """Returns (row_reward, col_reward)."""
    if row_action == 0 and col_action == 0:
        return (1.0, 2.0)
    if row_action == 0 and col_action == 1:
        return (1.0, 1.0)
    if row_action == 1 and col_action == 0:
        return (1.0, 1.0)
    if row_action == 1 and col_action == 1:
        return (2.0, 1.0)
    raise ValueError(f"Invalid actions: row={row_action}, col={col_action}")


def payoff_xx(row_action: Action, col_action: Action) -> Tuple[float, float]:
    """Returns (row_reward, col_reward) for X-X interactions."""
    if row_action == 0 and col_action == 0:
        return (2.0, 2.0)
    if row_action == 0 and col_action == 1:
        return (1.0, 1.0)
    if row_action == 1 and col_action == 0:
        return (1.0, 1.0)
    if row_action == 1 and col_action == 1:
        return (1.0, 1.0)
    raise ValueError(f"Invalid actions: row={row_action}, col={col_action}")


def payoff_yy(row_action: Action, col_action: Action) -> Tuple[float, float]:
    """Returns (row_reward, col_reward) for Y-Y interactions."""
    if row_action == 0 and col_action == 0:
        return (1.0, 1.0)
    if row_action == 0 and col_action == 1:
        return (1.0, 1.0)
    if row_action == 1 and col_action == 0:
        return (1.0, 1.0)
    if row_action == 1 and col_action == 1:
        return (2.0, 2.0)
    raise ValueError(f"Invalid actions: row={row_action}, col={col_action}")


# -----------------------------
# Graph helpers
# -----------------------------
def validate_adjacency(adjacency: Dict[AgentId, List[AgentId]], n_agents: int) -> None:
    for i in range(n_agents):
        if i not in adjacency:
            raise ValueError(f"adjacency missing node {i}")
        for j in adjacency[i]:
            if j < 0 or j >= n_agents:
                raise ValueError(f"Invalid neighbor id {j} for node {i}")


def make_undirected(adjacency: Dict[AgentId, List[AgentId]], n_agents: int) -> Dict[AgentId, List[AgentId]]:
    """Ensures the graph is symmetric (i <-> j)."""
    out = {i: list(adjacency.get(i, [])) for i in range(n_agents)}
    for i in range(n_agents):
        for j in out[i]:
            if i not in out[j]:
                out[j].append(i)
    return {i: sorted(set(neis)) for i, neis in out.items()}


# -----------------------------
# Defaults for THIS assignment graph
# Node indexing:
# 0: X1, 1: X2, 2: Y1, 3: Y2, 4: X3, 5: X4, 6: X5
# -----------------------------
def default_agent_types_7() -> List[AgentType]:
    return ["X", "X", "Y", "Y", "X", "X", "X"]


def default_adjacency_7() -> Dict[AgentId, List[AgentId]]:
    # Undirected edges (both directions implied via `undirected=True`):
    # X1—Y1
    # Y1—X4
    # Y1—X3
    # X4—X3
    # X3—Y2
    # X3—X5
    # X5—Y2
    # X2—Y2
    return {
        0: [2],           # X1 — Y1
        1: [3],           # X2 — Y2
        2: [0, 4, 5],     # Y1 — X1, X3, X4
        3: [1, 4, 6],     # Y2 — X2, X3, X5
        4: [2, 3, 5, 6],  # X3 — Y1, Y2, X4, X5
        5: [2, 4],        # X4 — Y1, X3
        6: [3, 4],        # X5 — Y2, X3
    }


# -----------------------------
# Environment
# -----------------------------
@dataclass
class CoordinationGame:
    n_agents: int
    agent_types: Sequence[AgentType]        # length n_agents, each "X" or "Y"
    adjacency: Dict[AgentId, List[AgentId]] # neighbor list per agent
    undirected: bool = True                 # if True, force symmetric edges
    aggregate: str = "sum"                  # "sum" or "mean" over neighbor interactions

    def __post_init__(self) -> None:
        if len(self.agent_types) != self.n_agents:
            raise ValueError("agent_types must have length n_agents")
        for t in self.agent_types:
            if t not in ("X", "Y"):
                raise ValueError(f"Invalid agent type {t}. Use 'X' or 'Y'.")
        if self.undirected:
            self.adjacency = make_undirected(self.adjacency, self.n_agents)
        validate_adjacency(self.adjacency, self.n_agents)
        if self.aggregate not in ("sum", "mean"):
            raise ValueError("aggregate must be 'sum' or 'mean'")

    def step(self, actions: Sequence[Action]) -> List[float]:
        if len(actions) != self.n_agents:
            raise ValueError("actions must have length n_agents")
        for a in actions:
            if a not in (0, 1):
                raise ValueError(f"Invalid action {a}; must be 0 or 1")

        rewards = [0.0 for _ in range(self.n_agents)]
        counts = [0 for _ in range(self.n_agents)]

        for i in range(self.n_agents):
            for j in self.adjacency[i]:
                # avoid double counting undirected edges
                if self.undirected and j < i:
                    continue

                ri, rj = self._pair_reward(i, j, actions[i], actions[j])

                rewards[i] += ri
                rewards[j] += rj
                counts[i] += 1
                counts[j] += 1

        if self.aggregate == "mean":
            for i in range(self.n_agents):
                if counts[i] > 0:
                    rewards[i] /= counts[i]
        return rewards

    def _pair_reward(self, i: AgentId, j: AgentId, ai: Action, aj: Action) -> Tuple[float, float]:
        ti = self.agent_types[i]
        tj = self.agent_types[j]
        
        if ti == "X" and tj == "X":
            return payoff_xx(ai, aj)
        if ti == "Y" and tj == "Y":
            return payoff_yy(ai, aj)

        # Y is row, X is col
        if ti == "Y" and tj == "X":
            return payoff(ai, aj)
        if ti == "X" and tj == "Y":
            row_r, col_r = payoff(aj, ai)
            return col_r, row_r

        raise ValueError(f"Invalid agent types on edge ({i},{j}): {ti}-{tj}")


def make_default_game_7() -> CoordinationGame:
    return CoordinationGame(
        n_agents=7,
        agent_types=default_agent_types_7(),
        adjacency=default_adjacency_7(),
        undirected=True,
        aggregate="sum",
    )
