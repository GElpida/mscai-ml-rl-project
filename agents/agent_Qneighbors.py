from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

Action = int  # 0 -> action 1, 1 -> action 2
State = Tuple  # hashable tuple state


class QLearningNeighborAgent:
    """
    Q-learning agent with epsilon-greedy exploration that *observes neighbors' actions*.

    Intended usage is self-play: instantiate one agent per node in the graph, all
    using this same class, and train them simultaneously in the repeated game.

    Recommended Markov state (one-step memory):
        state = (agent_type, own_last_action, neighbors_last_actions_tuple)

    Notes:
    - "neighbors_last_actions_tuple" must follow a fixed ordering of neighbor ids
      (we store neighbors sorted) so the state is consistent across steps.
    - You can still pass any custom state tuple into select_action/update if you
      prefer to construct state outside the agent.
    """

    def __init__(
        self,
        agent_id: int,
        agent_type: str,  # "X" or "Y"
        neighbors: Sequence[int],
        n_actions: int = 2,
        alpha: float = 0.1,
        gamma: float = 0.95,
    ):
        self.id = agent_id
        self.type = agent_type
        self.neighbors = sorted(set(int(n) for n in neighbors))

        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma

        self.Q: Dict[State, List[float]] = {}

        self.last_state: Optional[State] = None
        self.last_action: Optional[Action] = None

    def build_state(self, last_actions: Sequence[Action]) -> State:
        """
        Build the default local-observation state from the global last-actions vector.
        """
        own_last = int(last_actions[self.id])
        neis_last = tuple(int(last_actions[j]) for j in self.neighbors)
        return (self.type, own_last, neis_last)

    # ----------------------------------
    # Action selection (epsilon-greedy)
    # ----------------------------------
    def select_action(self, state: State, epsilon: float) -> Action:
        if state not in self.Q:
            self.Q[state] = [0.0] * self.n_actions

        if random.random() < epsilon:
            action = random.randint(0, self.n_actions - 1)
        else:
            q_vals = self.Q[state]
            max_q = max(q_vals)
            best_actions = [i for i, q in enumerate(q_vals) if q == max_q]
            action = random.choice(best_actions)

        self.last_state = state
        self.last_action = action
        return action

    def select_action_from_observation(self, last_actions: Sequence[Action], epsilon: float) -> Action:
        """
        Convenience wrapper around build_state(...) + select_action(...).
        """
        return self.select_action(self.build_state(last_actions), epsilon)

    # ----------------------------------
    # Q-learning update
    # ----------------------------------
    def update(self, state: State, action: Action, reward: float, next_state: State) -> None:
        if state not in self.Q:
            self.Q[state] = [0.0] * self.n_actions
        if next_state not in self.Q:
            self.Q[next_state] = [0.0] * self.n_actions

        q_sa = self.Q[state][action]
        target = reward + self.gamma * max(self.Q[next_state])
        self.Q[state][action] = q_sa + self.alpha * (target - q_sa)
