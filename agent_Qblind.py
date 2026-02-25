from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

Action = int  # 0 -> action 1, 1 -> action 2
State = Tuple  # generic tuple state (can be empty ())


class QLearningAgent:
    """
    Blind / agnostic Q-learning agent:
    - Does NOT observe opponents' actions.
    - Learns only from (state, own_action, own_reward, next_state).

    You decide what "state" means outside the agent. For the fully-blind case,
    you can simply use state=() (stateless bandit-like learning).
    """

    def __init__(
        self,
        agent_id: int,
        agent_type: str,           # "X" or "Y"
        n_actions: int = 2,
        alpha: float = 0.1,
        gamma: float = 0.95,
    ):
        self.id = agent_id
        self.type = agent_type
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma

        # Q-table: Dict[state -> List[action_values]]
        self.Q: Dict[State, List[float]] = {}

        # For convenience/debugging
        self.last_state: Optional[State] = None
        self.last_action: Optional[Action] = None

    # ----------------------------------
    # Action selection (ε-greedy)
    # ----------------------------------
    def select_action(self, state: State, epsilon: float) -> Action:
        if state not in self.Q:
            self.Q[state] = [0.0] * self.n_actions

        # exploration
        if random.random() < epsilon:
            action = random.randint(0, self.n_actions - 1)
        else:
            q_vals = self.Q[state]
            max_q = max(q_vals)
            # tie-breaking randomly
            best_actions = [i for i, q in enumerate(q_vals) if q == max_q]
            action = random.choice(best_actions)

        self.last_state = state
        self.last_action = action
        return action

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
