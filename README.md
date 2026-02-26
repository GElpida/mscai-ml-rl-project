# MSC AI — ML/RL Project: Coordination Repeated Game (7 agents)

Η εργασία υλοποιεί ένα repeated coordination game πάνω σε γράφο 7 πρακτόρων και 2 Q-learning agents:

- **Blind Q-learning**: ο πράκτορας δεν παρατηρεί ενέργειες άλλων (μόνο τη δική του κατάσταση/ανταμοιβή).
- **Neighbor-observing Q-learning**: ο πράκτορας παρατηρεί τις τελευταίες ενέργειες των γειτόνων του στο γράφο.

Οι κύριες εκτελέσεις βρίσκονται στα `experiments/` και αποθηκεύουν plots + `network.gif` μέσα στο `results/`.

## Setup

1) Python (προτείνεται 3.10+).

2) Εγκατάσταση dependencies:

```bash
pip install -r requirements.txt
```

## Δομή

```
agents/        Q-learning agents
experiments/   training + plots + network GIF
game/          game rules (payoffs, adjacency graph, step())
results/       παραγόμενα outputs (plots/GIF)
```

Σημαντικά αρχεία:

- `game/coordination_repeated_game.py`: environment (payoffs, graph, `CoordinationGame.step`)
- `agents/agent_Qblind.py`: blind Q-learning agent
- `agents/agent_Qneighbors.py`: Q-learning agent που παρατηρεί γείτονες
- `experiments/run_blind_main.py`: training/plots για blind setup
- `experiments/run_neighbors_main.py`: training/plots για neighbors setup

Περισσότερα για τους agents: `agents/README.md`.

## Usage

### 1) Blind self-play

Τρέχει Q-learning self-play όπου το state είναι “own-only” (ο agent δεν βλέπει άλλους):

```bash
python experiments/run_blind_main.py
```

Outputs: `results/Q_blind/`

### 2) Neighbors self-play

Τρέχει Q-learning self-play όπου το state περιλαμβάνει και τις τελευταίες ενέργειες των γειτόνων:

```bash
python experiments/run_neighbors_main.py
```

Outputs: `results/Q_neighbors/`

## Παραμετροποίηση (RunConfig)

Για κάθε experiment μπορούν να παραμετροποιηθούν:

- `episodes`: πόσα episodes θα τρέξουν
- `horizon`: πόσα rounds ανά episode
- `alpha`, `gamma`: Q-learning hyperparameters
- `eps_start`, `eps_drop`, `eps_drop_every`: ε-greedy schedule
- `out_dir`: φάκελος αποθήκευσης (relative στο root του repo)