# Agents

Ο φάκελος `agents/` περιέχει υλοποιήσεις Q-learning πρακτόρων (ε-greedy) για το repeated coordination game του repo.

## Κοινές συμβάσεις

- **Ενέργειες (`Action`)**: `0` = “action 1”, `1` = “action 2” (ίδια κωδικοποίηση με `game/coordination_repeated_game.py`).
- **Κατάσταση (`State`)**: οποιοδήποτε *hashable* `tuple` (χρησιμοποιείται ως key στο Q-table).
- **Q-table**: `Q[state] -> [Q(s,0), Q(s,1)]`
- **API που χρησιμοποιούν τα experiments**:
  - `select_action(state, epsilon) -> action`
  - `update(state, action, reward, next_state) -> None`

## `agent_Qblind.py` — Blind Q-learning

Αρχείο: `agents/agent_Qblind.py`

Class: `QLearningAgent`

- “Blind” = **δεν παρατηρεί** ενέργειες άλλων πρακτόρων.
- Το τι σημαίνει `state` το αποφασίζει ο runner (π.χ. `experiments/run_blind_main.py` χτίζει state από *μόνο* τη δική του τελευταία ενέργεια).
- Χρήσιμο όταν θέλουμε να δούμε τι μαθαίνει ο agent χωρίς πληροφορία για το τι έκαναν οι γείτονες.

## `agent_Qneighbors.py` — Q-learning με παρατήρηση γειτόνων

Αρχείο: `agents/agent_Qneighbors.py`

Class: `QLearningNeighborAgent`

Σε αυτή την εκδοχή ο agent “ξέρει” τις ενέργειες των γειτόνων του στο γράφο, επειδή το state περιλαμβάνει την τελευταία ενέργεια κάθε γείτονα.

### Neighbors / γράφος

- Στο `__init__(..., neighbors=...)` περνάς τη λίστα γειτόνων από το περιβάλλον (συνήθως `neighbors = game.adjacency[i]`).
- Ο agent αποθηκεύει τους γείτονες ταξινομημένους (`sorted`) ώστε το `neighbors_last_actions_tuple` να έχει **σταθερή σειρά**.

### Default state

Η “προτεινόμενη” Markov κατάσταση 1-step memory είναι:

`(agent_type, own_last_action, neighbors_last_actions_tuple)`

και χτίζεται με:

- `build_state(last_actions)` όπου `last_actions` είναι ένα vector μήκους `n_agents` με τις τελευταίες ενέργειες όλων των πρακτόρων.

Υπάρχει και convenience method:

- `select_action_from_observation(last_actions, epsilon)`

### Τι σημαίνει “ο agent γνωρίζει τις ενέργειες των γειτόνων”

Το “knowledge” εδώ είναι observation που του δίνει ο runner:

1. Ο runner κρατάει `last_actions[i]` για όλους τους agents.
2. Σε κάθε round, πριν την επιλογή action, ο agent κάνει `build_state(last_actions)` και διαβάζει `last_actions[j]` μόνο για `j in neighbors`.
3. Μετά το `game.step(actions)`, ο runner ενημερώνει `last_actions[i] = actions[i]` για να είναι διαθέσιμες στον επόμενο γύρο.

## Πώς χρησιμοποιούνται (self-play)

Δες τα runners:

- `experiments/run_blind_main.py` (blind agent)
- `experiments/run_neighbors_main.py` (neighbor-observing agent)

Και στα δύο, γίνεται self-play: δημιουργείται ένας agent ανά κόμβο και όλοι εκπαιδεύονται ταυτόχρονα στο ίδιο περιβάλλον.

