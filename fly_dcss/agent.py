"""Small fixed encoder/readout and local feedback update; not a learned policy claim."""
import numpy as np
from .circuit import Circuit, Parameters
from .webtiles import ACTIONS, DELTAS

ACTION_NAMES = tuple(ACTIONS)


def encode(observation):
    """Keep unknown and remembered cells separate, never encode remembered monsters."""
    p = observation["player"]
    values = [p["hp"] / max(1, p["hp_max"]), p["mp"] / max(1, p["mp_max"]), 1.0]
    for cell in observation["cells"]:
        visible = cell["visibility"] == "visible"
        values.extend([float(cell["visibility"] == "unknown"),
                       float(cell["visibility"] == "remembered"), float(visible),
                       float(visible and cell["glyph"] in ".<>"),
                       float(visible and cell["monster"])])
    return np.asarray(values)


def permitted_actions(observation):
    cells = {(c["dx"], c["dy"]): c for c in observation["cells"]}
    permitted = ["wait"]
    for action in ACTION_NAMES[:-1]:
        cell = cells.get(DELTAS[action])
        # Deliberately conservative first loop: no doors, water, unknown terrain.
        if cell and cell["visibility"] == "visible" and (cell["glyph"] in ".<>" or cell["monster"]):
            permitted.append(action)
    return permitted


class LoopAgent:
    def __init__(self, graph, *, seed=7, epsilon=0.25, compact=False):
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be in [0, 1]")
        self.rng = np.random.default_rng(seed)
        self.epsilon = epsilon
        self.compact = compact
        self.seed = seed
        incoming = np.zeros(graph.n, dtype=np.float64)
        for start in range(0, len(graph.indices), 1_000_000):
            sl = slice(start, start + 1_000_000)
            incoming += np.bincount(graph.indices[sl], weights=graph.weight[sl], minlength=graph.n)
        self.circuit = Circuit(graph, Parameters(weight_scale=0.5 / max(1.0, incoming.max()), learning_rate=0.02),
                               dtype=np.float32 if compact else np.float64)
        self.projection = None
        self.feature_indices = None
        self.readout = self.rng.uniform(0, 1, (len(ACTION_NAMES), graph.n))
        self.readout = self.readout.astype(np.float32 if compact else np.float64)
        self.readout /= self.readout.sum(axis=1, keepdims=True)
        self.visited = set()

    def external(self, observation):
        features = encode(observation)
        if self.compact:
            if self.feature_indices is None:
                # Four fixed feature links per neuron; no N-by-feature dense matrix.
                count = self.circuit.graph.n * 4
                self.feature_indices = (np.arange(count, dtype=np.int32) % len(features)).reshape(-1, 4)
                self.rng.shuffle(self.feature_indices, axis=0)
                self.projection = self.rng.uniform(0.1, 1, self.feature_indices.shape).astype(np.float32)
                self.projection /= self.projection.sum(axis=1, keepdims=True)
            return np.sum(self.projection * features[self.feature_indices], axis=1).astype(np.float32)
        if self.projection is None:
            self.projection = self.rng.uniform(0, 1, (self.circuit.graph.n, len(features)))
            self.projection /= self.projection.sum(axis=1, keepdims=True)
        return self.projection @ features

    def mapping_summary(self):
        return {"mapping_kind": "fixed engineered mappings, not biological sensory/motor assignments",
                "seed": self.seed, "input_nodes": self.circuit.graph.n,
                "encoded_features": 128, "features_connected": (int(len(np.unique(self.feature_indices))) if self.feature_indices is not None else 0) if self.compact else 128,
                "input_links_per_node": 4 if self.compact else 128,
                "readout_nodes_per_action": self.circuit.graph.n,
                "actions": list(ACTION_NAMES), "readout_finite": bool(np.isfinite(self.readout).all()),
                "readout_row_sums": self.readout.sum(axis=1).tolist()}

    def choose(self, observation):
        state = self.circuit.step(self.external(observation), learning=False)
        scores = self.readout @ state
        permitted = permitted_actions(observation)
        indices = [ACTION_NAMES.index(a) for a in permitted]
        explore = self.rng.random() < self.epsilon
        index = int(self.rng.choice(indices)) if explore else max(indices, key=lambda i: scores[i])
        return ACTION_NAMES[index], {"scores": dict(zip(ACTION_NAMES, scores.tolist())),
                                    "permitted": permitted, "exploratory": explore}

    @staticmethod
    def location(observation):
        p = observation["player"]
        return (p.get("place"), p.get("depth"), p["pos"]["x"], p["pos"]["y"])

    def feedback(self, before, after):
        self.visited.add(self.location(before))
        new_location = self.location(after) not in self.visited
        self.visited.add(self.location(after))
        delta_hp = after["player"]["hp"] - before["player"]["hp"]
        delta_turn = after["player"]["turn"] - before["player"]["turn"]
        if delta_turn < 0:
            raise ValueError("game turn regressed")
        reward = float(np.clip(0.05 * new_location + delta_hp / max(1, before["player"]["hp_max"]) - 0.01 * max(1, delta_turn), -1, 1))
        # Feedback uses resultant observation, retained activity and edge-local eligibility.
        self.circuit.step(self.external(after), modulator=reward, learning=True)
        return {"reward": reward, "new_location": new_location, "hp_delta": delta_hp,
                "turn_delta": delta_turn, "weights_delta_l1": self.circuit.last_update_l1,
                "state_min": float(self.circuit.state.min()),
                "state_max": float(self.circuit.state.max()),
                "state_mean": float(self.circuit.state.mean()), "circuit_time": self.circuit.time}
