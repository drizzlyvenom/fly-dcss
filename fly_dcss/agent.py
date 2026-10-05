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
    def __init__(self, graph, *, seed=7, epsilon=0.25):
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be in [0, 1]")
        self.rng = np.random.default_rng(seed)
        self.epsilon = epsilon
        incoming = np.bincount(graph.indices, weights=graph.weight, minlength=graph.n)
        self.circuit = Circuit(graph, Parameters(weight_scale=0.5 / max(1.0, incoming.max()), learning_rate=0.02))
        self.projection = None
        self.readout = self.rng.uniform(0, 1, (len(ACTION_NAMES), graph.n))
        self.readout /= self.readout.sum(axis=1, keepdims=True)
        self.visited = set()

    def external(self, observation):
        features = encode(observation)
        if self.projection is None:
            self.projection = self.rng.uniform(0, 1, (self.circuit.graph.n, len(features)))
            self.projection /= self.projection.sum(axis=1, keepdims=True)
        return self.projection @ features

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
        old = self.circuit.weights.copy()
        # Feedback uses resultant observation, retained activity and edge-local eligibility.
        self.circuit.step(self.external(after), modulator=reward, learning=True)
        return {"reward": reward, "new_location": new_location, "hp_delta": delta_hp,
                "turn_delta": delta_turn, "weights_delta_l1": float(np.abs(self.circuit.weights - old).sum()),
                "weights_before": old.tolist(), "weights_after": self.circuit.weights.tolist(),
                "state": self.circuit.state.tolist(), "circuit_time": self.circuit.time}
