"""Deterministic synthetic three-neuron example, with no real connectome data."""
import json
import numpy as np
from .circuit import Circuit, Graph


def synthetic_graph():
    # Invented body IDs: 101 -> 102 -> 103 -> 102 (last edge inhibitory).
    return Graph(np.array([0, 1, 2, 3]), np.array([1, 2, 1]),
                 np.array([4, 3, 2]), np.array([1, 1, -1]),
                 np.array([101, 102, 103]))


def run():
    circuit = Circuit(synthetic_graph(), plastic=np.array([True, True, False]))
    initial = circuit.weights.copy()
    for step in range(12):
        circuit.step([1.0 if step < 4 else 0.0, 0.0, 0.0],
                     modulator=[0.0, 1.0, 1.0] if step == 7 else 0.0)
    return {"data": "synthetic, not a biological circuit", "steps": 12,
            "state": circuit.state.round(6).tolist(),
            "initial_weights": initial.round(6).tolist(),
            "final_weights": circuit.weights.round(6).tolist()}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
