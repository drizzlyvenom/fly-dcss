#!/usr/bin/env python3
"""Bounded functional check of prepared full-graph recurrence and action readout."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fly_dcss.agent import ACTION_NAMES, LoopAgent
from fly_dcss.malecns import load_malecns


def check(path):
    graph, metadata = load_malecns(path)
    agent = LoopAgent(graph, compact=True, seed=7)
    circuit = agent.circuit
    # One excitatory source with an actual off-diagonal outgoing edge.
    source = next(i for i in np.flatnonzero((graph.sign > 0) & (np.diff(graph.indptr) > 0))
                  if np.any(graph.indices[graph.indptr[i]:graph.indptr[i + 1]] != i))
    row = slice(graph.indptr[source], graph.indptr[source + 1])
    pulse = np.float32(0.5)
    circuit.state[source] = pulse
    expected_current = np.bincount(graph.indices[row],
                                   weights=circuit.weights[row] * pulse,
                                   minlength=graph.n)
    current = circuit._recurrent_current()
    np.testing.assert_allclose(current, expected_current, rtol=1e-6, atol=1e-10)
    decay = np.exp(-circuit.parameters.dt / circuit.parameters.tau_state)
    expected_state = (decay * circuit.state +
                      (1 - decay) * np.clip(expected_current, 0, 1)).astype(np.float32)
    actual = circuit.step(np.zeros(graph.n, dtype=np.float32), learning=False)
    np.testing.assert_allclose(actual, expected_state, rtol=1e-6, atol=1e-9)
    # Remove the directly pulsed node: remaining signal must traverse source edges.
    reached = actual.copy()
    reached[source] = 0
    scores = agent.readout @ reached
    if not np.all(scores > 0) or not np.isfinite(scores).all():
        raise AssertionError('recurrent-only pulse did not reach every action readout')
    return {"scope": metadata['scope'], "nodes": graph.n, "edges": len(graph.indices),
            "source_body_id": int(graph.body[source]), "source_outgoing_rows": int(row.stop - row.start),
            "downstream_active_nodes_excluding_source": int(np.count_nonzero(reached)),
            "max_current_absolute_error": float(np.max(np.abs(current - expected_current))),
            "all_five_readouts_receive_recurrent_signal": True,
            "recurrent_only_action_scores": dict(zip(ACTION_NAMES, scores.tolist())),
            "learning_disabled": True,
            "interpretation": "Functional engineered wiring check, not biological port validation or task efficacy"}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--malecns', required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.malecns), indent=2))
