import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from fly_dcss.circuit import Circuit, Graph, Parameters, load_converted
from fly_dcss.demo import run, synthetic_graph


class CircuitTests(unittest.TestCase):
    def pair(self, **params):
        graph = Graph(np.array([0, 1, 1]), np.array([1]),
                      np.array([5]), np.array([1, 1]), np.array([10, 20]))
        return Circuit(graph, Parameters(**params))

    def test_state_evolves_synchronously(self):
        c = self.pair()
        a = np.exp(-0.5)
        assert_allclose(c.step([1, 0]), [1 - a, 0])
        assert_allclose(c.step([0, 0]), [a * (1 - a), 0.5 * (1 - a)**2])

    def test_learning_disabled_freezes_weights_not_state(self):
        c = self.pair()
        original = c.weights.copy()
        for _ in range(10):
            c.step([1, 0], modulator=1, learning=False)
        assert_array_equal(c.weights, original)
        self.assertGreater(c.state[1], 0)
        self.assertGreater(c.eligibility[0], 0)

    def test_delayed_modulation_uses_eligibility(self):
        c = self.pair()
        for _ in range(3):
            c.step([1, 0])
        old_weight = c.weights.copy()
        old_trace = c.eligibility.copy()
        # Remove residual activity to isolate the stored trace, not fresh pairing.
        c.state[:] = 0
        c.step([0, 0], modulator=1)
        expected_trace = old_trace * np.exp(-1 / 5)
        assert_allclose(c.eligibility, expected_trace)
        assert_allclose(c.weights, old_weight + 0.1 * expected_trace)
        self.assertGreater(c.weights[0], old_weight[0])

    def test_trace_decay_and_no_modulator(self):
        c = self.pair()
        for _ in range(3):
            c.step([1, 0])
        initial = c.eligibility.copy()
        c.state[:] = 0
        for _ in range(4):
            c.step([0, 0])
        assert_allclose(c.eligibility, initial * np.exp(-4 / 5))
        assert_array_equal(c.weights, [0.5])

    def test_trace_required_for_weight_change(self):
        c = self.pair()
        c.step([0, 0], modulator=100)
        assert_array_equal(c.weights, [0.5])

    def test_bounds_and_negative_modulation(self):
        c = self.pair(learning_rate=100)
        for _ in range(5):
            c.step([1, 1], modulator=1)
        assert_array_equal(c.weights, [1])
        c.step([1, 1], modulator=-1)
        assert_array_equal(c.weights, [0])
        self.assertTrue(np.all((c.state >= 0) & (c.state <= 1)))

    def test_nondefault_dt_decay_and_learning_scale(self):
        c = self.pair(dt=0.25)
        c.state[:] = 0
        c.eligibility[:] = 0.4
        c.step([0, 0], modulator=2)
        expected = 0.4 * np.exp(-0.25 / 5)
        assert_allclose(c.eligibility, [expected])
        assert_allclose(c.weights, [0.5 + 0.1 * 0.25 * 2 * expected])
        # Decay is invariant to partitioning an interval without activity.
        short = self.pair(dt=0.25)
        long = self.pair(dt=1)
        short.eligibility[:] = long.eligibility[:] = 0.4
        for _ in range(4):
            short.step([0, 0])
        long.step([0, 0])
        assert_allclose(short.eligibility, long.eligibility)

    def test_reject_scaled_learning_overflow(self):
        with self.assertRaises(ValueError):
            Parameters(learning_rate=1e308, dt=2)
        c = self.pair(learning_rate=1e308)
        with self.assertRaises(ValueError):
            c.step([0, 0], modulator=2)
        self.assertEqual(c.time, 0)
        c.step([0, 0], modulator=0)
        assert_array_equal(c.weights, [0.5])

    def test_current_overflow_does_not_poison_state(self):
        g = Graph(np.array([0, 1, 2, 2]), np.array([2, 2]),
                  np.array([1., 1.]), np.ones(3), np.array([1, 2, 3]))
        c = Circuit(g, Parameters(weight_scale=1e308, weight_max=1e308))
        c.state[:] = [1, 1, 0]
        with self.assertRaises(ValueError):
            c.step([0, 0, 0])
        assert_array_equal(c.state, [1, 1, 0])
        assert_array_equal(c.eligibility, [0, 0])
        self.assertEqual(c.time, 0)

    def test_local_modulator_and_plastic_mask(self):
        g = Graph(np.array([0, 2, 2, 2]), np.array([1, 2]),
                  np.array([5, 5]), np.ones(3), np.array([1, 2, 3]))
        c = Circuit(g)
        for _ in range(5):
            c.step([1, 0, 0], modulator=[0, 1, 0])
        self.assertGreater(c.weights[0], 0.5)
        self.assertEqual(c.weights[1], 0.5)
        masked = Circuit(g, plastic=np.array([False, True]))
        for _ in range(5):
            masked.step([1, 1, 1], modulator=1)
        self.assertEqual(masked.weights[0], 0.5)
        self.assertEqual(masked.eligibility[0], 0)
        self.assertGreater(masked.weights[1], 0.5)

    def test_inhibitory_sign_and_graph_unchanged(self):
        g = synthetic_graph()
        c = Circuit(g)
        c.state[:] = [0, 0, 1]
        c.step([0, 0.1, 0], learning=False)
        self.assertEqual(c.state[1], 0)
        assert_array_equal(g.weight, [4, 3, 2])
        assert_array_equal(g.sign, [1, 1, -1])
        with self.assertRaises(ValueError):
            g.weight[0] = 100

    def test_reproducibility_and_independent_state(self):
        self.assertEqual(run(), run())
        a, b = self.pair(), self.pair()
        result = a.step([1, 0])
        result[:] = 0
        self.assertGreater(a.state[0], 0)
        assert_array_equal(b.state, [0, 0])

    def test_invalid_inputs_do_not_mutate(self):
        c = self.pair()
        for inp, mod in [([1], 0), ([np.nan, 0], 0), ([1, 0], [1]),
                         ([1, 0], np.inf)]:
            with self.assertRaises(ValueError):
                c.step(inp, modulator=mod)
        self.assertEqual(c.time, 0)
        assert_array_equal(c.state, [0, 0])
        for kwargs in [{"dt": 0}, {"tau_state": -1}, {"learning_rate": -1},
                       {"weight_max": np.nan}, {"weight_scale": -1}]:
            with self.assertRaises(ValueError):
                Parameters(**kwargs)


class CompactCircuitTests(unittest.TestCase):
    @staticmethod
    def graph(*, compact=False):
        rng = np.random.default_rng(71)
        degrees = np.array([3, 0, 5, 2, 1, 6, 3])
        return Graph(np.r_[0, np.cumsum(degrees)],
                     rng.integers(0, len(degrees), degrees.sum(),
                                  dtype=np.int32 if compact else np.int64),
                     rng.uniform(0, 4, degrees.sum()).astype(
                         np.float32 if compact else np.float64),
                     np.array([1, -1, 1, -1, 1, 1, -1]),
                     np.arange(len(degrees)) + 100)

    @staticmethod
    def reference_step(graph, p, state, trace, weights, external, modulator,
                       plastic, learning):
        # Original, unchunked float64 equations retained as an independent
        # reference for ordering, signed recurrence and clipped local learning.
        pre, post = graph.pre, graph.indices
        current = np.bincount(post, weights=weights * graph.sign[pre] * state[pre],
                              minlength=graph.n)
        a, b = np.exp(-p.dt / p.tau_state), np.exp(-p.dt / p.tau_eligibility)
        next_state = a * state + (1 - a) * np.clip(external + current, 0, 1)
        next_trace = (b * trace + (1 - b) * state[pre] * next_state[post]) * plastic
        if learning:
            signal = p.learning_rate * p.dt * np.asarray(modulator)
            local = signal if signal.ndim == 0 else signal[post]
            weights = np.clip(weights + local * next_trace, 0, p.weight_max)
        return next_state, next_trace, weights

    def test_float64_chunked_matches_original_equations(self):
        g = self.graph()
        p = Parameters(dt=0.25, weight_scale=0.2, learning_rate=3, weight_max=0.9)
        plastic = np.arange(len(g.indices)) % 3 != 0
        for chunk in (1, 4, 1_000_000, None):
            with self.subTest(chunk=chunk):
                c = Circuit(g, p, plastic=plastic, edge_chunk_size=chunk)
                state, trace, weights = c.state.copy(), c.eligibility.copy(), c.weights.copy()
                for step in range(30):
                    external = np.linspace(-0.2, 0.8, g.n) + 0.1 * (step % 3)
                    modulator = (np.linspace(-2, 2, g.n) if step % 2 else
                                 (1.5 if step < 15 else -1.5))
                    learning = step % 5 != 0
                    state, trace, weights = self.reference_step(
                        g, p, state, trace, weights, external, modulator, plastic, learning)
                    c.step(external, modulator=modulator, learning=learning)
                    assert_allclose(c.state, state, rtol=1e-13, atol=1e-15)
                    assert_allclose(c.eligibility, trace, rtol=1e-13, atol=1e-15)
                    assert_allclose(c.weights, weights, rtol=1e-13, atol=1e-15)
                self.assertEqual(c.state.dtype, np.float64)

    def test_compact_chunked_matches_float64_and_reuses_edge_buffers(self):
        g = self.graph(compact=True)
        plastic = np.arange(len(g.indices)) % 4 != 0
        p = Parameters(weight_scale=0.15, learning_rate=0.3)
        reference = Circuit(g, p, plastic=plastic, edge_chunk_size=None)
        c = Circuit(g, p, plastic=plastic, dtype=np.float32, edge_chunk_size=3)
        weights_buffer, trace_buffer = c.weights, c.eligibility
        self.assertEqual(c.pre.dtype, np.int32)
        for step in range(40):
            external = np.linspace(0, 0.9, g.n)
            modulator = np.linspace(-2, 2, g.n) * (-1 if step > 20 else 1)
            learning = step % 4 != 0
            before = c.weights.astype(np.float64)
            reference.step(external, modulator=modulator, learning=learning)
            c.step(external, modulator=modulator, learning=learning)
            assert_allclose(c.state, reference.state, rtol=2e-6, atol=2e-7)
            assert_allclose(c.eligibility, reference.eligibility, rtol=3e-6, atol=2e-7)
            assert_allclose(c.weights, reference.weights, rtol=3e-6, atol=2e-7)
            expected_l1 = np.abs(c.weights.astype(np.float64) - before).sum()
            self.assertAlmostEqual(c.last_update_l1, expected_l1, places=14)
            self.assertIs(c.weights, weights_buffer)
            self.assertIs(c.eligibility, trace_buffer)
        for array in (c.state, c.eligibility, c.weights):
            self.assertEqual(array.dtype, np.float32)
        assert_array_equal(c.eligibility[~plastic], 0)
        assert_array_equal(c.weights[~plastic], reference.weights[~plastic].astype(np.float32))

    def test_recurrent_reduction_never_exceeds_configured_chunk(self):
        g = self.graph(compact=True)
        c = Circuit(g, dtype=np.float32, edge_chunk_size=4)
        original = np.bincount
        sizes = []

        def bounded_bincount(indices, *, weights, minlength):
            sizes.append(len(indices))
            self.assertLessEqual(len(indices), 4)
            self.assertEqual(len(indices), len(weights))
            return original(indices, weights=weights, minlength=minlength)

        with patch("fly_dcss.circuit.np.bincount", side_effect=bounded_bincount):
            c.step(np.ones(g.n), modulator=1)
        self.assertEqual(sum(sizes), len(g.indices))
        self.assertEqual(len(sizes), 5)

    def test_compact_bounds_and_actual_clipped_delta(self):
        g = self.graph(compact=True)
        c = Circuit(g, Parameters(learning_rate=1e30, weight_max=0.5),
                    dtype=np.float32, edge_chunk_size=2)
        for step in range(12):
            before = c.weights.astype(np.float64)
            c.step(np.ones(g.n), modulator=1e30 if step < 6 else -1e30)
            self.assertAlmostEqual(c.last_update_l1,
                                   np.abs(c.weights.astype(np.float64) - before).sum())
            for values in (c.state, c.eligibility, c.weights):
                self.assertTrue(np.all(np.isfinite(values)))
                self.assertTrue(np.all(values >= 0))
            self.assertTrue(np.all(c.state <= 1))
            self.assertTrue(np.all(c.eligibility <= 1))
            self.assertTrue(np.all(c.weights <= 0.5))
        c.step(np.ones(g.n), learning=False)
        self.assertEqual(c.last_update_l1, 0)
        c.step(np.ones(g.n), modulator=0)
        self.assertEqual(c.last_update_l1, 0)

    def test_compact_scaling_precedes_narrowing(self):
        g = Graph(np.array([0, 1]), np.array([0], dtype=np.int32),
                  np.array([1e100]), np.ones(1), np.array([10]))
        c = Circuit(g, Parameters(weight_scale=1e-100), dtype=np.float32)
        assert_array_equal(c.weights, [1])

    def test_compact_empty_graph_and_argument_validation(self):
        g = Graph(np.array([0, 0]), np.array([], dtype=np.int32),
                  np.array([], dtype=np.float32), np.ones(1), np.array([10]))
        c = Circuit(g, dtype=np.float32, edge_chunk_size=1)
        c.step([1], modulator=1)
        self.assertGreater(c.state[0], 0)
        self.assertEqual(c.last_update_l1, 0)
        for chunk in (0, -1, True, np.bool_(False), 1.5, "2"):
            with self.subTest(chunk=chunk), self.assertRaises(ValueError):
                Circuit(g, edge_chunk_size=chunk)
        for dtype in (np.float16, np.int32, np.complex64):
            with self.subTest(dtype=dtype), self.assertRaises(ValueError):
                Circuit(g, dtype=dtype)
        with self.assertRaisesRegex(ValueError, "weight_max"):
            Circuit(g, Parameters(weight_max=1e100), dtype=np.float32)
        for mask in (np.array([True]), np.array([], dtype=float)):
            with self.assertRaises(ValueError):
                Circuit(g, plastic=mask)

    def test_overflow_across_chunks_is_atomic(self):
        g = Graph(np.array([0, 1, 2, 2]), np.array([2, 2], dtype=np.int32),
                  np.ones(2), np.ones(3), np.arange(3))
        c = Circuit(g, Parameters(weight_scale=1e308, weight_max=1e308),
                    edge_chunk_size=1)
        c.state[:] = [1, 1, 0]
        old_weights = c.weights.copy()
        with self.assertRaisesRegex(ValueError, "current overflow"):
            c.step([0, 0, 0], modulator=1)
        assert_array_equal(c.state, [1, 1, 0])
        assert_array_equal(c.eligibility, 0)
        assert_array_equal(c.weights, old_weights)
        self.assertEqual(c.time, 0)
        self.assertEqual(c.last_update_l1, 0)


class AdapterTests(unittest.TestCase):
    def test_npz_roundtrip_and_reordered_induced_subgraph(self):
        g = synthetic_graph()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.npz"
            np.savez(path, **vars(g))
            subset = load_converted(path, body_ids=[103, 102])
        assert_array_equal(subset.body, [103, 102])
        assert_array_equal(subset.indptr, [0, 1, 2])
        assert_array_equal(subset.indices, [1, 0])
        assert_array_equal(subset.weight, [2, 3])
        assert_array_equal(subset.sign, [-1, 1])

    def test_edgeless_selection(self):
        g = synthetic_graph().subset([101])
        self.assertEqual(len(g.indices), 0)
        self.assertGreater(Circuit(g).step([1])[0], 0)

    def test_bad_graph_and_selection(self):
        g = synthetic_graph()
        for field, value in [("indptr", [0, 2, 1, 3]), ("indices", [1, 2, 8]),
                             ("weight", [4, -1, 2]), ("weight", [4, np.nan, 2]),
                             ("sign", [1, 0, -1]), ("body", [101, 101, 103]),
                             ("indices", [1., 2., 1.])]:
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                Graph(**(vars(g) | {field: value}))
        for ids in [[], [101, 101], [999]]:
            with self.assertRaises(ValueError):
                g.subset(ids)

    def test_compact_arrays_are_preserved_copied_and_read_only(self):
        source = dict(indptr=np.array([0, 2, 2], dtype=np.int32),
                      indices=np.array([0, 1], dtype=np.int32),
                      weight=np.array([2, 3], dtype=np.float32),
                      sign=np.array([1, -1], dtype=np.float32),
                      body=np.array([101, 102], dtype=np.int32))
        g = Graph(**source)
        self.assertEqual(g.indptr.dtype, np.int64)
        self.assertEqual(g.body.dtype, np.int64)
        self.assertEqual(g.indices.dtype, np.int32)
        self.assertEqual(g.weight.dtype, np.float32)
        self.assertEqual(g.pre.dtype, np.int32)
        for name, original in source.items():
            stored = getattr(g, name)
            self.assertFalse(np.shares_memory(original, stored))
            self.assertFalse(stored.flags.writeable)
        source["weight"][:] = 100
        source["indices"][:] = 0
        assert_array_equal(g.weight, [2, 3])
        assert_array_equal(g.indices, [0, 1])
        for ids in ([102, 101], [102]):
            subset = g.subset(ids)
            self.assertEqual(subset.indices.dtype, np.int32)
            self.assertEqual(subset.weight.dtype, np.float32)

    def test_missing_npz_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.npz"
            np.savez(path, weight=[1])
            with self.assertRaisesRegex(ValueError, "missing NPZ arrays"):
                load_converted(path, body_ids=[101])


if __name__ == "__main__":
    unittest.main()
