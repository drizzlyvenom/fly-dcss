import tempfile
import unittest
from pathlib import Path
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

    def test_missing_npz_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.npz"
            np.savez(path, weight=[1])
            with self.assertRaisesRegex(ValueError, "missing NPZ arrays"):
                load_converted(path, body_ids=[101])


if __name__ == "__main__":
    unittest.main()
