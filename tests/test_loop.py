import json
import unittest
import numpy as np
from fly_dcss.agent import LoopAgent, encode, permitted_actions
from fly_dcss.demo import synthetic_graph
from fly_dcss.webtiles import MessageDecoder, ObservationState, UnsafeInput, LocalConnection, UNSEEN, MM_UNSEEN, tile_bits


def ready_state():
    state = ObservationState()
    for message in [
        {"msg": "player", "hp": 20, "hp_max": 20, "mp": 0, "mp_max": 0,
         "turn": 0, "pos": {"x": 0, "y": 0}, "hidden": 42},
        {"msg": "input_mode", "mode": 1}, {"msg": "ui_state", "state": 0},
        {"msg": "map", "clear": True, "cells": [
            {"x": 0, "y": 0, "g": "@", "t": {"bg": 1}},
            {"g": ".", "t": {"bg": 1}, "mon": None}]}]:
        state.apply(message)
    return state


class ProtocolTests(unittest.TestCase):
    def test_fragments_and_server_control_prefix(self):
        decoder = MessageDecoder()
        raw = b'{"msg":"version","text":"0.17.1"}\n*{"msg":"flush_messages"}\n'
        messages = []
        for byte in raw:
            messages.extend(decoder.feed(bytes([byte])))
        self.assertEqual([m['msg'] for m in messages], ['version', 'flush_messages'])

    def test_delta_merge_and_privacy(self):
        state = ready_state()
        self.assertNotIn('hidden', state.player)
        state.apply({'msg': 'map', 'cells': [{'x': 1, 'y': 0, 'g': 'g', 'mon': {'typedata': {'avghp': 100}, 'id': 4}}]})
        self.assertTrue(state.cells[(1, 0)]['monster'])
        self.assertNotIn('typedata', json.dumps(state.snapshot()))
        state.apply({'msg': 'map', 'cells': [{'x': 1, 'y': 0, 't': {'bg': UNSEEN}}]})
        cell = next(c for c in state.snapshot()['cells'] if c['dx'] == 1 and c['dy'] == 0)
        self.assertEqual(cell['visibility'], 'remembered')
        self.assertFalse(cell['monster'])
        self.assertIn('unknown', {c['visibility'] for c in state.snapshot()['cells']})
        state.apply({'msg': 'map', 'clear': True})
        self.assertEqual(state.cells, {})

    def test_magical_mapping_is_not_current_sight(self):
        state = ready_state()
        state.apply({'msg': 'map', 'cells': [{'x': 1, 'y': 0, 'g': 'g',
                     't': {'bg': MM_UNSEEN}, 'mon': {'id': 2}}]})
        obs = state.snapshot()
        cell = next(c for c in obs['cells'] if c['dx'] == 1 and c['dy'] == 0)
        self.assertEqual(cell['visibility'], 'remembered')
        self.assertFalse(cell['monster'])
        self.assertEqual(permitted_actions(obs), ['wait'])

    def test_prompt_and_menu_guard(self):
        for message in [{'msg':'input_mode','mode':7}, {'msg':'menu'}, {'msg':'ui_state','state':1}, {'msg':'player','hp':0}]:
            state = ready_state()
            state.apply(message)
            self.assertFalse(state.ready)
            with self.assertRaises(UnsafeInput):
                state.snapshot()
            connection = LocalConnection.__new__(LocalConnection)
            connection.state = state
            connection.send = lambda _: self.fail('unsafe key sent')
            with self.assertRaises(UnsafeInput):
                connection.act('wait')

    def test_exact_protocol_version(self):
        state = ObservationState()
        state.apply({'msg': 'version', 'text': 'Dungeon Crawl Stone Soup 0.17.10'})
        self.assertNotEqual(state.version, '0.17.1')
        state.apply({'msg': 'version', 'text': 'Dungeon Crawl Stone Soup 0.17.1'})
        self.assertEqual(state.version, '0.17.1')

    def test_tile_bit_pairs(self):
        self.assertEqual(tile_bits([-1, 1]), 0x1ffffffff)
        self.assertEqual(tile_bits(UNSEEN) & UNSEEN, UNSEEN)

    def test_malformed_map_rejected(self):
        with self.assertRaises(ValueError):
            ObservationState().apply({'msg':'map', 'cells':[{'g':'.'}]})


class AgentTests(unittest.TestCase):
    def test_encoding_and_mask(self):
        obs = ready_state().snapshot()
        self.assertEqual(encode(obs).shape, (128,))
        self.assertEqual(set(permitted_actions(obs)), {'east','wait'})
        self.assertTrue(np.isfinite(encode(obs)).all())

    def test_continuous_action_feedback_loop(self):
        state = ready_state()
        agent = LoopAgent(synthetic_graph(), seed=7)
        initial = agent.circuit.weights.copy()
        before = state.snapshot()
        action, _ = agent.choose(before)
        self.assertIn(action, permitted_actions(before))
        state.apply({'msg':'player','turn':1})
        result = agent.feedback(before, state.snapshot())
        self.assertEqual(result['reward'], -0.01)
        self.assertGreater(result['weights_delta_l1'], 0)
        self.assertFalse(np.array_equal(initial, agent.circuit.weights))
        previous = agent.circuit.state.copy()
        agent.choose(state.snapshot())
        self.assertEqual(agent.circuit.time, 3)
        self.assertFalse(np.array_equal(previous, agent.circuit.state))
        self.assertTrue(np.isfinite(agent.circuit.weights).all())

    def test_seeded_reproducibility(self):
        observation = ready_state().snapshot()
        a, b = [LoopAgent(synthetic_graph(), seed=2) for _ in range(2)]
        self.assertEqual(a.choose(observation), b.choose(observation))


if __name__ == '__main__':
    unittest.main()

class CompactAgentTests(unittest.TestCase):
    def test_compact_mapping_and_bounded_feedback(self):
        from fly_dcss.circuit import Graph
        n = 64
        graph = Graph(np.arange(n + 1), np.roll(np.arange(n, dtype=np.int32), -1),
                      np.ones(n, dtype=np.float32), np.ones(n), np.arange(n))
        agent = LoopAgent(graph, compact=True)
        observation = ready_state().snapshot()
        external = agent.external(observation)
        self.assertEqual(external.shape, (n,))
        self.assertEqual(agent.projection.shape, (n, 4))
        self.assertEqual(agent.mapping_summary()['features_connected'], 128)
        self.assertTrue(np.allclose(agent.readout.sum(axis=1), 1))
        self.assertTrue(np.all(agent.readout > 0))
        agent.choose(observation)
        feedback = agent.feedback(observation, observation)
        self.assertNotIn('weights_before', feedback)
        self.assertNotIn('weights_after', feedback)
        self.assertNotIn('state', feedback)
        self.assertLess(len(json.dumps(feedback)), 1000)
        self.assertEqual(agent.circuit.state.dtype, np.dtype('float32'))

class CompactLogTests(unittest.TestCase):
    def test_finite_summary_is_json_safe(self):
        from fly_dcss.run_loop import finite_chunks, write_record
        import io
        self.assertTrue(finite_chunks(np.zeros(1_000_001, dtype=np.float32)))
        values = np.zeros(1_000_001, dtype=np.float32)
        values[-1] = np.nan
        self.assertFalse(finite_chunks(values))
        agent = LoopAgent(synthetic_graph(), compact=True)
        agent.external(ready_state().snapshot())
        stream = io.StringIO()
        write_record(stream, 'mapping', **agent.mapping_summary())
        self.assertEqual(json.loads(stream.getvalue())['kind'], 'mapping')

class FunctionalConnectivityTests(unittest.TestCase):
    def test_recurrent_pulse_reaches_all_readouts(self):
        from unittest.mock import patch
        from tools.check_connectome import check
        with patch('tools.check_connectome.load_malecns',
                   return_value=(synthetic_graph(), {'scope': 'synthetic'})):
            result = check('unused')
        self.assertTrue(result['all_five_readouts_receive_recurrent_signal'])
        self.assertGreater(result['downstream_active_nodes_excluding_source'], 0)
        self.assertLess(result['max_current_absolute_error'], 1e-10)
