import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from fly_dcss.connectome import load_flyhero


class ConnectomeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'tiny.json'
        self.data = {'neurons': {'role': ['vpn', 'kc', 'apl', 'dn'],
                                'type': ['a', 'b', 'c', 'd'], 'tag': [0]*4,
                                'x': [0]*4, 'y': [0]*4},
                     'edges': {'pre': [0, 0, 1, 2], 'post': [1, 2, 3, 1],
                               'w': [2, 5, 4, -3]}}

    def tearDown(self):
        self.tmp.cleanup()

    def load(self, **kwargs):
        self.path.write_text(json.dumps(self.data))
        return load_flyhero(self.path, expected_sha256=None, **kwargs)

    def test_selection_and_orientation(self):
        g, p = self.load(n=3)
        np.testing.assert_array_equal(g.body, [0, 2, 1])
        np.testing.assert_array_equal(g.sign, [1, -1, 1])
        np.testing.assert_array_equal(g.indptr, [0, 2, 3, 3])
        np.testing.assert_array_equal(g.indices, [1, 2, 2])
        np.testing.assert_array_equal(g.weight, [5, 2, 3])
        self.assertIn('NOT biological', p['id_namespace'])
        self.assertFalse(p['verified_pinned_source'])
        self.assertAlmostEqual(p['recommended_weight_scale'], .05)

    def test_checksum_required_by_default(self):
        self.load(n=3)
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            load_flyhero(self.path, n=3)

    def test_mixed_signs_rejected(self):
        self.data['edges']['w'][0] = -2
        with self.assertRaisesRegex(ValueError, 'mixed signs'):
            self.load(n=3)

    def test_disconnected_request_rejected(self):
        with self.assertRaisesRegex(ValueError, 'reachable'):
            self.load(n=3, seed=3)

    def test_bad_endpoint_rejected(self):
        self.data['edges']['post'][0] = 9
        with self.assertRaisesRegex(ValueError, 'edge arrays'):
            self.load(n=3)

    def test_duplicate_pairs_rejected(self):
        self.data['edges']['post'][0] = 2
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.load(n=3)

    def test_noninteger_count_rejected(self):
        self.data['edges']['w'][0] = 1.5
        with self.assertRaisesRegex(ValueError, 'integer'):
            self.load(n=3)


if __name__ == '__main__':
    unittest.main()
