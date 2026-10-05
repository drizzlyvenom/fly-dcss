from pathlib import Path
import tempfile
import unittest

import numpy as np

from fly_dcss.malecns import SOURCES, download_sources, load_malecns, prepare_malecns

try:
    import pyarrow as pa
    import pyarrow.feather as feather
except ImportError:
    pa = feather = None


@unittest.skipIf(pa is None, "optional conversion dependency pyarrow is not installed")
class MaleCNSTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.raw = self.root / "raw"
        self.raw.mkdir()
        self.out = self.root / "converted"
        self.write("annotations", {
            "bodyId": pa.array([1001, 1002, 1003, 1004, 1005], type=pa.int64()),
            "superclass": ["descending_neuron", None, "vnc_motor", None, "cb_intrinsic"],
            "status": ["Traced", "Traced", "Traced", "Glia", "Traced"],
        })
        self.write("neurotransmitters", {
            "body": pa.array([1001, 1002, 1003, 1004], type=pa.int64()),
            "consensus_nt": ["gaba", "glutamate", "dopamine", None],
        })
        self.write("connections", {
            "body_pre": pa.array([1001, 1001, 1002, 1003, 1001, 1004, 9999,
                                  1001, 1001, 1002, 0, None, 1003], type=pa.int64()),
            "body_post": pa.array([1002, 1002, 1003, 1001, 1001, 1001, 1001,
                                   1003, 1003, 1001, 1001, 1001, 1002], type=pa.int64()),
            "weight": pa.array([1, 2, 4, 5, 1, 3, 2, 0, -1, None, 1, 1, 7], type=pa.int64()),
        })

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, key, columns):
        feather.write_feather(pa.table(columns), self.raw / SOURCES[key]["filename"],
                              chunksize=3)

    def convert(self):
        return prepare_malecns(self.raw, self.out, scope="full-annotated", verify_source=False)

    def test_all_selected_nodes_and_positive_rows_preserved(self):
        metadata = self.convert()
        graph, loaded = load_malecns(self.out)
        np.testing.assert_array_equal(graph.body, [1001, 1002, 1003, 1005])
        np.testing.assert_array_equal(graph.indptr, [0, 3, 4, 6, 6])
        np.testing.assert_array_equal(graph.indices, [1, 1, 0, 2, 0, 1])
        np.testing.assert_array_equal(graph.weight, [1, 2, 1, 4, 5, 7])
        np.testing.assert_array_equal(graph.sign, [-1, -1, 1, 1])
        self.assertEqual(metadata["nodes"], 4)
        self.assertEqual(metadata["edges"], 6)
        self.assertEqual(metadata["retained_duplicate_pair_rows"], 1)
        self.assertEqual(metadata["retained_self_loop_rows"], 1)
        self.assertEqual(metadata["retained_weight_one_rows"], 2)
        self.assertEqual(metadata["retained_weight_two_rows"], 1)
        self.assertEqual(metadata["zero_indegree_nodes"], 1)
        self.assertEqual(metadata["zero_outdegree_nodes"], 1)
        self.assertEqual(metadata["omitted_status_traced_records"], 0)
        self.assertEqual(metadata["selected_superclass_counts"]["unassigned"], 1)
        self.assertEqual(metadata["provisional_unknown_sign_nodes"], 1)
        self.assertFalse(loaded["verified_pinned_source"])
        self.assertIsNone(loaded["source_node_union"])
        self.assertTrue(loaded["verified_source_csr_spot_checks"])

    def test_omissions_are_disjoint_and_explicit(self):
        m = self.convert()
        self.assertEqual(m["source_connection_rows"], 13)
        self.assertEqual(m["out_of_scope_connection_rows"], 2)
        self.assertEqual(m["invalid_endpoint_rows"], 2)
        self.assertEqual(m["invalid_or_nonpositive_weight_rows"], 3)
        self.assertEqual(m["source_positive_synapse_count_sum"], 25)
        self.assertEqual(m["retained_synapse_count_sum"], 20)
        self.assertEqual(m["omitted_synapse_count_sum"], 5)
        self.assertEqual(m["omitted_annotation_records"], 1)

    def test_compact_storage(self):
        self.convert()
        expected = {"body": "int32", "indices": "int32", "indptr": "int32",
                    "sign": "int8", "weight": "float32"}
        for name, dtype in expected.items():
            self.assertEqual(str(np.load(self.out / f"{name}.npy").dtype), dtype)

    def test_cached_array_corruption_is_rejected(self):
        self.convert()
        array = np.load(self.out / "weight.npy")
        array[0] += 1
        np.save(self.out / "weight.npy", array)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            load_malecns(self.out)

    def test_unverified_source_rejected_by_default(self):
        with self.assertRaisesRegex(ValueError, "checksum"):
            prepare_malecns(self.raw, self.out, scope="full-annotated")

    def test_scope_is_required_and_unknown_scope_rejected(self):
        with self.assertRaises(TypeError):
            prepare_malecns(self.raw, self.out)
        with self.assertRaisesRegex(ValueError, "scope"):
            prepare_malecns(self.raw, self.out, scope="all")

    def test_existing_output_is_not_overwritten(self):
        self.convert()
        with self.assertRaisesRegex(ValueError, "empty"):
            self.convert()

    def test_duplicate_annotation_ids_rejected(self):
        self.write("annotations", {"bodyId": pa.array([1001, 1001], type=pa.int64()),
                                   "superclass": ["cb_intrinsic"] * 2,
                                   "status": ["Traced"] * 2})
        with self.assertRaisesRegex(ValueError, "duplicate annotation"):
            self.convert()

    def test_empty_neuron_scope_rejected(self):
        self.write("annotations", {"bodyId": pa.array([1001], type=pa.int64()),
                                   "superclass": [None], "status": ["Glia"]})
        with self.assertRaisesRegex(ValueError, "empty"):
            self.convert()

    def test_download_budget_rejected_before_network(self):
        with self.assertRaisesRegex(ValueError, "budget"):
            download_sources(self.root / "new", max_download_bytes=1)

    def test_existing_source_not_silently_overwritten(self):
        with self.assertRaisesRegex(ValueError, "existing source"):
            download_sources(self.raw)

    def test_huge_positive_weights_rejected_before_overflow(self):
        self.write("connections", {"body_pre": pa.array([1001, 1001], type=pa.int64()),
                                   "body_post": pa.array([1002, 1002], type=pa.int64()),
                                   "weight": pa.array([2**62, 2**62], type=pa.int64())})
        with self.assertRaisesRegex(ValueError, "float32 integer range"):
            self.convert()

    def test_unsigned_ids_outside_int64_rejected(self):
        self.write("connections", {"body_pre": pa.array([2**63 + 1], type=pa.uint64()),
                                   "body_post": pa.array([1002], type=pa.int64()),
                                   "weight": pa.array([1], type=pa.int64())})
        with self.assertRaisesRegex(ValueError, "signed int64"):
            self.convert()


if __name__ == "__main__":
    unittest.main()
