"""Verified MaleCNS v1.0 tables -> complete explicitly selected neuron graph.

The raw release also contains millions of segmentation fragments. A neuron
scope is always explicit; no synapse-weight threshold or CNS boundary cut is
applied within that scope. See docs/MALECNS_DATA.md for the scientific limits.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import urllib.request

try:
    import resource
except ImportError:  # Windows can load/prepare data without Unix RSS reporting.
    resource = None

import numpy as np

from .circuit import Graph

BASE_URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
SOURCES = {
    "annotations": {
        "filename": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
        "bytes": 14483314,
        "sha256": "2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2",
        "md5": "50a7718770c57220f160ba4f431ab89e",
        "generation": "1780494878811468",
    },
    "neurotransmitters": {
        "filename": "body-neurotransmitters-male-cns-v1.0.feather",
        "bytes": 43282834,
        "sha256": "95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621",
        "md5": "3d842b12fe5c49eefade528d7dd24a1f",
        "generation": "1780894899156750",
    },
    "connections": {
        "filename": "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
        "bytes": 1051241946,
        "sha256": "e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1",
        "md5": "f30e9dcca25cfd021bf1e7b3d975599e",
        "generation": "1780494887545976",
    },
}
# Independently measured from the SHA256-pinned files, including annotations,
# NT-only IDs and every raw connection endpoint. These are NOT neuron counts.
PINNED_SOURCE_NODE_UNION = 88404403
ARRAYS = ("indptr", "indices", "weight", "sign", "body")
SCOPES = ("full-annotated",)
INHIBITORY = {"gaba", "glutamate", "histamine"}
KNOWN_NT = INHIBITORY | {"acetylcholine", "dopamine", "serotonin", "octopamine"}


def _digests(path):
    sha, md5 = hashlib.sha256(), hashlib.md5(usedforsecurity=False)
    with Path(path).open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            sha.update(block)
            md5.update(block)
    return {"sha256": sha.hexdigest(), "md5": md5.hexdigest(),
            "bytes": Path(path).stat().st_size}


def download_sources(directory, *, max_download_bytes=1_200_000_000,
                     timeout=900):
    """Download only three pinned official files, bounded by bytes and time.

    Existing files must match SHA256 and MD5 before reuse. Never silently
    replace an unexpected existing file. Partial downloads use .part files.
    No source data is written into this package or downloaded by the loader.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    needed = sum(s["bytes"] for s in SOURCES.values()
                 if not (directory / s["filename"]).exists())
    if needed > max_download_bytes:
        raise ValueError(f"download requires {needed} bytes, over the budget")
    deadline = time.monotonic() + timeout
    for source in SOURCES.values():
        path = directory / source["filename"]
        if path.exists():
            actual = _digests(path)
            if any(actual[k] != source[k] for k in actual):
                raise ValueError(f"existing source checksum/size mismatch: {path}")
            continue
        partial = path.with_suffix(path.suffix + ".part")
        written = 0
        if time.monotonic() >= deadline:
            raise TimeoutError("MaleCNS download time budget exhausted")
        request = urllib.request.Request(BASE_URL + source["filename"])
        with urllib.request.urlopen(request, timeout=min(60, timeout)) as response:
            if response.headers.get("Content-Length") not in (None, str(source["bytes"])):
                raise ValueError("official source Content-Length changed")
            with partial.open("wb") as handle:
                while block := response.read(1024 * 1024):
                    written += len(block)
                    if written > source["bytes"] or time.monotonic() >= deadline:
                        raise ValueError("download exceeded byte/time budget")
                    handle.write(block)
        actual = _digests(partial)
        if any(actual[k] != source[k] for k in actual):
            raise ValueError(f"download checksum/size mismatch: {partial}")
        partial.replace(path)
    return {key: directory / source["filename"] for key, source in SOURCES.items()}


def _batches(path, columns):
    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc
    except ImportError as exc:
        raise ImportError("Conversion needs pyarrow; install pyarrow>=14,<24") from exc
    with pa.memory_map(str(path), "r") as handle:
        reader = ipc.open_file(handle)
        missing = set(columns) - set(reader.schema.names)
        if missing:
            raise ValueError(f"missing source columns: {sorted(missing)}")
        for i in range(reader.num_record_batches):
            batch = reader.get_batch(i)
            yield batch.select(columns)


def _integer_column(batch, name):
    column = batch.column(name)
    # Source has int64 IDs/counts; nulls are represented separately, not cast
    # through float64 (which could lose precision in another release).
    import pyarrow as pa
    if not pa.types.is_integer(column.type):
        raise ValueError(f"{name} must be an integer source column")
    valid = column.is_valid().to_numpy(zero_copy_only=False)
    values = column.fill_null(0).to_numpy(zero_copy_only=False)
    if values.dtype.kind == "u" and np.any(values > np.iinfo(np.int64).max):
        raise ValueError(f"{name} contains values outside signed int64 range")
    return values.astype(np.int64, copy=False), valid


def _positions(body, ids):
    position = np.searchsorted(body, ids)
    keep = position < len(body)
    keep[keep] &= body[position[keep]] == ids[keep]
    return position, keep


def _edges(path, body):
    for batch in _batches(path, ["body_pre", "body_post", "weight"]):
        pre, vp = _integer_column(batch, "body_pre")
        post, vq = _integer_column(batch, "body_post")
        weight, vw = _integer_column(batch, "weight")
        good_weight = vw & (weight > 0)
        # Reject unsupported inputs before arithmetic. Every integer through
        # 2**24 is exactly representable in float32; release counts max at 2591.
        if np.any(weight[good_weight] > 2**24):
            raise ValueError("synapse count exceeds exact float32 integer range")
        if len(weight) > np.iinfo(np.int64).max // 2**24:
            raise ValueError("source batch could overflow int64 synapse totals")
        good_ids = vp & vq & (pre > 0) & (post > 0)
        valid = good_weight & good_ids
        p, kp = _positions(body, pre)
        q, kq = _positions(body, post)
        keep = valid & kp & kq
        stats = {
            "source_connection_rows": len(pre),
            "invalid_or_nonpositive_weight_rows": int((~good_weight).sum()),
            "invalid_endpoint_rows": int((good_weight & ~good_ids).sum()),
            "out_of_scope_connection_rows": int((valid & ~keep).sum()),
            "source_positive_synapse_count_sum": int(weight[valid].sum(dtype=np.int64)),
            "retained_synapse_count_sum": int(weight[keep].sum(dtype=np.int64)),
            "retained_weight_one_rows": int((keep & (weight == 1)).sum()),
            "retained_weight_two_rows": int((keep & (weight == 2)).sum()),
            "retained_self_loop_rows": int((keep & (pre == post)).sum()),
        }
        yield p[keep], q[keep], weight[keep], stats


def prepare_malecns(raw_directory, output_directory, *, scope,
                    verify_source=True, progress=None):
    """Stream the full release into a CSR directory; never use an edge list.

    full-annotated: every annotation with a nonempty superclass OR status=Traced
    (167,216 for pinned v1.0). This retains the 516 traced records that have no
    superclass. It keeps all valid positive rows
    between selected IDs, including weights 1/2, self-loops and duplicate rows.
    The argument has no default so neuron selection cannot be implicit.
    """
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}")
    started = time.monotonic()
    raw_directory, output_directory = Path(raw_directory), Path(output_directory)
    if output_directory.exists() and any(output_directory.iterdir()):
        raise ValueError("output directory must be absent or empty")
    output_directory.mkdir(parents=True, exist_ok=True)
    paths, source_records = {}, {}
    for key, source in SOURCES.items():
        path = raw_directory / source["filename"]
        actual = _digests(path)
        verified = all(actual[k] == source[k] for k in actual)
        if verify_source and not verified:
            raise ValueError(f"source checksum/size mismatch: {path}")
        paths[key] = path
        source_records[key] = {**actual, "filename": source["filename"],
                               "verified": verified,
                               "url": BASE_URL + source["filename"] if verified else None,
                               "generation": source["generation"] if verified else None}
    pinned = all(s["verified"] for s in source_records.values())
    chosen, all_annotation_ids = [], []
    statuses, chosen_statuses, classes = Counter(), Counter(), Counter()
    excluded_traced = 0
    for batch in _batches(paths["annotations"], ["bodyId", "superclass", "status"]):
        ids, valid = _integer_column(batch, "bodyId")
        if not valid.all() or np.any(ids <= 0):
            raise ValueError("annotation body IDs must be positive and non-null")
        superclass, status = batch.column("superclass").to_pylist(), batch.column("status").to_pylist()
        select = np.array([bool(c) or s == "Traced"
                           for c, s in zip(superclass, status)])
        all_annotation_ids.append(ids.copy())
        chosen.append(ids[select])
        statuses.update(str(s) if s is not None else "unassigned" for s in status)
        chosen_statuses.update(str(s) if s is not None else "unassigned"
                               for s, k in zip(status, select) if k)
        classes.update(str(c) if c else "unassigned"
                       for c, k in zip(superclass, select) if k)
        excluded_traced += sum(s == "Traced" and not k for s, k in zip(status, select))
    all_annotation_ids = np.concatenate(all_annotation_ids)
    if len(np.unique(all_annotation_ids)) != len(all_annotation_ids):
        raise ValueError("duplicate annotation body IDs")
    body = np.sort(np.concatenate(chosen))
    if not len(body):
        raise ValueError("selected neuron scope is empty")
    body = body.astype(np.int32 if body[-1] <= np.iinfo(np.int32).max else np.int64)
    n = len(body)
    if n > np.iinfo(np.int32).max:
        raise ValueError("node count exceeds int32 CSR index capacity")
    degree, incoming = np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64)
    counts = Counter()
    min_weight, max_weight = None, 0
    if progress:
        progress(f"Counting all source rows for {n:,} selected nodes")
    for pre, post, weight, stats in _edges(paths["connections"], body):
        if (counts["source_positive_synapse_count_sum"] +
                stats["source_positive_synapse_count_sum"] > np.iinfo(np.int64).max):
            raise ValueError("source synapse total exceeds supported int64 range")
        np.add.at(degree, pre, 1)
        np.add.at(incoming, post, weight)
        counts.update(stats)
        if len(weight):
            min_weight = min(int(weight.min()), min_weight) if min_weight is not None else int(weight.min())
            max_weight = max(max_weight, int(weight.max()))
    edge_count = int(degree.sum())
    pointer_dtype = np.int32 if edge_count <= np.iinfo(np.int32).max else np.int64
    ptr = np.empty(n + 1, dtype=pointer_dtype)
    ptr[0] = 0
    np.cumsum(degree, out=ptr[1:])
    indices = np.lib.format.open_memmap(output_directory / "indices.npy", mode="w+",
                                       dtype=np.int32, shape=(edge_count,))
    weights = np.lib.format.open_memmap(output_directory / "weight.npy", mode="w+",
                                       dtype=np.float32, shape=(edge_count,))
    cursor = ptr[:-1].copy()
    if progress:
        progress(f"Writing {edge_count:,} rows, retaining weights 1 and 2")
    spot_checks = []
    for batch_number, (pre, post, weight, _) in enumerate(_edges(paths["connections"], body)):
        # Group only one Arrow batch at a time; stable ordering preserves each
        # source row, including repeated directed pairs, within its CSR row.
        order = np.argsort(pre, kind="stable")
        pre, post, weight = pre[order], post[order], weight[order]
        rows, starts, row_counts = np.unique(pre, return_index=True, return_counts=True)
        target = np.repeat(cursor[rows] - starts, row_counts) + np.arange(len(pre))
        converted = weight.astype(np.float32)
        if not np.array_equal(converted.astype(np.int64), weight):
            raise ValueError("synapse count cannot be represented exactly in float32")
        indices[target], weights[target] = post, converted
        cursor[rows] += row_counts
        if len(pre) and batch_number % 64 == 0:
            for k in sorted({0, len(pre) - 1}):
                spot_checks.append({"source_batch": batch_number,
                                    "body_pre": int(body[pre[k]]),
                                    "body_post": int(body[post[k]]),
                                    "weight": int(weight[k]),
                                    "csr_edge": int(target[k]),
                                    "pre_index": int(pre[k])})
    if not np.array_equal(cursor, ptr[1:]):
        raise ValueError("CSR edge fill did not match counted degrees")
    indices.flush()
    weights.flush()
    for sample in spot_checks:
        e, p = sample["csr_edge"], sample["pre_index"]
        if not (ptr[p] <= e < ptr[p + 1] and body[p] == sample["body_pre"]
                and body[indices[e]] == sample["body_post"] and weights[e] == sample["weight"]):
            raise ValueError("source direction/weight spot-check failed")
    sign = np.ones(n, dtype=np.int8)
    nt_seen = np.zeros(n, dtype=bool)
    nt_labels = Counter()
    for batch in _batches(paths["neurotransmitters"], ["body", "consensus_nt"]):
        ids, valid = _integer_column(batch, "body")
        pos, selected = _positions(body, ids)
        selected &= valid
        use = np.flatnonzero(selected)
        selected_pos = pos[use]
        if nt_seen[selected_pos].any() or len(np.unique(selected_pos)) != len(selected_pos):
            raise ValueError("duplicate neurotransmitter row for selected node")
        labels = batch.column("consensus_nt").take(use).to_pylist()
        for node, label in zip(selected_pos, labels):
            label = str(label).strip().lower() if label else "unknown"
            nt_labels[label] += 1
            sign[node] = -1 if label in INHIBITORY else 1
        nt_seen[selected_pos] = True
    nt_labels["missing_row"] = int((~nt_seen).sum())
    duplicates = 0
    # Check retained-pair multiplicity without constructing 64-bit pair keys
    # for the entire graph. Duplicate rows are counted, not coalesced/dropped.
    for row in range(n):
        start, end = int(ptr[row]), int(ptr[row + 1])
        if end - start > 1:
            duplicates += end - start - len(np.unique(indices[start:end]))
    for key, array in {"indptr": ptr, "body": body, "sign": sign}.items():
        np.save(output_directory / f"{key}.npy", array, allow_pickle=False)
    arrays = {key: {**_digests(output_directory / f"{key}.npy"),
                    "dtype": str(np.load(output_directory / f"{key}.npy", mmap_mode="r").dtype)}
              for key in ARRAYS}
    omitted_synapses = counts["source_positive_synapse_count_sum"] - counts["retained_synapse_count_sum"]
    unknown_sign_count = sum(v for k, v in nt_labels.items() if k not in KNOWN_NT)
    max_incoming = int(incoming.max(initial=0))
    metadata = {
        "format": "fly-dcss-malecns-csr-v1",
        "dataset": "MaleCNS v1.0" if pinned else "unverified MaleCNS-schema fixture",
        "verified_pinned_source": pinned,
        "release_date": "2026-06-08" if pinned else None,
        "source_page": "https://male-cns.janelia.org/download/" if pinned else None,
        "license": "CC BY 4.0" if pinned else "unverified",
        "license_url": "https://creativecommons.org/licenses/by/4.0/" if pinned else None,
        "attribution": ("MaleCNS v1.0; FlyEM (HHMI Janelia), University of Cambridge "
                        "Department of Zoology, MRC Laboratory of Molecular Biology, Google Research") if pinned else None,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sources": source_records, "arrays": arrays,
        "scope": scope,
        "selection": "nonempty annotation superclass OR status=Traced",
        "id_namespace": "official MaleCNS biological/segmentation bodyId",
        "nodes": n, "edges": edge_count,
        "source_node_union": PINNED_SOURCE_NODE_UNION if pinned else None,
        "source_node_union_definition": "unique IDs across annotations, neurotransmitters and raw connection endpoints; mostly fragments, not neurons",
        "omitted_source_nodes": PINNED_SOURCE_NODE_UNION - n if pinned else None,
        "source_nodes_without_annotation_record": PINNED_SOURCE_NODE_UNION - len(all_annotation_ids) if pinned else None,
        "source_annotation_records": len(all_annotation_ids),
        "omitted_annotation_records": len(all_annotation_ids) - n,
        "omitted_status_traced_records": excluded_traced,
        "annotation_status_counts": dict(statuses),
        "selected_status_counts": dict(chosen_statuses),
        "selected_superclass_counts": dict(classes),
        **dict(counts),
        "omitted_synapse_count_sum": omitted_synapses,
        "synapse_count_sum": counts["retained_synapse_count_sum"],
        "retained_duplicate_pair_rows": duplicates,
        "zero_indegree_nodes": int((incoming == 0).sum()),
        "zero_outdegree_nodes": int((degree == 0).sum()),
        "min_retained_weight": min_weight, "max_retained_weight": max_weight,
        "verified_source_csr_spot_checks": spot_checks,
        "neurotransmitter_consensus_counts": dict(nt_labels),
        "inhibitory_nodes": int((sign < 0).sum()),
        "provisional_unknown_sign_nodes": unknown_sign_count,
        "max_incoming_synapse_count": max_incoming,
        "recommended_weight_scale": 0.25 / max_incoming if max_incoming else 0.0,
        "direction": "CSR row=body_pre; indices=body_post; source direction unchanged",
        "weights": "unscaled positive integer synapse counts stored exactly as float32",
        "sign_policy": "consensus_nt gaba/glutamate/histamine -> -1; all others including unknown -> +1; modeling assumption, not measured physiology",
        "transformations": ["Explicit neuron scope selection; original body IDs retained",
                            "All positive inter-scope rows retained without count threshold",
                            "No brain/VNC boundary filtering, rewiring, sampling or duplicate coalescing",
                            "Presynaptic sign inferred from consensus neurotransmitter by an engineering rule"],
        "limitations": ["Raw fragment graph is larger than the annotated-neuron graph",
                        "Annotation completeness and neurotransmitter consensus are not ground truth for every node",
                        "Glutamate and modulatory effects can depend on receptors and circuit context",
                        "Game ports, rate dynamics and plasticity are engineered hypotheses, not validated fly behavior"],
        "conversion_elapsed_seconds": time.monotonic() - started,
        "conversion_peak_rss_kib": (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss /
                                    (1024 if __import__("sys").platform == "darwin" else 1)) if resource else None,
    }
    (output_directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def load_malecns(path, *, verify_arrays=True):
    """Return (Graph, provenance) for the entire prepared scope, no downloads.

    NPY files are mapped read-only; Graph performs the one validated owning
    copy. Conversion should run in a separate process before a full episode.
    """
    path = Path(path)
    metadata = json.loads((path / "metadata.json").read_text())
    if metadata.get("format") != "fly-dcss-malecns-csr-v1":
        raise ValueError("unsupported MaleCNS cache format")
    arrays = {}
    for key in ARRAYS:
        filename = path / f"{key}.npy"
        if verify_arrays and _digests(filename)["sha256"] != metadata["arrays"][key]["sha256"]:
            raise ValueError(f"cached array SHA256 mismatch: {key}")
        arrays[key] = np.load(filename, mmap_mode="r", allow_pickle=False)
    graph = Graph(**arrays)
    if graph.n != metadata["nodes"] or len(graph.indices) != metadata["edges"]:
        raise ValueError("cached graph dimensions disagree with metadata")
    return graph, metadata
