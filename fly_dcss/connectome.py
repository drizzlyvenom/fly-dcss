"""Small structural subsets of the separately cached, CC BY 4.0 Fly Hero graph.

Graph.body holds SOURCE-LOCAL INDICES here, NOT biological MaleCNS bodyIds.
The upstream file omits bodyIds. Signs and neuron selection inherit its modeling
assumptions; this is a wiring fixture, not a validated fly decision circuit.
"""
from collections import deque
import hashlib
import json
from pathlib import Path
import numpy as np
from .circuit import Graph

SOURCE_COMMIT = "99cdd390193e20a0e85718951de2ea8eb4dd6555"
SOURCE_SHA256 = "6814cc3cc1a65e1e58f589f44deff69a410b3e2e92a95c4f1c0a7b49f4b73d4a"
SOURCE_URL = f"https://raw.githubusercontent.com/bsgelman/fly-hero/{SOURCE_COMMIT}/data/subgraph.json"


def _integers(values, name):
    a = np.asarray(values)
    if a.ndim != 1 or a.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a one-dimensional integer array")
    return a.astype(np.int64)


def load_flyhero(path, *, n=128, seed=0, expected_sha256=SOURCE_SHA256):
    """Return (Graph, provenance) from a local file; never downloads or vendors.

    Deterministic breadth-first traversal starts at source-local seed (default
    0, a visual-projection neuron). Outgoing neighbors are ordered by descending
    absolute synapse count, then source-local index. Keep all edges induced by
    the first n reached neurons. This preserves directed reachability from seed,
    not biological representativeness. Refuse undersized reachable components.

    Default checksum pins the known release; passing None is explicitly for
    tests or independently verified input, and provenance records that fact.
    Raw unsigned counts are retained, with a recommended global circuit scale
    limiting each neuron's initial absolute recurrent input to <=0.25.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 2:
        raise ValueError("n must be an integer >= 2")
    path = Path(path)
    if path.stat().st_size > 20_000_000:
        raise ValueError("fixture file exceeds 20 MB limit")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("fixture SHA256 does not match expected source")
    data = json.loads(raw)
    try:
        neurons, edges = data["neurons"], data["edges"]
        count = len(neurons["role"])
        if count < n or any(len(neurons[k]) != count for k in ("type", "tag", "x", "y")):
            raise ValueError("inconsistent neuron arrays or requested subset too large")
        pre = _integers(edges["pre"], "pre")
        post = _integers(edges["post"], "post")
        w = _integers(edges["w"], "w")
    except (KeyError, TypeError) as exc:
        raise ValueError("invalid Fly Hero schema") from exc
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < count:
        raise ValueError("seed must be a source-local neuron index")
    if (len(pre) != len(post) or len(pre) != len(w) or len(w) == 0
            or np.any((pre < 0) | (pre >= count))
            or np.any((post < 0) | (post >= count)) or np.any(w == 0)
            or np.any(w == np.iinfo(np.int64).min)):
        raise ValueError("invalid edge arrays, endpoints or signed counts")
    signs = np.zeros(count, dtype=np.int8)
    signs[pre[w > 0]] = 1
    if np.any(signs[pre[w < 0]] == 1):
        raise ValueError("mixed signs for one presynaptic neuron")
    signs[pre[w < 0]] = -1
    # Unique pairs are necessary to preserve edge-level plasticity semantics.
    pairs = pre * count + post
    if len(np.unique(pairs)) != len(pairs):
        raise ValueError("duplicate directed pairs")
    order = np.lexsort((post, -np.abs(w), pre))
    ptr = np.r_[0, np.cumsum(np.bincount(pre, minlength=count))]
    selected, seen, queue = [seed], {seed}, deque([seed])
    while queue and len(selected) < n:
        current = queue.popleft()
        for edge in order[ptr[current]:ptr[current + 1]]:
            target = int(post[edge])
            if target not in seen:
                seen.add(target)
                selected.append(target)
                queue.append(target)
                if len(selected) == n:
                    break
    if len(selected) < n:
        raise ValueError("seed's outgoing reachable component is smaller than n")
    selected = np.asarray(selected, dtype=np.int64)
    remap = np.full(count, -1, dtype=np.int64)
    remap[selected] = np.arange(n)
    keep = (remap[pre] >= 0) & (remap[post] >= 0)
    rp, rq, rw = remap[pre[keep]], remap[post[keep]], np.abs(w[keep])
    order = np.lexsort((rq, rp))
    rp, rq, rw = rp[order], rq[order], rw[order]
    selected_signs = signs[selected].copy()
    unknown = selected[selected_signs == 0].tolist()
    selected_signs[selected_signs == 0] = 1  # no outgoing edges, so no effect
    graph = Graph(np.r_[0, np.cumsum(np.bincount(rp, minlength=n))],
                  rq, rw, selected_signs, selected)
    max_incoming = float(np.bincount(rq, weights=rw, minlength=n).max())
    verified = digest == SOURCE_SHA256
    provenance = {
        "dataset": "MaleCNS v1.0 via bsgelman/fly-hero" if verified else "unverified Fly Hero-schema input",
        "verified_pinned_source": verified,
        "source_url": SOURCE_URL if verified else None,
        "source_commit": SOURCE_COMMIT if verified else None,
        "source_sha256": digest,
        "license": "CC BY 4.0" if verified else "not verified",
        "license_url": "https://creativecommons.org/licenses/by/4.0/" if verified else None,
        "license_evidence": "https://male-cns.janelia.org/" if verified else None,
        "attribution": "FlyEM (HHMI Janelia), University of Cambridge Zoology, MRC LMB, Google Research; extraction by bsgelman/fly-hero" if verified else None,
        "id_namespace": "flyhero-source-local-index; NOT biological bodyIds",
        "selection": "directed breadth-first; descending count then index; induced edges",
        "seed_source_local_index": seed,
        "source_local_indices": selected.tolist(),
        "roles": [neurons["role"][i] for i in selected],
        "types": [neurons["type"][i] for i in selected],
        "sign_unknown_no_outgoing_indices": unknown,
        "neurons": n, "edges": len(rw), "synapse_count_sum": int(rw.sum()),
        "recommended_weight_scale": 0.25 / max_incoming,
        "transformations": ["Selected a small directed-reachable induced subset",
                            "Split signed counts into magnitude and presynaptic sign"],
        "limitations": ["Source omits biological bodyIds",
                        "Source removed VNC-to-brain edges",
                        "Source infers inhibitory sign for GABA/glutamate/histamine; other/unknown transmitters excitatory",
                        "Game inputs, outputs and learning remain engineering choices"],
    }
    return graph, provenance
