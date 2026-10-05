# MaleCNS source data and complete annotated/traced graph

## What "full" means here

The runnable graph has **167,216 modeled annotated/traced nodes and 25,587,572
directed connection rows**. It includes every v1.0 annotation whose `superclass`
is nonempty **or** whose `status` is `Traced`, then retains every positive source
connection between those IDs. There is no threshold of three synapses, sampled
subgraph, brain/VNC boundary removal, reverse-edge insertion, or rewiring.

This is the full selected annotated/traced graph, **not** the complete raw
segmentation graph, and not proof that every selected record is a biologically
confirmed neuron. The 166,700 classified records are supplemented by 516 traced
records without a superclass. All 165,122 `status=Traced` records are included;
the remaining selected statuses are 2,002 unassigned, 60 Anchor, and 32 Orphan.
No isolated selected node is discarded.

The official raw tables contain **88,404,403 distinct body IDs**, mostly
segmentation fragments. They must not be described as 88 million fly neurons.
That union includes all annotation IDs, all neurotransmitter-table IDs, and all
connection endpoints. It was independently counted using a packed body-ID
bitmap on the exact SHA256-pinned source files below; the converter only reports
this pinned audit value when all three source hashes match. The ID maximum was
1,571,863,634. The raw fragment graph was not simulated.

## Official source, license and credit

Release: **MaleCNS v1.0, June 8, 2026**, verified against the official
[release notes](https://male-cns.janelia.org/release/) and
[download page](https://male-cns.janelia.org/download/) on October 5, 2026.
The download page describes the connection-strength file as the complete
segment-to-segment graph. These files already use the release's `minconf-0.5`
synapse-confidence selection; this converter cannot restore data omitted
upstream. It introduces no additional synapse-confidence filter.

Dataset attribution: **MaleCNS v1.0; FlyEM at HHMI Janelia Research Campus,
University of Cambridge Department of Zoology, MRC Laboratory of Molecular
Biology, and Google Research.**

The official site links the dataset to
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/),
with its [legal code](https://creativecommons.org/licenses/by/4.0/legalcode.en).
Retain appropriate attribution, the source and license links, and an indication
of changes when sharing derived data; do not imply endorsement. This project's
changes are explicit node selection, CSR storage, and an inferred sign policy.
The dataset license does not assign a license to this repository's code.

Related publication: Berg et al., *Sexual dimorphism in the complete Drosophila
male central nervous system connectome*, Cell (2026),
[doi:10.1016/j.cell.2026.08.015](https://doi.org/10.1016/j.cell.2026.08.015).

Official HTTPS prefix:

```text
https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/
```

The three files total **1,109,008,094 bytes**. No data or generated binary arrays
are committed to git.

| File | Bytes | GCS generation | Server MD5, verified locally |
| --- | ---: | --- | --- |
| body-annotations-male-cns-v1.0-minconf-0.5.feather | 14,483,314 | 1780494878811468 | 50a7718770c57220f160ba4f431ab89e |
| body-neurotransmitters-male-cns-v1.0.feather | 43,282,834 | 1780894899156750 | 3d842b12fe5c49eefade528d7dd24a1f |
| connectome-weights-male-cns-v1.0-minconf-0.5.feather | 1,051,241,946 | 1780494887545976 | f30e9dcca25cfd021bf1e7b3d975599e |

Locally computed SHA256 values, pinned by the downloader and converter:

```text
2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2  body-annotations-male-cns-v1.0-minconf-0.5.feather
95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621  body-neurotransmitters-male-cns-v1.0.feather
e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1  connectome-weights-male-cns-v1.0-minconf-0.5.feather
```

## Exact retention and omissions

| Quantity | Count |
| --- | ---: |
| Source annotation records | 211,577 |
| Selected annotated/traced nodes | 167,216 |
| Omitted annotation records | 44,361 |
| Source IDs without any annotation record | 88,192,826 |
| Total omitted IDs from the three-table union | 88,237,187 |
| Source connection rows | 151,856,684 |
| Retained connection rows | 25,587,572 |
| Rows with one or both endpoints outside the selected scope | 126,269,112 |
| Invalid/nonpositive source rows | 0 |
| Source synapse-count sum | 311,833,243 |
| Retained synapse-count sum | 124,193,283 |
| Omitted synapse-count sum | 187,639,960 |
| Retained one-synapse rows | 10,301,874 |
| Retained two-synapse rows | 4,763,727 |
| Retained self-loop rows | 102 |
| Additional rows repeating a retained directed pair | 0 |
| Selected nodes with no retained incoming edges | 717 |
| Selected nodes with no retained outgoing edges | 1,609 |
| Minimum/maximum retained connection count | 1 / 2,591 |

Omitted annotation statuses: 11,864 Glia; 15,893 Orphan; 10,751 Unimportant;
3,470 unassigned; 1,832 Assign; and 551 Anchor. **Zero traced records are
omitted.** Records without classification or tracing evidence cannot be assumed
to be known neurons. These omissions are intentional and recorded, not hidden
under a claim to include every raw segment.

Connection rows are directed pairs with an aggregated synapse count, not one
row per individual synapse. The converter preserves source rows, including
duplicates if a future input has any; it does not coalesce them. Outgoing CSR
rows correspond to `body_pre`; `indices` refer to `body_post`. Weights are exact
unscaled counts stored as float32. Original body IDs are retained.

## Sign assumptions

The official `consensus_nt` is used without substituting another prediction:

- GABA, glutamate and histamine receive presynaptic sign -1
- Acetylcholine and other labels receive +1
- Unknown/missing labels receive provisional +1, with an explicit count

The selected records contain 103,746 acetylcholine; 29,303 glutamate; 22,073
GABA; 7,897 histamine; 392 dopamine; 101 octopamine; 48 serotonin; 3,142 unclear;
and 514 without an NT row. Thus 59,273 nodes use negative sign, and **3,656
nodes have a provisional unknown sign**. Dopamine, serotonin and octopamine
also have context-dependent effects; assigning them +1 is a simplifying model
choice, not proof of excitatory function. Glutamate's effect depends on receptor
and circuit context too. Connectivity is anatomical evidence; signs, rate
dynamics, plasticity and game input/output mappings are engineering hypotheses.

## Reproduce and load

Install the separately declared conversion dependency:

```bash
python -m pip install -r requirements-data.txt
python tools/prepare_malecns.py \
  --raw-dir "$HOME/fly-dcss-data/malecns/raw" \
  --output-dir "$HOME/fly-dcss-data/malecns/full-annotated" \
  --scope full-annotated --download
```

`--scope` is required. Both data directories must be outside the repository.
Existing raw files are reused only after checking their size, SHA256 and MD5.
Changed files fail rather than silently replacing the release. Downloading is
bounded to 1,200,000,000 bytes and 900 seconds by default. To retry an interrupted
download, rerun the command; unfinished `.part` files are restarted. Conversion
requires an absent or empty output directory and writes `metadata.json` last,
so a partial conversion cannot be loaded as complete.

```python
from fly_dcss.malecns import load_malecns
graph, provenance = load_malecns("/path/to/malecns/full-annotated")
assert graph.n == 167216
assert len(graph.indices) == 25587572
```

The loader verifies cached-array SHA256 hashes and Graph invariants. It loads
the entire prepared scope, has no subset argument, and never downloads.

Conversion scans Arrow record batches twice. Per-batch endpoint mapping and a
stable scatter build disk-backed NPY arrays without a Python list of all edges
or an in-memory whole-file connection table. Only annotation IDs and per-node
working arrays are held across batches. Stored arrays are int32 pointers, body
IDs and endpoint indices; float32 weights; and int8 signs. The Graph runtime
may widen its small per-node arrays while retaining compact edge arrays.

On October 5, 2026, the real conversion took **42.30 seconds**, with Linux
`resource.getrusage` peak RSS **1,375,072 KiB (1.31 GiB)**. NPY files total
206,206,164 bytes, about 196.7 MiB. This is a measured conversion result, not a
runtime memory or speed guarantee. Run conversion in a separate process before
starting the game. All arrays remain external in the output directory selected
by the conversion command.

Verification included **74 direction/count spot-checks across source batches**,
retained-pair duplicate checks, count reconciliation, source and output hashes,
and synthetic tests for traced-but-unclassified nodes, one/two-synapse edges,
cross-class edges, isolated nodes, self-loops, duplicate rows, null/invalid
rows, checksum failures, overflow rejection, and download limits. Full game
evidence is documented separately in [FULL_CONNECTOME_SMOKE.md](FULL_CONNECTOME_SMOKE.md).

Output SHA256 values (deterministic arrays; metadata timestamps differ):

```text
0e874326b12fb48f56c2de25c7631f6bb1ccc22e7407f7b4af34b56c4357f5b9  indptr.npy
00a9889870866804399d506dbc8d55040ea9ba015218fc27a743a401acb61703  indices.npy
77b43b0ad37fac0a1a4d4cb8660eac92e8a16155fcbc1141978b2e22d25d87cd  weight.npy
34510cbb528282befe7bb8215cdad4200d06fafe4f9c8876d377ef04a7a548bc  sign.npy
d0d8b013c5dceebfd337507a647eae361d19687223cf022cd2020227e1dd7c12  body.npy
```
