# Full annotated/traced graph smoke: 2026-10-05

Passed: the entire explicitly selected MaleCNS annotated/traced graph completed
five actual DCSS 0.17.1 actions and five game turns. This establishes connectivity,
transport, state retention, numerical finiteness and local-update plumbing only.
It does not establish that learning improves play or models fly physiology.

## Precisely what “full” means here

Pinned selection: every annotation record with a nonempty superclass OR
status=Traced. This yields **167,216 modeled nodes**: 166,700 classified records
plus 516 additional traced records with unknown superclass. All 165,122
status=Traced records are included. Brain and VNC are both included; no
brain↔VNC connections are deliberately removed.

Every positive source connection between selected endpoints is retained:
**25,587,572 directed edges**, representing **124,193,283 synapse counts**.
This includes 10,301,874 one-synapse and 4,763,727 two-synapse edges. No ≥3 filter
is used. The result includes 102 self-loops and zero duplicate endpoint pairs.

This is not the literal all-segment release. The three official tables contain
88,404,403 distinct segment/body IDs. The model excludes 88,237,187 of these:
44,361 other annotation records and 88,192,826 IDs absent from the annotation
table. It excludes 126,269,112 source connection rows / 187,639,960 synapse counts
because one or both endpoints lie outside the stated scope. No source rows were
invalid. The upstream release already applies min-confidence 0.5. Neither its
anatomical completeness nor all provisional annotations are independently
validated by this implementation.

## Checks

- Source file size, SHA-256 and MD5 match the pinned official files
- Raw positive rows partition exactly into retained and out-of-scope rows
- Converted CSR keeps body_pre→body_post direction and original body IDs
- 74 samples across the source file match CSR endpoint and synapse count
- Graph validation checks pointers, endpoint bounds, signs, finite nonnegative
  weights and unique node IDs; cache files are SHA-256 verified at load
- 717 selected nodes have no incoming connection, 1,609 have no outgoing
  connection; nodes are retained rather than silently discarded
- 128/128 observation features reach the fixed engineered input projection,
  four input links per modeled node; each of five action readouts includes all
  modeled nodes, with finite normalized positive weights
- A separate functional pulse on source body 10001 traversed its 313 actual
  outgoing connections, activating 313 downstream nodes. Full sparse recurrent
  current matched direct source-row calculation exactly (maximum error 0). With
  the directly pulsed node removed, the downstream signal reached all five
  action readouts. See [full-connectivity-result.json](full-connectivity-result.json)
  and run `python tools/check_connectome.py --malecns /path/to/prepared`
- Neural state, edge weights and eligibility remained finite
- Ten synchronous circuit steps (choice + feedback per action) advanced time
  2, 4, 6, 8, 10, retaining state and eligibility throughout
- Actions: east once, then wait four times; rewards 0.04, then four × −0.01
- Nonzero cumulative actual weight-update L1: 291.7707895913; net L1 from initial
  weights: 181.6932863914. These magnitudes are not evidence of useful learning

Presynaptic GABA/glutamate/histamine labels are modeled as inhibitory; this is a
heuristic. 3,656 unclear/missing transmitter labels use a provisional positive
sign. Modulatory transmitter/receptor effects are not biologically simulated.
The game input/output mappings are fixed engineering choices, not identified
sensory and motor pathways.

## Measured resources

Linux x86-64, Python 3.12.14, NumPy 2.3.5; conversion used PyArrow 23.0.1.
All work ran on CPU, without CUDA or paid compute.

- Raw download: 1,109,008,094 bytes across three official files
- Conversion: 42.298 seconds, 1,375,072 KiB peak process RSS (~1.31 GiB)
- Full game load/initialization/episode/checkpoint: 9.743 seconds
- Episode plus cleanup/checkpoint: 9.064 seconds
- Full game Python process peak RSS: 623,091,712 bytes (~594 MiB)
- Episode JSON: about 33 KiB; final circuit checkpoint: about 196 MiB

RSS is the measured Python/converter process high-water mark, not a whole-system
or GPU figure. The separate DCSS child is not included in Python RSS. Downloads,
conversion and gameplay were separate processes. Plan at least several GiB of
available RAM and several GiB of free disk locally, plus compiler/build headroom.
The literal 88-million-segment simulation was deliberately not attempted because
its per-node mappings and working vectors would exceed this environment's budget.

The full arrays remain sparse and float32/int32 where supported. Circuit steps
process at most 1,000,000 edges at a time; no dense N×N adjacency or BPTT/history
is allocated. Logs contain scalar summaries, with arrays saved only once at end.

## Reproduce and continue

See [LOCAL_HANDOFF.md](LOCAL_HANDOFF.md) for WSL2 build, verified download,
conversion and exact run commands. The bounded command uses `--steps 5 --seconds
180 --seed 7`. It stops at `action_limit` with exit status 0. A fresh run resets
both game and circuit; the final NPZ is for analysis, not resumable gameplay.

[full-smoke-result.json](full-smoke-result.json) records exact measured values,
output hashes and runtime hash. [MALECNS_DATA.md](MALECNS_DATA.md) records source
URLs, licenses, transformation and omission details. Raw data, game binaries,
checkpoints and episode logs are not committed.

Final fixture suite: 58 tests passed with warnings treated as errors and PyArrow
enabled, including converter preservation/overflow/cache checks and pulse wiring.
The base NumPy-only environment skips the optional Feather conversion tests.
