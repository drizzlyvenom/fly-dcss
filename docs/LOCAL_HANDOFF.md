# Full MaleCNS → DCSS local handoff

This is a CPU-only experiment skeleton. An RTX 3090 is not used. The input and
readout are fixed engineered mappings, not identified fly sensory/motor pathways.
A short smoke establishes data plumbing and bounded numerical updates, not game
competence or biological learning. No long training job is started.

## Windows / WSL2

Use an existing Ubuntu WSL2 shell (native Windows Python is unsupported because
this runner uses POSIX PTYs and inherited pipes). Put the repository, runtime,
and data in the WSL Linux filesystem, such as `~/fly-work`, rather than `/mnt/c`.
Allow enough WSL memory and disk for the full annotated-neuron graph; measured values
are in `FULL_CONNECTOME_SMOKE.md`. Leave headroom beyond the reported process RSS.

Install the build prerequisites in your own WSL environment:

```sh
sudo apt-get update
sudo apt-get install -y git python3 python3-venv build-essential perl curl tar patch ncurses-bin
mkdir -p ~/fly-work
cd ~/fly-work
git clone --branch feat/dcss-minimal-loop https://github.com/drizzlyvenom/fly-dcss.git
cd fly-dcss
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r requirements-data.txt
python -m unittest discover -s tests -v
bash tools/build_dcss.sh "$HOME/fly-work/dcss017"
```

The script builds the exact source/dependency revisions with checked hashes.
The opt-in pipe IPC patch changes transport, not gameplay. Build outputs and
original source/license notices stay in the runtime directory outside git.
The previous cloud build was tested; WSL build instructions have not been run on
the owner's machine. There are no CUDA, PyTorch, server, or API-key dependencies.

## Data and bounded run

Follow [MALECNS_DATA.md](MALECNS_DATA.md) to download and prepare the official
full annotated-neuron graph. All nonempty-superclass records plus additional status=Traced records across
brain and VNC are included (167,216 modeled annotated/traced nodes, including
516 traced records with unknown superclass); all positive connections whose endpoints are both
in that set are retained, including one- and two-synapse connections. The raw
source also contains 88,404,403 segment IDs, mostly unannotated fragments. Those
fragments and other unclassified/glial annotation rows are excluded from this
explicit `full-annotated` model; it is not the literal all-segment graph.
Keep raw and converted data outside the repository. The
release itself uses a synapse-confidence threshold of 0.5; this conversion cannot
recover lower-confidence synapses omitted upstream.

```sh
python tools/prepare_malecns.py --download --scope full-annotated \
  --raw-dir "$HOME/fly-work/malecns/raw" \
  --output-dir "$HOME/fly-work/malecns/prepared"

python tools/check_connectome.py --malecns "$HOME/fly-work/malecns/prepared"

python -m fly_dcss.run_loop \
  --crawl "$HOME/fly-work/dcss017/crawl-0.17.1/crawl-ref/source/crawl" \
  --malecns "$HOME/fly-work/malecns/prepared" \
  --output runs/full-01 --steps 5 --seconds 180 --seed 7
```

The `--malecns` path is the prepared array directory, not a raw Feather file.
Use a new output directory each run. A repeated path is rejected. Defaults are
20 actions / 60 seconds; accepted maxima are 1,000 / 600. The episode deadline
starts after data loading and circuit initialization; preparation, checkpoint
writing and child cleanup can add wall time. No background training is scheduled.

## What is retained and what resets

Within one episode, neuron/segment activity, edge eligibility and learned weights
persist through observation → choice → actual game action → feedback cycles.
All imported source edges participate; only local scalar state is retained, with
no trajectory autograd graph or BPTT. Float32 edge state and bounded edge chunks
keep storage linear in nodes plus edges.

Every command starts a fresh game and resets circuit activity/eligibility/weights.
`--seed` seeds the game and fixed input/readout/exploration mapping. It does not
promise bit-identical outcomes across NumPy/platform versions. The no-save child
is closed at the limit or on unsafe input. There is no game-resume CLI.

Outputs:

- `episode.jsonl`: source metadata, exact runtime hash, mapping summary, observed
  game state, action/scores, reward, and scalar circuit/update summaries
- `summary.json`: completion/stop reason, graph size, finite-state checks, update
  magnitudes, measured wall time and Python peak RSS
- `console.log`: capped PTY output for startup/protocol debugging
- `circuit-final.npz`: final state, weights, eligibility and circuit time; useful
  for offline inspection, not a complete resumable agent/game checkpoint

The final checkpoint omits game state, RNG/visited state and mapping arrays.
Do not claim resumed gameplay by loading only its weights. It is deliberately
uncompressed to avoid costly compression in this smoke harness. It can be large.
JSON logs never dump per-edge weights or full neural states each turn.

## Next experiments, when you choose to continue

Start with repeated short seeds and frozen-weight/no-feedback controls before
longer training. Inspect saturation, inhibitory/excitatory balance, reward scale,
activity distribution and action diversity. The raw source includes unannotated
segments and glia/orphan annotations that this model excludes. Transmitter signs
remain provisional. Any future experimental selection must be explicit and
report removed nodes/edges. A biologically motivated interface requires an independently
validated cell-type mapping. None of these efficacy studies has been completed.
