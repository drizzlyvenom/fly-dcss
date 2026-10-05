# Minimal local DCSS loop

This is a correctness smoke harness, not a training result. It runs actual DCSS
0.17.1 game logic, observes the player's WebTiles view, drives an existing small
rate circuit, sends one cardinal move or wait, applies result-derived local
weight updates, and repeats with state/eligibility/weights retained.

## Build and run

Linux, Python 3.10+, NumPy, GCC/G++, GNU make, Perl, curl, tar, patch, sha256sum,
and `tic` are needed. No GPU, account, public game server, root installation,
or paid service is used. The old game build takes several minutes. Archives
are about 19 MB total; extracted sources/build outputs are larger and stay
outside this repository. Build dependencies are pinned and checksum-verified.

```sh
python -m pip install -r requirements.txt
bash tools/build_dcss.sh /tmp/fly-dcss017-runtime
python -m unittest discover -s tests -v

# Download separately; this 12 MB CC BY 4.0 derived graph is not vendored.
curl --fail --location \
  https://raw.githubusercontent.com/bsgelman/fly-hero/99cdd390193e20a0e85718951de2ea8eb4dd6555/data/subgraph.json \
  -o /tmp/flyhero-subgraph.json

python -m fly_dcss.run_loop \
  --crawl /tmp/fly-dcss017-runtime/crawl-0.17.1/crawl-ref/source/crawl \
  --connectome /tmp/flyhero-subgraph.json --neurons 64 \
  --output runs/smoke-01 --steps 20 --seconds 60 --seed 7
```

`--output` must be a new directory, protecting existing logs/saves. Omit
`--connectome` for an explicitly synthetic three-neuron transport smoke.
This is never described as a biological run. Default limits: 20 actions,
60 seconds; hard accepted maxima: 1,000 actions, 600 seconds. There is no
persistent/long training job. A successful limit stop returns exit status 0;
unexpected prompts, timeout, malformed protocol or process failure return 1.

## Exact runtime and transport

The build uses official `crawl/crawl` tag `0.17.1`, with the tag's exact Lua,
SQLite, libpng and zlib submodule revisions, plus ncurses 6.5. Source archive
SHA256: `e19645994bb90c15a1d114437afc906f730e5b0342a35270cb9a5a2b79143491`.
`tools/build_dcss.sh` lists all source URLs and checksums. Standard release
metadata (`util/release_ver`) is set to 0.17.1 and modern-header compatibility
is supplied by compiler flags.

This is **not an unmodified stock binary**: `patches/dcss-0.17.1-pipes.patch`
adds opt-in inherited-pipe IPC to `tileweb.cc`. This cloud sandbox does not
support Unix datagram socket creation. The patch retains the same WebTiles
JSON and unchanged gameplay; it substitutes inherited read/write descriptors
for `socket/bind/sendto/recvfrom`. No network listener is created. The native
Unix socket fallback remains available with `--transport unix`, but has not
been smoke-tested here. The old Tornado/Python-2 WebTiles server is not used.

The Python runner passes only its two pipe descriptors, starts an isolated PTY
child, verifies executable/protocol version, requests a full native snapshot,
and consumes explicit `flush_messages` boundaries. The full snapshot matters:
initial differential HUD updates can omit zero-valued fields. Cleanup sends
SIGHUP to the isolated child, with SIGKILL fallback after three seconds; no quit
keystroke is sent to an unknown menu. `-no-save` avoids saving a playable game,
but Crawl can still write startup preferences and morgue/dump files. The run's
output directory contains all such files. Sources and caches stay separately
under the build directory.

## Observation and action boundary

Protocol semantics checked against exact-tag source: `tileweb.cc`, `tileweb.h`,
`defines.h`, `enum.h`, and `webserver/connection.py`.

- Player allowlist: HP/max HP, MP/max MP, turn, relative position, place/depth
- A 5×5 player-centred patch: glyph, visible/remembered/unknown, visible monster
  presence only; map deltas and full-map clears are merged
- Visibility masks both WebTiles `TILE_FLAG_UNSEEN` (0x40000) and
  `TILE_FLAG_MM_UNSEEN` (0x20000, mapped/detected). A remembered/map-known glyph is
  not current sight; remembered monsters are never encoded as present
- Monster type, `typedata.avghp`, full stats, inventory and other payloads are
  discarded. No dungeon Lua, hidden map, or save-state reading feeds the agent
- Gameplay keys are allowed only at `input_mode=1`, `ui_state=0`, no open menu,
  complete HUD, positive HP. Prompts, death, menus and missing observations stop
- The action set is north/east/south/west/wait (`k/l/j/h/.`). A conservative mask
  permits movement only into visible floor/stairs or a visible monster (bump
  combat). Doors, water, unknown terrain and other features are deferred

A flush marks a rendering batch; animations can flush before a command is
complete. The current conservative guard can therefore stop on an ordinary
animation/combat frame too. This is a known usability limit, not permission to
send a key early.

Only command mode is automated. This initial harness deliberately stops on
unhandled inventory, confirmation, targeting, or other prompts rather than
trying arbitrary keys. It does not explore full levels or clear the dungeon.

## Small circuit and feedback

The fixed 128-feature observation encoder keeps unknown, remembered and visible
flags distinct. A seeded nonnegative fixed projection drives the selected
neurons. A seeded fixed readout scores five actions from circuit rates; epsilon
0.25 adds explicit random exploration among permitted actions. These mappings
are engineering choices, not inferred fly sensory/motor anatomy.

Each action uses one circuit step with learning disabled. The resulting
observation then drives one feedback step with the existing local eligibility
rule and a scalar modulator. Reward is clipped to [-1,1]:

`0.05 * newly_visited_position + HP_change / previous_max_HP - 0.01 * max(1, turn_change)`

Weights initialize at `raw_count * 0.5 / max(1, max_absolute_incoming_count_sum)`;
learning rate 0.02, other `Parameters` defaults retained. All selected recurrent
edges are plastic. Neither projection nor readout is trained. Observation,
action, actual turn change, reward, scalar weight/state summaries and circuit
time are logged per transition in `episode.jsonl`. `summary.json` is concise;
`console.log` captures at most roughly 2 MB of terminal output.

A different seed can change both the game and engineering projections. Even
with a fixed seed, cross-build deterministic DCSS trajectories are not assumed.
Same-seed circuit unit tests check only our deterministic component.

## Data provenance and limits

The separately downloaded Fly Hero JSON is derived from MaleCNS v1.0:
6,900 nodes / 990,102 weighted directed pairs, 12,073,747 bytes. SHA256:
`6814cc3cc1a65e1e58f589f44deff69a410b3e2e92a95c4f1c0a7b49f4b73d4a`.
Default import verifies this exact hash. Directed breadth-first selection starts
at source-local index 0, visits outgoing neighbors by descending count then
index, and retains the induced edges among the first 64 reached neurons.
This yields 64 nodes, 1,252 edges and seven inhibitory-sign neurons.

Crucially, the upstream derived JSON omits biological body IDs. `Graph.body`
therefore holds explicitly labelled **source-local indices**, never claimed
MaleCNS body IDs. Source selection drops VNC-to-brain edges; inhibitory signs
are inferred for GABA/glutamate/histamine, with other/unknown transmitters
positive. This is genuine connectome-derived structure, but not a validated
biological decision circuit, complete brain, or physiological learning model.
The importer records the exact selection, source, attribution and limitations
in each episode's metadata. See [third-party notices](../THIRD_PARTY.md).

## Verification

34 unit tests cover prior dynamics/loading plus fixture schema/sign/checksum,
protocol fragmentation, delta merge, unknown/remembered visibility, hidden-data
exclusion, prompt/menu rejection, action mask, deterministic controller and
stateful nonzero local feedback updates. No scientific controls or performance
study are included; they remain later work after this functioning loop.

A real local 20-action smoke on 2026-10-05 used the patched 0.17.1 executable,
seed 7 and the pinned 64-neuron graph. It reached turn 20 with 19 waits and one
east move, HP 18→18, and nonzero weight updates on all 20 transitions. This is
end-to-end wiring evidence only. It does not show improved play or biological
learning. A compact checked run summary is in `docs/smoke-result.json`.


The full annotated/traced MaleCNS extension uses compact edge chunks and sparse
fixed input mappings. See [local handoff](LOCAL_HANDOFF.md). The final circuit
arrays are saved once as `circuit-final.npz`; they are not dumped every turn.
