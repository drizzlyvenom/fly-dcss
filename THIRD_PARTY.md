# Third-party provenance

## jppaquet/flybrain

The outgoing-CSR loading convention in `fly_dcss/circuit.py` is adapted from
`Connectome.__init__` in `engine.py`. The archive schema was checked against
`convert.py`. No upstream simulator, metadata loader, datasets or large binaries
are bundled. The rate dynamics and plasticity rule are new prototype code, not a
claim of equivalence to flybrain's LIF simulator or validated fly biology.

Sources inspected on 2026-10-05, repository revision
`c582cfdb90c49cfa7da71d0f5ec958dc17730223`:

- [engine.py](https://github.com/jppaquet/flybrain/blob/c582cfdb90c49cfa7da71d0f5ec958dc17730223/engine.py)
- [convert.py](https://github.com/jppaquet/flybrain/blob/c582cfdb90c49cfa7da71d0f5ec958dc17730223/convert.py)
- [MIT license](https://github.com/jppaquet/flybrain/blob/c582cfdb90c49cfa7da71d0f5ec958dc17730223/LICENSE)

Copyright (c) 2026 jppaquet. The full MIT notice is retained in
[LICENSES/flybrain-MIT.txt](LICENSES/flybrain-MIT.txt).

The converter describes its MaleCNS source data as Janelia/Google Research CC-BY.
That comment is not an independent verification of a dataset release's exact
license, attribution obligations or conditions. No real data is redistributed
here; verify the specific dataset's primary source and license before importing
or redistributing it. Synthetic test/demo values and IDs are invented locally.

NumPy is a runtime dependency installed separately, not vendored. This notice
does not assign a license to the rest of fly-dcss.

## Dungeon Crawl Stone Soup 0.17.1

The runtime is built separately from the official
[`crawl/crawl` tag `0.17.1`](https://github.com/crawl/crawl/tree/0.17.1).
It is not included in this repository. Copyright 1997–2015 Linley Henzell,
the dev team, and the contributors. DCSS is licensed under GPL version 2 or,
at your option, any later version; its source also identifies compatible
component-specific licenses. The full upstream notice and GPL text are retained
in [LICENSES/DCSS-GPL-2.0.txt](LICENSES/DCSS-GPL-2.0.txt).

[patches/dcss-0.17.1-pipes.patch](patches/dcss-0.17.1-pipes.patch) is a derived
modification of `crawl-ref/source/tileweb.cc`, distributed under GPL-2.0-or-later.
It adds an opt-in inherited-pipe transport for the existing WebTiles JSON
protocol because the experiment environment disallows Unix datagram sockets.
It preserves the original socket fallback and changes no gameplay rules,
observations, map generation, or neural/learning logic. Applying this patch
therefore produces a transport-patched 0.17.1 runtime, not a byte-identical
unmodified upstream executable.

[tools/build_dcss.sh](tools/build_dcss.sh) pins the source and required dependency
archive SHA-256 hashes. It downloads the exact Lua, SQLite, libpng, and zlib
submodule revisions referenced by tag 0.17.1, plus ncurses 6.5 from GNU's official
server. Their source trees and original license notices remain in the external
build directory; no dependency source or binary is vendored here. The script
adds explicit header includes for compatibility with modern GCC and supplies
standard release-version metadata. It does not install system-wide packages.

Protocol references:

- [WebTiles protocol implementation](https://github.com/crawl/crawl/blob/0.17.1/crawl-ref/source/tileweb.cc)
- [Input mode enum](https://github.com/crawl/crawl/blob/0.17.1/crawl-ref/source/defines.h)
- [Tile visibility flags](https://github.com/crawl/crawl/blob/0.17.1/crawl-ref/source/enum.h)
- [Upstream license](https://github.com/crawl/crawl/blob/0.17.1/crawl-ref/licence.txt)

This provenance notice does not assign a license to the rest of fly-dcss.

## MaleCNS-derived Fly Hero structural fixture

`fly_dcss/connectome.py` reads, but does not vendor, the graph published by
[bsgelman/fly-hero](https://github.com/bsgelman/fly-hero/tree/99cdd390193e20a0e85718951de2ea8eb4dd6555),
revision `99cdd390193e20a0e85718951de2ea8eb4dd6555`, `data/subgraph.json`.
The derived graph is identified as CC BY 4.0 separately from that project's MIT
code. The [official MaleCNS site](https://male-cns.janelia.org/) links the dataset's
[Creative Commons Attribution 4.0 license](https://creativecommons.org/licenses/by/4.0/).
Attribution: MaleCNS v1.0, FlyEM at HHMI Janelia, University of Cambridge Zoology,
MRC Laboratory of Molecular Biology and Google Research; extraction by
bsgelman/fly-hero. Source checked 2026-10-05.

The source JSON SHA256 is
`6814cc3cc1a65e1e58f589f44deff69a410b3e2e92a95c4f1c0a7b49f4b73d4a`.
Our transformation selects a small directed-reachable induced subset, splits
signed counts into magnitude/presynaptic sign, and scales counts into model
weights. Source-local indices are retained because the derived JSON omits
biological body IDs. The source removes VNC-to-brain edges and infers inhibitory
sign for GABA/glutamate/histamine; other/unknown transmitters are positive.
These assumptions are not independently validated here. No raw graph JSON,
full dataset or large binary is redistributed in git. Episode metadata records
attribution, selected local indices and exact source hash.

## Official MaleCNS full annotated/traced import

The full annotated/traced extension downloads official MaleCNS v1.0 annotation,
connection and neurotransmitter-prediction tables separately. See
[MALECNS_DATA.md](docs/MALECNS_DATA.md) for exact pinned URLs, hashes, CC BY 4.0
attribution, selection/exclusion counts and sign assumptions. No raw data or
converted arrays are redistributed in git. This model preserves all positive
source-table connections between the explicitly selected annotated/traced IDs,
including one- and two-synapse connections and brain/VNC boundaries. It excludes
the raw table's unannotated segmentation fragments; it is not an all-segment
simulation. The release's upstream min-confidence 0.5 remains in force.

PyArrow is an optional conversion-only dependency, installed separately; the
runtime continues to require NumPy only. Its Apache-2.0 distribution is not
vendored. Input projections, action readout, rate dynamics and local plasticity
are engineering choices and do not inherit anatomical validity from the source.
