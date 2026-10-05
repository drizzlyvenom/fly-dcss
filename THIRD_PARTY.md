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
