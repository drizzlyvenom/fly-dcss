#!/usr/bin/env python3
"""Download/convert the pinned official MaleCNS release outside the repository."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fly_dcss.malecns import SCOPES, download_sources, prepare_malecns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scope", choices=SCOPES, required=True,
                        help="Explicit neuron selection; no weight or brain/VNC filtering")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--download-byte-budget", type=int, default=1_200_000_000)
    parser.add_argument("--download-timeout", type=int, default=900)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if args.raw_dir.resolve().is_relative_to(repo) or args.output_dir.resolve().is_relative_to(repo):
        parser.error("keep source data and converted binary arrays outside the repository")
    if args.download:
        download_sources(args.raw_dir, max_download_bytes=args.download_byte_budget,
                         timeout=args.download_timeout)
    metadata = prepare_malecns(args.raw_dir, args.output_dir, scope=args.scope,
                              progress=lambda msg: print(msg, file=sys.stderr, flush=True))
    print(json.dumps({k: metadata[k] for k in
                     ("dataset", "scope", "nodes", "edges", "source_connection_rows",
                      "out_of_scope_connection_rows", "retained_weight_one_rows",
                      "retained_weight_two_rows", "synapse_count_sum",
                      "conversion_elapsed_seconds", "conversion_peak_rss_kib")}, indent=2))


if __name__ == "__main__":
    main()
