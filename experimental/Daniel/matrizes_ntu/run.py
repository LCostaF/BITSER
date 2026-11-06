#!/usr/bin/env python3
"""Generate three PNGs comparing matched mRNA/ncRNA NTU matrices in Mus musculus.

Usage:
    python experimental/Daniel/matrizes_ntu/run.py
    python experimental/Daniel/matrizes_ntu/run.py --output experimental/Daniel/matrizes_ntu/results/reuniao
"""

import argparse
import json
from pathlib import Path

import numpy as np

if __package__:
    from . import dataset, figures, ntu
else:
    import dataset
    import figures
    import ntu

ROOT = Path(__file__).resolve().parent


def execute(args):
    records = dataset.deduplicate(dataset.read_records(args.data_dir))
    pairs = dataset.build_pairs(records)
    support = ntu.structural_support()
    means = np.stack([ntu.mean_profile((pair[column] for pair in pairs), support)
                      for column in range(2)])

    args.output.mkdir(parents=True, exist_ok=True)
    figures.plot_support(support, args.output)
    figures.plot_means(means, support, len(pairs), args.output)
    figures.plot_difference(means, support, args.output)
    parameters = {"data_dir": str(args.data_dir.resolve()), "seed": dataset.SEED}
    (args.output / "parameters.json").write_text(
        json.dumps(parameters, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Mus musculus: {len(pairs):,} pares; {len(pairs):,} sequências por classe.")
    print(f"Suporte: {support.sum():,} transições possíveis.")
    print(f"3 PNGs e parâmetros: {args.output.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT.parents[1] / "data_plek",
                        help="PLEK directory containing Mus_musculus/sequences/")
    parser.add_argument("-o", "--output", type=Path, default=ROOT / "results" / "Mus_musculus")
    args = parser.parse_args()
    try:
        execute(args)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
