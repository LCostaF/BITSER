"""Read Mus musculus PLEK records and build deduplicated exact-length pairs.

FASTA labels must match mRNA/lncRNA filenames; lncRNA is reported as ncRNA.
Effective sequences contain uppercase A/C/G/T only: U is discarded, not mapped
to T. Each pair contains its mRNA and ncRNA sequences, in that order.
"""

from collections import defaultdict
from pathlib import Path

import numpy as np

CLASSES = ("mRNA", "ncRNA")
FASTA_NAMES = ("mRNA", "lncRNA")
SEED = 42


def read_fasta(path):
    """Yield headers and joined sequences, ignoring whitespace and blank lines."""
    header, chunks = None, []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header, chunks = line[1:].strip(), []
                if not header:
                    raise ValueError(f"Empty FASTA header in {path}")
            elif header is None:
                raise ValueError(f"Sequence before first FASTA header in {path}")
            else:
                chunks.append("".join(line.split()))
    if header is not None:
        yield header, "".join(chunks)


def normalize(sequence):
    return "".join(base for base in sequence.upper() if base in "ACGT")


def read_records(data_dir):
    """Read Mus_musculus/sequences, keeping each class in FASTA order."""
    records = {}
    for label, filename in zip(CLASSES, FASTA_NAMES):
        path = Path(data_dir) / "Mus_musculus" / "sequences" / f"{filename}.fasta"
        values = []
        for index, (header, raw) in enumerate(read_fasta(path)):
            _, separator, header_label = header.rpartition("|||")
            if not separator or header_label.strip() != filename:
                raise ValueError(f"{path.name}, record {index}: expected header label {filename!r}")
            values.append(normalize(raw))
        if not values:
            raise ValueError(f"No FASTA records in {path}")
        records[label] = values
    return records


def deduplicate(records):
    """Keep first FASTA occurrences within classes and remove shared sequences."""
    unique = {label: dict.fromkeys(records[label]) for label in CLASSES}
    shared = unique["mRNA"].keys() & unique["ncRNA"].keys()
    return {label: [sequence for sequence in unique[label] if sequence not in shared]
            for label in CLASSES}


def build_pairs(records):
    """Return (mRNA, ncRNA) pairs of equal length using fixed seed 42."""
    groups = {label: defaultdict(list) for label in CLASSES}
    for label in CLASSES:
        for sequence in records[label]:
            if len(sequence) >= 10:
                groups[label][len(sequence)].append(sequence)
    rng = np.random.default_rng(SEED)
    pairs = []
    for length in sorted(groups["mRNA"].keys() & groups["ncRNA"].keys()):
        mrna, ncrna = [groups[label][length] for label in CLASSES]
        size = min(len(mrna), len(ncrna))
        m_indices = rng.choice(len(mrna), size, replace=False)
        n_indices = rng.choice(len(ncrna), size, replace=False)
        pairs.extend((mrna[m], ncrna[n]) for m, n in zip(m_indices, n_indices))
    if not pairs:
        raise ValueError("No exact-length pairs of at least 10 bases remain")
    return pairs


