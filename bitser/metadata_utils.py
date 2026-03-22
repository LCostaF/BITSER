import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from Bio import SeqIO


def extract_class_from_header(
    header: str,
    delim: str,
    which: int = 1,
) -> str | None:
    """
    Extract class label from FASTA header using a delimiter string and occurrence index.

    Examples:
    - delim=" ", which=1   → second token (Dengue style: >AB074760 1 ...)
    - delim="|", which=-1  → after last pipe (SARS-CoV-2 style)
    - delim="genotype ", which=1 → after "genotype " (HBV style)

    The extracted token is automatically cleaned:
    - Only alphanumeric characters are kept (A-Z, a-z, 0-9)
    - Result is stripped of leading/trailing whitespace
    """
    if not delim:
        return None

    # Handle positive (from start) and negative (from end) indexing
    if which >= 0:
        parts = header.split(delim, maxsplit=which + 1)
        if len(parts) <= which:
            return None
        after_delim = parts[which]
    else:
        # Negative which → count from the end (which = -1 → last occurrence)
        parts = header.rsplit(delim, maxsplit=abs(which))
        if len(parts) <= abs(which):
            return None
        after_delim = parts[-1]

    # Take the first word/token after the delimiter
    tokens = after_delim.split()
    if not tokens:
        return None

    raw_token = tokens[0].strip()

    # Keep only alphanumeric characters
    clean_class = ''.join(c for c in raw_token if c.isalnum())

    if not clean_class:
        return None

    return clean_class


def generate_metadata(
    dataset_dir: str,
    train_count: int = 100,
    seed: int = 7,
    class_delim: str | None = None,
    class_which: int = 1,
) -> Path:
    """
    Generate metadata.tsv by parsing class labels from FASTA headers using a delimiter.

    Required: class_delim must be provided.
    """
    dataset_path = Path(dataset_dir).resolve()
    seq_dir = dataset_path / 'sequences'
    metadata_path = dataset_path / 'metadata.tsv'

    if not seq_dir.is_dir():
        raise FileNotFoundError(
            f'sequences/ folder not found in {dataset_path}'
        )

    if class_delim is None:
        raise ValueError(
            'class_delim is required. Examples:\n'
            '  --class-delim " " --class-which 1          # second token\n'
            '  --class-delim "|" --class-which -1        # after last |\n'
            '  --class-delim "genotype " --class-which 1 # HBV style'
        )

    random.seed(seed)

    rows = []
    seen_samples: set[str] = set()

    fasta_files = list(seq_dir.glob('*.f*'))  # .fasta, .fa, .fna, .fas...
    if not fasta_files:
        raise FileNotFoundError(f'No FASTA files found in {seq_dir}')

    for fasta in fasta_files:
        fasta_rel = f'sequences/{fasta.name}'
        full_path = seq_dir / fasta.name

        for i, record in enumerate(SeqIO.parse(full_path, 'fasta')):
            header = record.description.strip()   # full header without >

            class_label = extract_class_from_header(
                header=header,
                delim=class_delim,
                which=class_which,
            )
            if class_label is None:
                continue

            sample_id = header
            counter = 1
            while sample_id in seen_samples:
                sample_id = f'{sample_id}_{counter}'
                counter += 1
            seen_samples.add(sample_id)

            rows.append(
                {
                    'sample-id': sample_id,
                    'fasta_path': fasta_rel,
                    'class': class_label,
                    'split': '',
                    'record_index': i,
                }
            )

    if not rows:
        raise ValueError('No valid sequences found after header parsing.')

    # Group by class and perform per-class train/test split
    class_to_rows = defaultdict(list)
    for row in rows:
        class_to_rows[row['class']].append(row)

    final_rows = []
    for cls, group in class_to_rows.items():
        random.shuffle(group)
        n_train = min(train_count, len(group))
        for i, row in enumerate(group):
            row['split'] = 'train' if i < n_train else 'test'
            final_rows.append(row)

    # Write metadata.tsv
    with metadata_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'sample-id',
                'fasta_path',
                'class',
                'split',
                'record_index',
            ],
            delimiter='\t',
            lineterminator='\n',
        )
        writer.writeheader()
        writer.writerows(final_rows)

    print(f'Generated {metadata_path}')
    print(f'  → {len(final_rows)} entries')
    print(f'  → {len(class_to_rows)} classes')
    print(
        f'  → delim={class_delim!r}, which={class_which}, train_count={train_count}'
    )

    return metadata_path
