import csv
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

    if which >= 0:
        parts = header.split(delim, maxsplit=which + 1)
        if len(parts) <= which:
            return None
        after_delim = parts[which]
    else:
        parts = header.rsplit(delim, maxsplit=abs(which))
        if len(parts) <= abs(which):
            return None
        after_delim = parts[-1]

    tokens = after_delim.split()
    if not tokens:
        return None

    raw_token = tokens[0].strip()
    clean_class = ''.join(c for c in raw_token if c.isalnum())

    if not clean_class:
        return None

    return clean_class


def generate_metadata(
    dataset_dir: str,
    class_delim: str | None = None,
    class_which: int = 1,
) -> Path:
    """
    Generate metadata.tsv by parsing class labels from FASTA headers.

    Output columns: sample-id, fasta_path, class, record_index.
    No train/test split is written here; splitting is performed internally
    by the train command using a reproducible stratified split controlled
    by --seed.

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

    rows = []
    seen_samples: set[str] = set()

    fasta_files = list(seq_dir.glob('*.f*'))
    if not fasta_files:
        raise FileNotFoundError(f'No FASTA files found in {seq_dir}')

    short_sequences_found = False

    for fasta in fasta_files:
        fasta_rel = f'sequences/{fasta.name}'
        full_path = seq_dir / fasta.name

        for i, record in enumerate(SeqIO.parse(full_path, 'fasta')):
            header = record.description.strip()

            class_label = extract_class_from_header(
                header=header,
                delim=class_delim,
                which=class_which,
            )

            sequence = str(record.seq).strip()
            if len(sequence) < 1000:
                short_sequences_found = True

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
                    'record_index': i,
                }
            )

    if not rows:
        raise ValueError('No valid sequences found after header parsing.')

    class_to_rows = defaultdict(list)
    for row in rows:
        class_to_rows[row['class']].append(row)

    if len(class_to_rows) < 2:
        raise ValueError(
            'Dataset has fewer than 2 valid classes. Classification requires at least 2 classes.'
        )

    for cls, group in class_to_rows.items():
        if len(group) < 180:
            print(
                f'Warning: Class "{cls}" has only {len(group)} samples (<180 recommended). '
                'Performance may be impacted.'
            )

    if short_sequences_found:
        print(
            'Warning: Some sequences have length < 1000 characters. '
            'Performance may be impacted as BITSER was not intended to be used with smaller sequences.'
        )

    with metadata_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['sample-id', 'fasta_path', 'class', 'record_index'],
            delimiter='\t',
            lineterminator='\n',
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f'Generated {metadata_path}')
    print(f'  → {len(rows)} entries')
    print(f'  → {len(class_to_rows)} classes')
    print(f'  → delim={class_delim!r}, which={class_which}')

    return metadata_path
