import os
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from Bio import SeqIO
from joblib import Parallel, delayed

from bitser.genetic_texture_analysis import calc_bwp, calc_bws, calc_hist
from bitser.sequence_utils import translate

POWERS_OF_TWO = 2 ** np.arange(8)


def process_file(file_in, allowed_ids, class_map, flank, translate_sequences):
    """
    Extract features from sequences in a FASTA file, restricted to allowed IDs.

    :param file_in: Path to FASTA file
    :param allowed_ids: Set of sequence IDs to process
    :param class_map: Dict mapping sequence ID -> class label
    :param flank: Sliding window size
    :param translate_sequences: Whether to translate nucleotide sequences
    """

    try:
        feature_batch = []
        headers = []
        sequences = []

        with open(file_in, encoding='utf-8') as handle:
            for record in SeqIO.parse(handle, 'fasta'):

                seq_id = record.id

                if seq_id not in allowed_ids:
                    continue

                seq_record = ''.join(
                    ch
                    for ch in str(record.seq).upper()
                    if ch in {'A', 'C', 'G', 'T'}
                )

                headers.append(record.description)
                sequences.append(seq_record)

                hist_center = calc_hist(
                    seq_record, flank, translate_sequences, True
                )

                bws = calc_bws(hist_center)
                bwp = calc_bwp(hist_center)

                class_label = class_map[seq_id]

                concat_features = hist_center + [bws, bwp, class_label]

                feature_batch.append(concat_features)

        return np.array(feature_batch, dtype=object), headers, sequences

    except Exception as e:
        print(f'Error processing file {file_in}: {e}')
        return np.array([]), [], []


def extract_features_from_metadata(
    metadata_path,
    split=None,
    flank: int = 8,
    translate_sequences=False,
    n_jobs=-1,
):
    """
    Extract features using a metadata table describing sequences.

    Metadata must contain columns:
        sample-id
        fasta_path
        class
        split (optional)

    :param metadata_path: Path to metadata TSV
    :param split: Filter metadata rows by split (train/test/etc)
    :param flank: Sliding window size
    :param translate_sequences: Translate sequences or not
    :param n_jobs: Parallel jobs
    """

    metadata = pd.read_csv(metadata_path, sep='\t', comment='#')

    if split is not None and 'split' in metadata.columns:
        metadata = metadata[metadata['split'] == split]

    base_dir = Path(metadata_path).parent

    # Group metadata by FASTA file
    grouped = metadata.groupby('fasta_path')

    tasks = []

    for fasta_path, group in grouped:

        full_path = base_dir / fasta_path

        allowed_ids = set(group['sample-id'])

        class_map = dict(zip(group['sample-id'], group['class']))

        tasks.append((full_path, allowed_ids, class_map))

    results = Parallel(n_jobs=n_jobs)(
        delayed(process_file)(
            str(file_path), allowed_ids, class_map, flank, translate_sequences
        )
        for file_path, allowed_ids, class_map in tasks
    )

    features_list = [r[0] for r in results if len(r[0]) > 0]
    headers_list = [r[1] for r in results]
    sequences_list = [r[2] for r in results]

    if len(features_list) == 0:
        raise ValueError('No sequences were processed. Check metadata.')

    all_headers = [h for sublist in headers_list for h in sublist]
    all_sequences = [s for sublist in sequences_list for s in sublist]

    return np.vstack(features_list), all_headers, all_sequences
