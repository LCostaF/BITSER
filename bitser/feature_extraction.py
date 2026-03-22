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


def process_file(
    file_in, allowed_indices, class_map_by_index, flank, translate_sequences
):
    """
    Now: allowed_indices = set of record_index values we care about
    class_map_by_index = {record_index: class_label, ...}
    """
    feature_batch = []
    headers = []
    sequences = []

    try:
        with open(file_in) as handle:
            for i, record in enumerate(SeqIO.parse(handle, 'fasta')):
                if i not in allowed_indices:
                    continue

                seq_str = ''.join(
                    ch for ch in str(record.seq).upper() if ch in 'ACGT'
                )

                headers.append(record.description)
                sequences.append(seq_str)

                hist_center = calc_hist(
                    seq_str, flank, translate_sequences, True
                )
                bws = calc_bws(hist_center)
                bwp = calc_bwp(hist_center)

                class_label = class_map_by_index[i]

                concat_features = hist_center + [bws, bwp, class_label]
                feature_batch.append(concat_features)

        return np.array(feature_batch, dtype=object), headers, sequences

    except Exception as e:
        print(f'Error processing {file_in}: {e}')
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
        allowed_indices = set(group['record_index'])
        class_map_by_index = dict(zip(group['record_index'], group['class']))

        tasks.append((full_path, allowed_indices, class_map_by_index))

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
