import os
import pickle
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from Bio import SeqIO
from joblib import Parallel, delayed

from bitser.genetic_texture_analysis import calc_bwp, calc_bws, calc_hist

POWERS_OF_TWO = 2 ** np.arange(8)


def process_file(
    file_in, allowed_indices, class_map_by_index, flank, translate_sequences
):
    """
    Processes FASTA files and extracts features.
    """
    feature_batch = []
    headers = []
    sequences = []
    tu_orders = []
    trans_matrices = []

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

                hist_center, tu_order, trans_matrix = calc_hist(
                    seq_str, flank, translate_sequences, True
                )
                bws = calc_bws(hist_center)
                bwp = calc_bwp(hist_center)

                class_label = class_map_by_index[i]

                concat_features = hist_center + [bws, bwp, class_label]
                feature_batch.append(concat_features)

                tu_orders.append(tu_order)
                trans_matrices.append(trans_matrix)

        return (
            np.array(feature_batch, dtype=object),
            headers,
            sequences,
            tu_orders,
            trans_matrices,
        )

    except Exception as e:
        print(f'Error processing {file_in}: {e}')
        return np.array([], dtype=object), [], [], [], []


def extract_features_from_metadata(
    base_dir,
    split=None,
    flank: int = 8,
    translate_sequences=False,
    n_jobs=-1,
):
    """
    Extract features using a metadata table describing sequences.
    """
    metadata_path = os.path.join(base_dir, 'metadata.tsv')
    metadata = pd.read_csv(metadata_path, sep='\t', comment='#')

    if split is not None and 'split' in metadata.columns:
        metadata = metadata[metadata['split'] == split]

    grouped = metadata.groupby('fasta_path')

    tasks = []

    for fasta_path, group in grouped:
        full_path = os.path.join(base_dir, fasta_path)
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
    headers_list = [r[1] for r in results if len(r[0]) > 0]
    sequences_list = [r[2] for r in results if len(r[0]) > 0]
    tu_orders_list = [r[3] for r in results if len(r[0]) > 0]
    trans_matrices_list = [r[4] for r in results if len(r[0]) > 0]

    if len(features_list) == 0:
        raise ValueError('No sequences were processed. Check metadata.')

    all_headers = [h for sublist in headers_list for h in sublist]
    all_sequences = [s for sublist in sequences_list for s in sublist]
    all_tu_orders = [t for sublist in tu_orders_list for t in sublist]
    all_trans_matrices = [
        m for sublist in trans_matrices_list for m in sublist
    ]

    log_path = os.path.join(base_dir, 'tu_transition_log.pkl')
    log_data = {
        'headers': all_headers,
        'tu_orders': all_tu_orders,
        'transition_matrices': all_trans_matrices,
    }
    with open(log_path, 'wb') as f:
        pickle.dump(log_data, f)

    return np.vstack(features_list), all_headers, all_sequences
