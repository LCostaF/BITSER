import os
from fnmatch import fnmatch

import numpy as np
import numpy.typing as npt
from Bio import SeqIO
from joblib import Parallel, delayed

from bitser.genetic_texture_analysis import calc_bwp, calc_bws, calc_hist
from bitser.sequence_utils import translate

POWERS_OF_TWO = 2 ** np.arange(8)


def count_sequences_in_file(file_path):
    """
    Count the number of sequences in a FASTA file
    :param file_path: Path to the FASTA file
    :return: Number of sequences in the file
    """
    try:
        count = 0
        with open(file_path, encoding='utf-8') as handle:
            for _ in SeqIO.parse(handle, 'fasta'):
                count += 1
        return count
    except Exception as e:
        print(f'Error counting sequences in {file_path}: {e}')
        return 0


def process_file(file_in, flank, translate_sequences, file_seq_counts):
    """
    # Extract features from an individual FASTA file
    :param file_in: Path to the FASTA file
    :param flank: Size of the sliding window that runs through the sequence
    :param translate_sequences: Boolean for if the sequences should be translated or not
    :param file_seq_counts: Dictionary mapping file paths to their sequence counts
    :return: Numpy array of features, and sequence headers and the sequences
    """
    try:
        file_name = os.path.basename(file_in).split('.')[0]
        feature_batch = []
        headers = []
        sequences = []

        with open(file_in, encoding='utf-8') as handle:
            for record in SeqIO.parse(handle, 'fasta'):
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
                concat_features = hist_center + [bws, bwp, file_name]
                feature_batch.append(concat_features)

        return np.array(feature_batch, dtype=object), headers, sequences
    except Exception as e:
        print(f'Error processing file {file_in}: {e}')
        return np.array([]), [], []


def extract_features_from_path(
    dir_path, flank: int = 8, translate_sequences=False, n_jobs=-1
):
    """
    # Perform feature extraction on all FASTA files in a directory
    :param dir_path: The path to the target directory
    :param flank: Size of the sliding window that runs through the sequence
    :param translate_sequences: Boolean for if the sequences should be translated or not
    :param n_jobs: Maximum number of concurrently running jobs for Parallel execution
    :return: Numpy V Stacked features
    """
    files = [
        os.path.join(dir_path, name)
        for name in os.listdir(dir_path)
        if fnmatch(name, '*.fasta')
    ]

    file_seq_counts = {}
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_file)(
            file_in, flank, translate_sequences, file_seq_counts
        )
        for file_in in files
    )

    features_list = [r[0] for r in results]
    headers_list = [r[1] for r in results]
    sequences_list = [r[2] for r in results]

    all_headers = [h for sublist in headers_list for h in sublist]
    all_sequences = [s for sublist in sequences_list for s in sublist]

    return np.vstack(features_list), all_headers, all_sequences
