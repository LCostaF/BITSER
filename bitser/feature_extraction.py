import os
from fnmatch import fnmatch

import numpy as np
from Bio import SeqIO
from joblib import Parallel, delayed

from bitser.sequence_utils import translate

EIIP_Nucleotide: dict[str, float] = {
    'A': 0.1260,
    'G': 0.0806,
    'T': 0.1335,
    'C': 0.1340,
}

EIIP_AminoAcid: dict[str, float] = {
    'L': 0.0000,
    'I': 0.0000,
    'N': 0.0036,
    'G': 0.0050,
    'V': 0.0057,
    'E': 0.0058,
    'P': 0.0198,
    'H': 0.0242,
    'K': 0.0371,
    'A': 0.0373,
    'Y': 0.0516,
    'W': 0.0548,
    'Q': 0.0761,
    'M': 0.0823,
    'S': 0.0829,
    'C': 0.0829,
    'T': 0.0941,
    'F': 0.0946,
    'R': 0.0959,
    'D': 0.1263,
}


def compare_neighbor(center, neighbor, add_center=False):
    """
    # Compare V={V0, V1, ... V8} where V0 represents the intensity value of the central pixel and
    # V(i) the intensity value of neighbouring pixel i. E(i) = 0 if V(i) < V0, E(i) = 1 if V(i) = V0,
    # E(i) = 2 if V(i) > V0 // V(i) called "neighbor" and V0 called "center" in the function below.
    #
    # Compares a neighboring pixel to the central pixel, returning values 0 or 1, based on the
    # comparison.
    :param center: Numerical value of the "center" (leftmost) sequence letter
    :param neighbor: Numerical value of the neighboring sequence letter
    :param add_center: Boolean to determine if the center value is added when neighbor == center (elif (return 1) case)
    :return: Numerical value 0 or 1 (1 + center if add_center)

    >>> compare_neighbor(0.1260, 0.0806)
    0
    >>> compare_neighbor(0.1335, 0.1260)
    0
    >>> compare_neighbor(0.1260, 0.1260)
    1
    >>> compare_neighbor(0.1260, 0.1340)
    1
    >>> compare_neighbor(0.1260, 0.1260, add_center=True)
    1.126
    """
    if neighbor < center:
        return 0
    elif neighbor >= center:
        return (1 + center) if add_center else 1


def return_tu_array(subseq, eiip, flank: int = 8, add_center=False):
    """
    # Transform a neighborhood (subsequence) into a Texture Unit (array of 8 elements, results of the 8
    # comparisons between the leftmost value and the 8 other values).
    # Example: V={A, C, T, G, G, A, G, A, T} -> TU={1,1,0,0,1,0,1,1}.
    :param subseq: String, subsequence with 9 elements, it is the portion of the sequence which will be analyzed for the neighborhood array
    :param eiip: The EIIP values dictionary
    :param flank: Size of the sliding window that runs through the sequence
    :param add_center: Boolean to determine if the center value is added when neighbor == center
    :return: The array with the comparison values of the center and each neighbor

    >>> return_tu_array("CACTCACTA", EIIP_Nucleotide)
    [0, 1, 0, 1, 0, 1, 0, 0]
    >>> return_tu_array("AGGGGCAGA", EIIP_Nucleotide)
    [0, 0, 0, 0, 1, 1, 0, 1]
    >>> return_tu_array("TAAGCAACT", EIIP_Nucleotide)
    [0, 0, 0, 1, 0, 0, 1, 1]
    >>> return_tu_array("CTCATCGTG", EIIP_Nucleotide)
    [0, 1, 0, 0, 1, 0, 0, 0]
    >>> return_tu_array("ACGTTAGGG", EIIP_Nucleotide)
    [1, 0, 1, 1, 1, 0, 0, 0]
    """
    texture_unit = []
    for y in range(0, flank):
        character = subseq[y + 1]
        if (character in eiip) and (subseq[0] in eiip):
            texture_unit.append(
                compare_neighbor(eiip[subseq[0]], eiip[character], add_center)
            )
        else:
            texture_unit.append(0)
    return texture_unit


def calc_tu_number(tu_array, flank: int = 8):
    """
    # Find the Texture Unit Number based on the input Texture Unit array (tu_array).
    #
    # Function takes the Texture Unit array (tu_array) and sums the values,
    # returning the Texture Unit Number (total_sum) ([0,1,2,...,255]).
    :param tu_array: Array of neighborhood comparison values (0, 1, or 1+center).
    :param flank: Size of the sliding window that runs through the sequence
    :return: Texture Unit Number value for the neighborhood

    >>> calc_tu_number([0, 0, 0, 0, 0, 0, 0, 0])
    0
    >>> calc_tu_number([0, 1, 0, 1, 0, 1, 0, 0])
    42
    >>> calc_tu_number([1, 1, 0, 1, 0, 0, 1, 1])
    203
    >>> calc_tu_number([1, 1, 0, 0, 1, 0, 1, 1])
    211
    >>> calc_tu_number([1, 1, 1, 1, 1, 1, 1, 1])
    255
    """
    tu_array = np.array(tu_array)
    powers_of_two = 2 ** np.arange(flank)
    total_sum = np.dot(tu_array, powers_of_two)
    return min(round(total_sum), 255)


def calc_hist(seq, flank: int = 8, translated=False, add_center=False):
    """
    # Calculate the texture unit histogram.
    :param seq: String. A genetic sequence extracted from a FASTA file
    :param flank: Size of the sliding window that runs through the sequence
    :param translated: Boolean to determine if the sequence is to be translated or not
    :param add_center: Boolean to determine if the center value is added when neighbor == center (elif (return 1) case)
    :return: The texture unit histogram (hist)
    """
    # Pick dictionary based on translation boolean
    eiip = EIIP_AminoAcid if translated else EIIP_Nucleotide
    # Translate sequence if translated == True
    if translated:
        seq = translate(seq)

    # Texture unit histogram
    hist = np.zeros(256, dtype=int)

    # # Loop through the sequence
    i = 0
    while i + flank < len(seq):
        # Populate array with neighborhood comparison results
        array = return_tu_array(seq[i:], eiip, flank, add_center)
        # Increment histogram position based on texture unit calculation
        hist[calc_tu_number(array, flank)] += 1
        i += 1
    return hist.tolist()


def calc_bws(hist):
    """
    # BWS - Compare "symmetry" between first half and last half of the sequence
    :param hist: A texture unit histogram
    :return: The BWS value
    """
    hist = np.array(hist)
    total_sum = np.sum(np.abs(hist[:127] - hist[128:255]))
    total_div = np.sum(hist)
    return 1 - (total_sum / total_div)


def calc_bwp(hist):
    """
    # BWP - Compare "palindrome" degree of the sequence (start of the sequence to middle, and end of the sequence to
    # middle)
    :param hist: A texture unit histogram
    :return: The BWP value
    """
    hist = np.array(hist)
    total_sum = np.sum(np.abs(hist[:127] - hist[255:128:-1]))
    total_div = np.sum(hist)
    return 1 - (total_sum / total_div)


def process_file(file_in, flank, translate_sequences):
    """
    # Extract features from an individual FASTA file
    :param file_in: Path to the FASTA file
    :param flank: Size of the sliding window that runs through the sequence
    :param translate_sequences: Boolean for if the sequences should be translated or not
    :return: Numpy array of features
    """
    try:
        file_name = os.path.basename(file_in).split('.')[0]
        feature_batch = []
        with open(file_in, encoding="utf-8") as handle:
            for record in SeqIO.parse(handle, "fasta"):
                seq_record = str(record.seq).upper()
                hist_center = calc_hist(seq_record, flank, translate_sequences, True)
                bws = calc_bws(hist_center)
                bwp = calc_bwp(hist_center)
                concat_features = hist_center + [bws, bwp, file_name]
                feature_batch.append(concat_features)
        return np.array(feature_batch, dtype=object)
    except Exception as e:
        print(f"Error processing file {file_in}: {e}")
        return np.array([])


def extract_features_from_path(dir_path, flank: int = 8, translate_sequences=False, n_jobs=-1):
    """
    # Perform feature extraction on all FASTA files in a directory
    :param dir_path: The path to the target directory
    :param flank: Size of the sliding window that runs through the sequence
    :param translate_sequences: Boolean for if the sequences should be translated or not
    :param n_jobs: Maximum number of concurrently running jobs for Parallel execution
    :return: Numpy V Stacked features
    """
    files = [os.path.join(dir_path, name) for name in os.listdir(dir_path)
             if fnmatch(name, "*.fasta")]

    # Use joblib to parallelize the processing of each file
    features = Parallel(n_jobs=n_jobs)(delayed(process_file)(file_in, flank, translate_sequences) for file_in in files)

    return np.vstack(features)
