# distutils: language=c++
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

import numpy as np

cimport numpy as np

import cython

from libc.math cimport round
from libcpp.string cimport string
from libcpp.vector cimport vector

# Define numpy array types
np.import_array()
ctypedef np.uint8_t DTYPE_t

# Constants
cdef int MAX_TU_NUMBER = 255
cdef int[8] POWERS_OF_TWO = [1, 2, 4, 8, 16, 32, 64, 128]

# Forward declaration of functions
cdef extern from "genetic_utils.h":
    string translate(const string& seq) nogil

# EIIP
EIIP_NUCLEOTIDE = {
    'G': 0.0806,
    'A': 0.1260,
    'T': 0.1335,
    'C': 0.1340
}

EIIP_AMINO_ACID = {
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

cpdef float calculate_nucleotide_value(str seq, int index):
    """
    Calculate nucleotide value as sum of current and next two nucleotides
    """
    cdef:
        float total = 0.0
        int seq_len = len(seq)
        int i

    for i in range(index, min(index + 3, seq_len)):
        total += EIIP_NUCLEOTIDE.get(seq[i], 0)
    return total

cpdef np.ndarray[DTYPE_t, ndim=1] return_tu_array(str subseq, int flank=8, bint add_center=False):
    """
    Transform a neighborhood into a Texture Unit array using new value calculation
    """
    cdef:
        float center_val
        list neighbor_vals
        np.ndarray[DTYPE_t, ndim=1] result
        int i

    # Calculate values for all positions in subsequence
    cdef int sub_len = len(subseq)
    cdef list values = [calculate_nucleotide_value(subseq, j) for j in range(sub_len)]

    center_val = values[0]
    neighbor_vals = values[1:flank + 1]
    
    result = np.zeros(flank, dtype=np.uint8)
    for i in range(flank):
        if neighbor_vals[i] >= center_val:
            result[i] = 1
    
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int calc_tu_number(np.ndarray[DTYPE_t, ndim=1] tu_array, int flank=8):
    """
    Find the Texture Unit Number based on the input Texture Unit array.
    
    Args:
        tu_array: Array of neighborhood comparison values (0 or 1)
        flank: Size of the sliding window
        
    Returns:
        Texture Unit Number value for the neighborhood (between 0 and 255)
    """
    cdef:
        int result = 0
        int i
    
    for i in range(flank):
        result += tu_array[i] * POWERS_OF_TWO[i]
    
    return min(result, MAX_TU_NUMBER)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list calc_hist(str seq, int flank=24, bint translated=False, bint add_center=False):
    """
    Calculate the texture unit histogram.
    """
    cdef:
        dict eiip
        str processed_seq = seq
        np.ndarray[DTYPE_t, ndim=1] tu_numbers
        int i, length
        np.ndarray[np.int64_t, ndim=1] histogram

    flank=24
    
    # Only use EIIP for amino acids if translated
    if translated:
        eiip = EIIP_AMINO_ACID
        processed_seq = translate(seq.encode('utf-8')).decode('utf-8')
    else:
        # For nucleotide sequences, we don't use eiip anymore
        pass
    
    length = len(processed_seq) - flank
    tu_numbers = np.zeros(length, dtype=np.uint8)
    
    for i in range(0, length, 3):
        tu_numbers[i] = calc_tu_number(
            return_tu_array(processed_seq[i:i + flank + 1], flank, add_center),
            flank
        )
    
    histogram = np.bincount(tu_numbers, minlength=256)
    return histogram.tolist()


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef float calc_bws(list hist):
    """
    BWS - Compare "symmetry" between first half and last half of the sequence
    
    Args:
        hist: A texture unit histogram
        
    Returns:
        The BWS value
    """
    cdef:
        np.ndarray[np.int64_t, ndim=1] histogram = np.array(hist, dtype=np.int64)
        np.int64_t total_sum = 0
        np.int64_t total_div = 0
        int i
    
    for i in range(127):
        total_sum += abs(histogram[i] - histogram[i + 128])
    
    total_div = np.sum(histogram)
    
    if total_div == 0:
        return 0.0
    return 1.0 - (total_sum / float(total_div))


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef float calc_bwp(list hist):
    """
    BWP - Compare "palindrome" degree of the sequence
    
    Args:
        hist: A texture unit histogram
        
    Returns:
        The BWP value
    """
    cdef:
        np.ndarray[np.int64_t, ndim=1] histogram = np.array(hist, dtype=np.int64)
        np.int64_t total_sum = 0
        np.int64_t total_div = 0
        int i
    
    for i in range(127):
        total_sum += abs(histogram[i] - histogram[255 - i])
    
    total_div = np.sum(histogram)
    
    if total_div == 0:
        return 0.0
    return 1.0 - (total_sum / float(total_div))


# Define a more efficient specialized version for processing large sequences
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list process_sequence_batch(list sequences, int flank=8, bint translated=False):
    """
    Process a batch of sequences efficiently.
    
    Args:
        sequences: List of genetic sequences
        flank: Size of the sliding window
        translated: Whether to translate the sequences
        
    Returns:
        List of histogram results
    """
    cdef:
        int num_sequences = len(sequences)
        list results = []
        int i
    
    for i in range(num_sequences):
        results.append(calc_hist(sequences[i], flank, translated))
    
    return results