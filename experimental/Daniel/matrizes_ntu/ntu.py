"""Encode BITSER transitions and average matrices with equal weight per sequence.

EIIP order: G < A < T < C. A neighbor >= the reference sets bit i (weight 2**i).
Rows are source NTUs and columns are destination NTUs.
"""

from itertools import islice, product

import numpy as np

_RANK = np.full(256, 255, dtype=np.uint8)
_RANK[np.frombuffer(b"GATC", dtype=np.uint8)] = np.arange(4)


def _encode_ranks(ranks):
    size = max(0, ranks.shape[-1] - 8)
    units = np.zeros(ranks.shape[:-1] + (size,), dtype=np.uint16)
    for bit in range(8):
        comparison = ranks[..., bit + 1:bit + 1 + size] >= ranks[..., :size]
        units |= comparison.astype(np.uint16) << bit
    return units


def encode(sequence):
    """Return overlapping NTUs from a normalized A/C/G/T string."""
    ranks = _RANK[np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)]
    if np.any(ranks == 255):
        raise ValueError("Sequence must contain only uppercase A/C/G/T")
    return _encode_ranks(ranks)


def _count_units(units):
    indices = 256 * units[..., :-1].astype(np.int64) + units[..., 1:]
    return np.bincount(indices.ravel(), minlength=65536).reshape(256, 256)


def transition_counts(sequence):
    """Count consecutive NTUs; fewer than ten bases produce no transitions."""
    return _count_units(encode(sequence))


def structural_support():
    """Find possible transitions by enumerating 4**10 words in bounded batches."""
    words = product(range(4), repeat=10)
    support = np.zeros((256, 256), dtype=bool)
    while batch := list(islice(words, 32768)):
        support |= _count_units(_encode_ranks(np.asarray(batch, dtype=np.uint8))) > 0
    return support


def mean_profile(sequences, support):
    """Accumulate normalized matrices without retaining individual matrices."""
    mean = np.zeros((256, 256))
    count = 0
    for sequence in sequences:
        if len(sequence) < 10:
            raise ValueError("A transition profile requires at least ten effective bases")
        mean += transition_counts(sequence) / (len(sequence) - 9)
        count += 1
    if not count:
        raise ValueError("No sequences selected")
    mean /= count
    if mean[~support].any() or not np.isclose(mean.sum(), 1):
        raise ValueError("Invalid normalized matrix or incomplete structural support")
    return mean
