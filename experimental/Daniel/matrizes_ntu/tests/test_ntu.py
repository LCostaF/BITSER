"""Independent checks of NTU coding, support and equally weighted mean matrices."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ntu


def scalar_counts(sequence):
    rank = {base: value for value, base in enumerate("GATC")}
    units = [sum(1 << bit for bit in range(8)
                 if rank[sequence[start + bit + 1]] >= rank[sequence[start]])
             for start in range(len(sequence) - 8)]
    counts = np.zeros((256, 256), dtype=int)
    for source, destination in zip(units[:-1], units[1:]):
        counts[source, destination] += 1
    return units, counts


class NTUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.support = ntu.structural_support()

    def test_encoding_and_counts_match_scalar_reference(self):
        rng = np.random.default_rng(7)
        sequences = [base * 25 for base in "ACGT"]
        sequences += ["".join(rng.choice(list("ACGT"), length))
                      for length in (0, 8, 9, 10, 11, 25, 500) for _ in range(10)]
        for sequence in sequences:
            units, counts = scalar_counts(sequence)
            np.testing.assert_array_equal(ntu.encode(sequence), units)
            np.testing.assert_array_equal(ntu.transition_counts(sequence), counts)
            self.assertEqual(counts.sum(), max(0, len(sequence) - 9))
            self.assertFalse(counts[~self.support].any())
        with self.assertRaises(ValueError):
            ntu.encode("ACGTN")

    def test_structural_support(self):
        self.assertEqual(self.support.shape, (256, 256))
        self.assertEqual(self.support.dtype, np.dtype(bool))
        self.assertEqual(self.support.sum(), 8748)
        self.assertFalse(self.support[128, 132])
        self.assertFalse(self.support[132, 184])
        self.assertTrue(self.support[255, 255])

    def test_mean_weights_sequences_equally(self):
        sequences = ["A" * 10, "ACGT" * 20]
        counts = [scalar_counts(sequence)[1] for sequence in sequences]
        expected = sum(matrix / (len(sequence) - 9)
                       for sequence, matrix in zip(sequences, counts)) / 2
        actual = ntu.mean_profile(iter(sequences), self.support)
        np.testing.assert_allclose(actual, expected)
        self.assertAlmostEqual(actual.sum(), 1)
        pooled = sum(counts) / sum(matrix.sum() for matrix in counts)
        self.assertFalse(np.allclose(actual, pooled))
        for invalid in ([], ["ACGT"]):
            with self.assertRaises(ValueError):
                ntu.mean_profile(invalid, self.support)
        with self.assertRaises(ValueError):
            ntu.mean_profile(sequences, np.zeros((256, 256), dtype=bool))


if __name__ == "__main__":
    unittest.main()
