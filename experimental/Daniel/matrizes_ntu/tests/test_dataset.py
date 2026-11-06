"""Boundary and reproducibility tests for FASTA normalization and pair selection."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset import build_pairs, deduplicate, normalize, read_fasta, read_records


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.organism = self.root / "Mus_musculus"
        (self.organism / "sequences").mkdir(parents=True)

    def tearDown(self):
        self.temporary.cleanup()

    def fasta(self, filename, sequences):
        path = self.organism / "sequences" / f"{filename}.fasta"
        path.write_text("".join(f">record_{i} ||| {filename}\n{seq}\n"
                                for i, seq in enumerate(sequences)), encoding="utf-8")
        return path

    def test_fasta_normalization_and_labels(self):
        path = self.fasta("mRNA", ["ac guN\ntTA", "CC"])
        records = list(read_fasta(path))
        self.assertEqual(records[0][1], "acguNtTA")
        self.assertEqual(normalize(records[0][1]), "ACGTTTA")
        self.assertEqual(normalize("aCuUgTNRYSWKMBDHV-. 123"), "ACTTGT")
        self.fasta("lncRNA", ["C" * 10])
        path.write_text(">id ||| lncRNA\nAAAAAAAAAA\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "header label"):
            read_records(self.root)
        for contents in ("ACGT\n>id\nACGT", ">\nACGT"):
            with self.subTest(contents=contents):
                path.write_text(contents, encoding="utf-8")
                with self.assertRaises(ValueError):
                    list(read_fasta(path))

    def test_deduplication_keeps_first_normalized_occurrence(self):
        self.fasta("mRNA", ["a" * 10 + "UN", "A" * 10 + "T", "G" * 10, "T" * 10])
        self.fasta("lncRNA", ["C" * 10, "G" * 10, "c" * 10])
        records = read_records(self.root)
        self.assertEqual(records["mRNA"], ["A" * 10 + "T", "A" * 10 + "T", "G" * 10, "T" * 10])
        clean = deduplicate(records)
        self.assertEqual(clean, {"mRNA": ["A" * 10 + "T", "T" * 10], "ncRNA": ["C" * 10]})

    def test_pairs_are_reproducible_equal_length_and_without_replacement(self):
        records = {"mRNA": ["A" * 11, "A" * 10, "G" * 10, "AC" * 5, "ACG"],
                   "ncRNA": ["C" * 10, "T" * 10, "C" * 11, "T" * 12]}
        pairs = build_pairs(records)
        self.assertEqual(pairs, build_pairs(records))
        self.assertEqual([len(mrna) for mrna, _ in pairs], [10, 10, 11])
        self.assertTrue(all(len(mrna) == len(ncrna) for mrna, ncrna in pairs))
        for column, label in enumerate(("mRNA", "ncRNA")):
            selected = [pair[column] for pair in pairs]
            self.assertEqual(len(set(selected)), len(pairs))
            self.assertTrue(set(selected) <= set(records[label]))
        with self.assertRaises(ValueError):
            build_pairs({"mRNA": ["A" * 10], "ncRNA": ["T" * 11]})


if __name__ == "__main__":
    unittest.main()
