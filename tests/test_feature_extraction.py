import doctest
import tempfile
import os

import numpy as np

from bitser import feature_extraction


def test_doctests():
    result = doctest.testmod(feature_extraction)
    assert result.failed == 0


def test_calc_hist():
    seq = 'ATGGTGCATCTGACTCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTGAACGTGGATGAAGTTGGTGGTGAGGCCCTGGGCAGGCTGCTGGTGGTCTACCCTTGGACCCAGAGGTTCTTTGAGTCCTTTGGGGATCTGTCCACTCCTGATGCTGTTATGGGCAACCCTAAGGTGAAGGCTCATGGCAAGAAAGTGCTCGGTGCCTTTAGTGATGGCCTGGCTCACCTGGACAACCTCAAGGGCACCTTTGCCACACTGAGTGAGCTGCACTGTGACAAGCTGCACGTGGATCCTGAGAACTTCAGGCTCCTGGGCAACGTGCTGGTCTGTGTGCTGGCCCATCACTTTGGCAAAGAATTCACCCCACCAGTGCAGGCTGCCTATCAGAAAGTGGTGGCTGGTGTGGCTAATGCCCTGGCCCACAAGTATCAC'
    hist = feature_extraction.calc_hist(seq)
    assert len(hist) == 256
    assert hist == [13, 6, 3, 3, 3, 2, 3, 2, 3, 1, 1, 1, 1, 1, 0, 0, 6, 1, 1, 1, 3, 1, 1, 0, 0, 0, 2, 1, 3, 1, 0, 2, 3, 5, 1, 0, 2, 2, 2, 0, 1, 1, 1, 0, 1, 0, 1, 1, 2, 0, 1, 0, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 2, 3, 3, 2, 2, 1, 3, 1, 0, 1, 3, 2, 0, 1, 2, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 2, 0, 1, 1, 2, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 2, 1, 2, 1, 1, 1, 2, 0, 0, 2, 0, 0, 1, 1, 2, 1, 1, 0, 0, 3, 3, 0, 1, 3, 4, 2, 3, 2, 1, 0, 4, 0, 2, 0, 0, 0, 1, 0, 0, 1, 3, 0, 1, 1, 1, 0, 0, 0, 0, 0, 3, 0, 1, 0, 1, 2, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 2, 0, 1, 1, 2, 1, 0, 0, 0, 2, 0, 2, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 2, 1, 0, 1, 0, 0, 0, 3, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 3, 1, 1, 2, 5, 0, 0, 3, 2, 1, 2, 1, 0, 3, 0, 2, 1, 3, 1, 0, 1, 4, 1, 1, 1, 0, 2, 3, 2, 1, 0, 3, 1, 2, 2, 139]


def test_calc_bws():
    hist = [13, 6, 3, 3, 3, 2, 3, 2, 3, 1, 1, 1, 1, 1, 0, 0, 6, 1, 1, 1, 3, 1, 1, 0, 0, 0, 2, 1, 3, 1, 0, 2, 3, 5, 1, 0, 2, 2, 2, 0, 1, 1, 1, 0, 1, 0, 1, 1, 2, 0, 1, 0, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 2, 3, 3, 2, 2, 1, 3, 1, 0, 1, 3, 2, 0, 1, 2, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 2, 0, 1, 1, 2, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 2, 1, 2, 1, 1, 1, 2, 0, 0, 2, 0, 0, 1, 1, 2, 1, 1, 0, 0, 3, 3, 0, 1, 3, 4, 2, 3, 2, 1, 0, 4, 0, 2, 0, 0, 0, 1, 0, 0, 1, 3, 0, 1, 1, 1, 0, 0, 0, 0, 0, 3, 0, 1, 0, 1, 2, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 2, 0, 1, 1, 2, 1, 0, 0, 0, 2, 0, 2, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 2, 1, 0, 1, 0, 0, 0, 3, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 3, 1, 1, 2, 5, 0, 0, 3, 2, 1, 2, 1, 0, 3, 0, 2, 1, 3, 1, 0, 1, 4, 1, 1, 1, 0, 2, 3, 2, 1, 0, 3, 1, 2, 2, 139]
    bws = feature_extraction.calc_bws(hist)
    assert bws == 0.6651270207852193


def test_calc_bwp():
    hist = [13, 6, 3, 3, 3, 2, 3, 2, 3, 1, 1, 1, 1, 1, 0, 0, 6, 1, 1, 1, 3, 1, 1, 0, 0, 0, 2, 1, 3, 1, 0, 2, 3, 5, 1, 0, 2, 2, 2, 0, 1, 1, 1, 0, 1, 0, 1, 1, 2, 0, 1, 0, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 2, 3, 3, 2, 2, 1, 3, 1, 0, 1, 3, 2, 0, 1, 2, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 2, 0, 1, 1, 2, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 2, 1, 2, 1, 1, 1, 2, 0, 0, 2, 0, 0, 1, 1, 2, 1, 1, 0, 0, 3, 3, 0, 1, 3, 4, 2, 3, 2, 1, 0, 4, 0, 2, 0, 0, 0, 1, 0, 0, 1, 3, 0, 1, 1, 1, 0, 0, 0, 0, 0, 3, 0, 1, 0, 1, 2, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 2, 0, 1, 1, 2, 1, 0, 0, 0, 2, 0, 2, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 2, 1, 0, 1, 0, 0, 0, 3, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 3, 1, 1, 2, 5, 0, 0, 3, 2, 1, 2, 1, 0, 3, 0, 2, 1, 3, 1, 0, 1, 4, 1, 1, 1, 0, 2, 3, 2, 1, 0, 3, 1, 2, 2, 139]
    bwp = feature_extraction.calc_bwp(hist)
    assert bwp == 0.37644341801385683


def test_process_file():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as tmp_file:
        tmp_file.write(">test_sequence\nATGGTGCATCTGACTCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTGAACGTGGATGAAGTTGGTGGTGAGGCCCTGGGCAGGCTGCTGGTGGTCTACCCTTGGACCCAGAGGTTCTTTGAGTCCTTTGGGGATCTGTCCACTCCTGATGCTGTTATGGGCAACCCTAAGGTGAAGGCTCATGGCAAGAAAGTGCTCGGTGCCTTTAGTGATGGCCTGGCTCACCTGGACAACCTCAAGGGCACCTTTGCCACACTGAGTGAGCTGCACTGTGACAAGCTGCACGTGGATCCTGAGAACTTCAGGCTCCTGGGCAACGTGCTGGTCTGTGTGCTGGCCCATCACTTTGGCAAAGAATTCACCCCACCAGTGCAGGCTGCCTATCAGAAAGTGGTGGCTGGTGTGGCTAATGCCCTGGCCCACAAGTATCAC\n")
        tmp_file_path = tmp_file.name

    result = feature_extraction.process_file(tmp_file_path, flank=8, translate_sequences=False)

    os.remove(tmp_file_path)

    assert len(result) > 0
    assert isinstance(result, np.ndarray)
    assert result.shape[1] == 259


def test_process_file_invalid_file():
    result = feature_extraction.process_file("non_existent_file.fasta", flank=8, translate_sequences=False)
    assert len(result) == 0  # Ensure the function handles the error gracefully


def test_extract_features_from_path():
    with tempfile.TemporaryDirectory() as tmp_dir:
        fasta_content = ">test_sequence\nATGGTGCATCTGACTCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTGAACGTGGATGAAGTTGGTGGTGAGGCCCTGGGCAGGCTGCTGGTGGTCTACCCTTGGACCCAGAGGTTCTTTGAGTCCTTTGGGGATCTGTCCACTCCTGATGCTGTTATGGGCAACCCTAAGGTGAAGGCTCATGGCAAGAAAGTGCTCGGTGCCTTTAGTGATGGCCTGGCTCACCTGGACAACCTCAAGGGCACCTTTGCCACACTGAGTGAGCTGCACTGTGACAAGCTGCACGTGGATCCTGAGAACTTCAGGCTCCTGGGCAACGTGCTGGTCTGTGTGCTGGCCCATCACTTTGGCAAAGAATTCACCCCACCAGTGCAGGCTGCCTATCAGAAAGTGGTGGCTGGTGTGGCTAATGCCCTGGCCCACAAGTATCAC\n"
        for i in range(3):
            with open(os.path.join(tmp_dir, f"test_{i}.fasta"), 'w') as f:
                f.write(fasta_content)

        result = feature_extraction.extract_features_from_path(tmp_dir, flank=8, translate_sequences=False, n_jobs=-1)

        assert len(result) == 3
        assert isinstance(result, np.ndarray)
        assert result.shape[1] == 259
