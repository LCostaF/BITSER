import doctest
import os
import tempfile
from unittest.mock import mock_open, patch

import numpy as np

from bitser import feature_extraction


def test_doctests():
    result = doctest.testmod(feature_extraction)
    assert result.failed == 0


def test_calc_hist():
    seq = 'ATGGTGCATCTGACTCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTGAACGTGGATGAAGTTGGTGGTGAGGCCCTGGGCAGGCTGCTGGTGGTCTACCCTTGGACCCAGAGGTTCTTTGAGTCCTTTGGGGATCTGTCCACTCCTGATGCTGTTATGGGCAACCCTAAGGTGAAGGCTCATGGCAAGAAAGTGCTCGGTGCCTTTAGTGATGGCCTGGCTCACCTGGACAACCTCAAGGGCACCTTTGCCACACTGAGTGAGCTGCACTGTGACAAGCTGCACGTGGATCCTGAGAACTTCAGGCTCCTGGGCAACGTGCTGGTCTGTGTGCTGGCCCATCACTTTGGCAAAGAATTCACCCCACCAGTGCAGGCTGCCTATCAGAAAGTGGTGGCTGGTGTGGCTAATGCCCTGGCCCACAAGTATCAC'
    hist = feature_extraction.calc_hist(seq)
    assert len(hist) == 256
    assert hist == [
        13,
        6,
        3,
        3,
        3,
        2,
        3,
        2,
        3,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        6,
        1,
        1,
        1,
        3,
        1,
        1,
        0,
        0,
        0,
        2,
        1,
        3,
        1,
        0,
        2,
        3,
        5,
        1,
        0,
        2,
        2,
        2,
        0,
        1,
        1,
        1,
        0,
        1,
        0,
        1,
        1,
        2,
        0,
        1,
        0,
        1,
        2,
        0,
        1,
        2,
        1,
        2,
        0,
        1,
        2,
        2,
        3,
        3,
        2,
        2,
        1,
        3,
        1,
        0,
        1,
        3,
        2,
        0,
        1,
        2,
        1,
        1,
        0,
        0,
        1,
        0,
        1,
        0,
        0,
        0,
        0,
        2,
        0,
        1,
        1,
        2,
        1,
        1,
        1,
        0,
        1,
        1,
        0,
        1,
        1,
        0,
        0,
        2,
        1,
        2,
        1,
        1,
        1,
        2,
        0,
        0,
        2,
        0,
        0,
        1,
        1,
        2,
        1,
        1,
        0,
        0,
        3,
        3,
        0,
        1,
        3,
        4,
        2,
        3,
        2,
        1,
        0,
        4,
        0,
        2,
        0,
        0,
        0,
        1,
        0,
        0,
        1,
        3,
        0,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        3,
        0,
        1,
        0,
        1,
        2,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        0,
        0,
        2,
        0,
        1,
        1,
        2,
        1,
        0,
        0,
        0,
        2,
        0,
        2,
        1,
        1,
        1,
        0,
        1,
        0,
        1,
        0,
        0,
        1,
        2,
        1,
        0,
        1,
        0,
        0,
        0,
        3,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        1,
        1,
        0,
        0,
        0,
        1,
        3,
        1,
        1,
        2,
        5,
        0,
        0,
        3,
        2,
        1,
        2,
        1,
        0,
        3,
        0,
        2,
        1,
        3,
        1,
        0,
        1,
        4,
        1,
        1,
        1,
        0,
        2,
        3,
        2,
        1,
        0,
        3,
        1,
        2,
        2,
        139,
    ]


def test_calc_bws():
    hist = [
        13,
        6,
        3,
        3,
        3,
        2,
        3,
        2,
        3,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        6,
        1,
        1,
        1,
        3,
        1,
        1,
        0,
        0,
        0,
        2,
        1,
        3,
        1,
        0,
        2,
        3,
        5,
        1,
        0,
        2,
        2,
        2,
        0,
        1,
        1,
        1,
        0,
        1,
        0,
        1,
        1,
        2,
        0,
        1,
        0,
        1,
        2,
        0,
        1,
        2,
        1,
        2,
        0,
        1,
        2,
        2,
        3,
        3,
        2,
        2,
        1,
        3,
        1,
        0,
        1,
        3,
        2,
        0,
        1,
        2,
        1,
        1,
        0,
        0,
        1,
        0,
        1,
        0,
        0,
        0,
        0,
        2,
        0,
        1,
        1,
        2,
        1,
        1,
        1,
        0,
        1,
        1,
        0,
        1,
        1,
        0,
        0,
        2,
        1,
        2,
        1,
        1,
        1,
        2,
        0,
        0,
        2,
        0,
        0,
        1,
        1,
        2,
        1,
        1,
        0,
        0,
        3,
        3,
        0,
        1,
        3,
        4,
        2,
        3,
        2,
        1,
        0,
        4,
        0,
        2,
        0,
        0,
        0,
        1,
        0,
        0,
        1,
        3,
        0,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        3,
        0,
        1,
        0,
        1,
        2,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        0,
        0,
        2,
        0,
        1,
        1,
        2,
        1,
        0,
        0,
        0,
        2,
        0,
        2,
        1,
        1,
        1,
        0,
        1,
        0,
        1,
        0,
        0,
        1,
        2,
        1,
        0,
        1,
        0,
        0,
        0,
        3,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        1,
        1,
        0,
        0,
        0,
        1,
        3,
        1,
        1,
        2,
        5,
        0,
        0,
        3,
        2,
        1,
        2,
        1,
        0,
        3,
        0,
        2,
        1,
        3,
        1,
        0,
        1,
        4,
        1,
        1,
        1,
        0,
        2,
        3,
        2,
        1,
        0,
        3,
        1,
        2,
        2,
        139,
    ]
    bws = feature_extraction.calc_bws(hist)
    assert bws == 0.6651270207852193


def test_calc_bwp():
    hist = [
        13,
        6,
        3,
        3,
        3,
        2,
        3,
        2,
        3,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        6,
        1,
        1,
        1,
        3,
        1,
        1,
        0,
        0,
        0,
        2,
        1,
        3,
        1,
        0,
        2,
        3,
        5,
        1,
        0,
        2,
        2,
        2,
        0,
        1,
        1,
        1,
        0,
        1,
        0,
        1,
        1,
        2,
        0,
        1,
        0,
        1,
        2,
        0,
        1,
        2,
        1,
        2,
        0,
        1,
        2,
        2,
        3,
        3,
        2,
        2,
        1,
        3,
        1,
        0,
        1,
        3,
        2,
        0,
        1,
        2,
        1,
        1,
        0,
        0,
        1,
        0,
        1,
        0,
        0,
        0,
        0,
        2,
        0,
        1,
        1,
        2,
        1,
        1,
        1,
        0,
        1,
        1,
        0,
        1,
        1,
        0,
        0,
        2,
        1,
        2,
        1,
        1,
        1,
        2,
        0,
        0,
        2,
        0,
        0,
        1,
        1,
        2,
        1,
        1,
        0,
        0,
        3,
        3,
        0,
        1,
        3,
        4,
        2,
        3,
        2,
        1,
        0,
        4,
        0,
        2,
        0,
        0,
        0,
        1,
        0,
        0,
        1,
        3,
        0,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        3,
        0,
        1,
        0,
        1,
        2,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        0,
        0,
        2,
        0,
        1,
        1,
        2,
        1,
        0,
        0,
        0,
        2,
        0,
        2,
        1,
        1,
        1,
        0,
        1,
        0,
        1,
        0,
        0,
        1,
        2,
        1,
        0,
        1,
        0,
        0,
        0,
        3,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        1,
        1,
        0,
        0,
        0,
        1,
        3,
        1,
        1,
        2,
        5,
        0,
        0,
        3,
        2,
        1,
        2,
        1,
        0,
        3,
        0,
        2,
        1,
        3,
        1,
        0,
        1,
        4,
        1,
        1,
        1,
        0,
        2,
        3,
        2,
        1,
        0,
        3,
        1,
        2,
        2,
        139,
    ]
    bwp = feature_extraction.calc_bwp(hist)
    assert bwp == 0.37644341801385683


def test_process_file():
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.fasta', delete=False
    ) as tmp_file:
        tmp_file.write(
            '>test_sequence\nATGGTGCATCTGACTCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTGAACGTGGATGAAGTTGGTGGTGAGGCCCTGGGCAGGCTGCTGGTGGTCTACCCTTGGACCCAGAGGTTCTTTGAGTCCTTTGGGGATCTGTCCACTCCTGATGCTGTTATGGGCAACCCTAAGGTGAAGGCTCATGGCAAGAAAGTGCTCGGTGCCTTTAGTGATGGCCTGGCTCACCTGGACAACCTCAAGGGCACCTTTGCCACACTGAGTGAGCTGCACTGTGACAAGCTGCACGTGGATCCTGAGAACTTCAGGCTCCTGGGCAACGTGCTGGTCTGTGTGCTGGCCCATCACTTTGGCAAAGAATTCACCCCACCAGTGCAGGCTGCCTATCAGAAAGTGGTGGCTGGTGTGGCTAATGCCCTGGCCCACAAGTATCAC\n'
        )
        tmp_file_path = tmp_file.name

    file_seq_counts = {tmp_file_path: 1}

    result = feature_extraction.process_file(
        tmp_file_path,
        flank=8,
        translate_sequences=False,
        file_seq_counts=file_seq_counts,
    )

    os.remove(tmp_file_path)

    assert len(result) > 0
    assert isinstance(result, np.ndarray)
    assert result.shape[1] == 259


def test_process_file_invalid_file():
    file_seq_counts = {}

    result = feature_extraction.process_file(
        'non_existent_file.fasta',
        flank=8,
        translate_sequences=False,
        file_seq_counts=file_seq_counts,
    )
    assert len(result) == 0


def test_extract_features_from_path():
    with tempfile.TemporaryDirectory() as tmp_dir:
        fasta_content = '>test_sequence\nATGGTGCATCTGACTCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTGAACGTGGATGAAGTTGGTGGTGAGGCCCTGGGCAGGCTGCTGGTGGTCTACCCTTGGACCCAGAGGTTCTTTGAGTCCTTTGGGGATCTGTCCACTCCTGATGCTGTTATGGGCAACCCTAAGGTGAAGGCTCATGGCAAGAAAGTGCTCGGTGCCTTTAGTGATGGCCTGGCTCACCTGGACAACCTCAAGGGCACCTTTGCCACACTGAGTGAGCTGCACTGTGACAAGCTGCACGTGGATCCTGAGAACTTCAGGCTCCTGGGCAACGTGCTGGTCTGTGTGCTGGCCCATCACTTTGGCAAAGAATTCACCCCACCAGTGCAGGCTGCCTATCAGAAAGTGGTGGCTGGTGTGGCTAATGCCCTGGCCCACAAGTATCAC\n'
        for i in range(3):
            with open(os.path.join(tmp_dir, f'test_{i}.fasta'), 'w') as f:
                f.write(fasta_content)

        result = feature_extraction.extract_features_from_path(
            tmp_dir, flank=8, translate_sequences=False, n_jobs=-1
        )

        assert len(result) == 3
        assert isinstance(result, np.ndarray)
        assert result.shape[1] == 259


# ---------------------------------------------------------------------------------------------------
def test_return_tu_array_with_unknown_characters():
    """Test that return_tu_array handles characters not found in the EIIP dictionary."""
    # Test when the center (first) character is not in the dictionary
    subseq = 'XACTGACTG'  # X is not in EIIP_Nucleotide
    result = feature_extraction.return_tu_array(
        subseq, feature_extraction.EIIP_Nucleotide
    )
    # When center is unknown, all comparisons should result in 0
    assert result == [0, 0, 0, 0, 0, 0, 0, 0]

    # Test when a neighbor character is not in the dictionary
    subseq = 'AXTGACTGA'  # X is not in EIIP_Nucleotide and is in position 1 (second character)
    result = feature_extraction.return_tu_array(
        subseq, feature_extraction.EIIP_Nucleotide
    )
    # The second element should be 0, others should be calculated normally
    assert result[0] == 0
    assert len(result) == 8


def test_calc_hist_with_translation():
    """Test that calc_hist translates the sequence when translated=True."""
    # Use a longer sequence to avoid division by zero in BWS/BWP calculations
    seq = 'ATGGTGCATCTGACTCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTGAACGTGGATGAAGAC'

    # First get histogram without translation
    hist_nucleotide = feature_extraction.calc_hist(seq, translated=False)

    # Then get histogram with translation
    hist_amino = feature_extraction.calc_hist(seq, translated=True)

    # The histograms should be different when translation is applied
    assert hist_nucleotide != hist_amino

    # Check basic properties of the histograms
    assert sum(hist_nucleotide) > 0, 'Nucleotide histogram is empty'
    assert sum(hist_amino) > 0, 'Amino acid histogram is empty'

    bws_nucleotide = feature_extraction.calc_bws(hist_nucleotide)
    bwp_nucleotide = feature_extraction.calc_bwp(hist_nucleotide)
    bws_amino = feature_extraction.calc_bws(hist_amino)
    bwp_amino = feature_extraction.calc_bwp(hist_amino)

    # Check that results are valid numbers (not NaN or infinite)
    assert not np.isnan(bws_nucleotide) and not np.isinf(bws_nucleotide)
    assert not np.isnan(bwp_nucleotide) and not np.isinf(bwp_nucleotide)
    assert not np.isnan(bws_amino) and not np.isinf(bws_amino)
    assert not np.isnan(bwp_amino) and not np.isinf(bwp_amino)


def test_count_sequences_in_file_exception():
    """Test that count_sequences_in_file handles exceptions properly."""
    # Create a non-existent file path
    file_path = '/nonexistent/path/to/file.fasta'

    # Mock open to raise an exception when called with this path
    with patch('builtins.open', mock_open()) as mock_file:
        mock_file.side_effect = FileNotFoundError('File not found')

        # Capture stdout to verify the error message
        with patch('builtins.print') as mock_print:
            result = feature_extraction.count_sequences_in_file(file_path)

            # Check that the function returns 0 on error
            assert result == 0

            # Check that an error message was printed
            mock_print.assert_called_once()
            call_args = mock_print.call_args[0][0]
            assert 'Error counting sequences in' in call_args
            assert 'File not found' in call_args


def test_extract_features_with_translation():
    """Test that extract_features_from_path works with translation."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Use a longer sequence that will generate meaningful histograms even after translation
        fasta_content = """>test_sequence
        ATGGTGCATCTGACTCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTGAACGTGGATGAAGTTGGTGGTGAGGCC
        CTGGGCAGGCTGCTGGTGGTCTACCCTTGGACCCAGAGGTTCTTTGAGTCCTTTGGGGATCTGTCCACTCCTGATGCTGTTATG
        GGCAACCCTAAGGTGAAGGCTCATGGCAAGAAAGTGCTCGGTGCCTTTAGTGATGGCCTGGCTCACCTGGACAACCTCAAGGGC
        ACCTTTGCCACACTGAGTGAGCTGCACTGTGACAAGCTGCACGTGGATCCTGAGAACTTCAGGCTCCTGGGCAACGTGCTGGTC
        """

        # Create a test FASTA file
        with open(os.path.join(tmp_dir, 'test.fasta'), 'w') as f:
            f.write(fasta_content)

        # Extract features with translation
        result_translated = feature_extraction.extract_features_from_path(
            tmp_dir, flank=8, translate_sequences=True, n_jobs=1
        )

        # Extract features without translation
        result_untranslated = feature_extraction.extract_features_from_path(
            tmp_dir, flank=8, translate_sequences=False, n_jobs=1
        )

        # Verify that the results have the same shape
        assert result_translated.shape == result_untranslated.shape

        bws_translated = result_translated[0][-3]
        bwp_translated = result_translated[0][-2]
        assert not np.isnan(bws_translated) and not np.isinf(bws_translated)
        assert not np.isnan(bwp_translated) and not np.isinf(bwp_translated)

        # The BWS/BWP values should be different between translated and untranslated
        bws_untranslated = result_untranslated[0][-3]
        bwp_untranslated = result_untranslated[0][-2]
        assert bws_translated != bws_untranslated
        assert bwp_translated != bwp_untranslated
