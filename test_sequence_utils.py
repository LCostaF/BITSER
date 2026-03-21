import doctest

from bitser import sequence_utils


def test_doctests():
    result = doctest.testmod(sequence_utils)
    assert result.failed == 0
