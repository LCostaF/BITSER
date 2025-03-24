from Bio.Seq import Seq


# Translate RNA/DNA sequence into aminoacids sequence
def translate(seq: str) -> str:
    """
    :param seq: A genetic sequence in string format
    :return: The translated sequence
    >>> translate('ACTGGTCAATGCATGCCC')
    'TGQCMP'
    >>> translate('CTAGTATCAGATGCCAGT')
    'LVSDAS'
    >>> translate('TGCAGTACCATGGATCAG')
    'CSTMDQ'
    >>> translate('GAGATAGATACTAGTACA')
    'EIDTST'
    >>> translate('ATCGGTCAGTAGACTAGG')
    'IGQ*TR'
    """
    sequence = Seq(seq)
    return str(sequence.translate())
