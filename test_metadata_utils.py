# tests/test_metadata_utils.py
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pytest

from bitser.metadata_utils import (
    extract_class_from_header,
    generate_metadata,
)


def test_generate_metadata_missing_delim_raises(tmp_path):
    dataset_dir = tmp_path / 'no_delim'
    (dataset_dir / 'sequences').mkdir(parents=True)
    (dataset_dir / 'sequences' / 'dummy.fa').write_text('>seq\nACGT\n')

    with pytest.raises(ValueError, match='class_delim is required'):
        generate_metadata(str(dataset_dir), class_delim=None)
