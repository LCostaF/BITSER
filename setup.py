# -*- coding: utf-8 -*-
from setuptools import setup

packages = ['bitser']

package_data = {'': ['*']}

install_requires = [
    'bio>=1.7.1,<2.0.0',
    'cython>=3.0.10,<4.0.0',
    'joblib>=1.4.2,<2.0.0',
    'matplotlib>=3.10.1,<4.0.0',
    'numpy>=2.2.3,<3.0.0',
    'pandas>=2.2.3,<3.0.0',
    'rich>=13.9.4,<14.0.0',
    'scikit-learn>=1.6.1,<2.0.0',
    'seaborn>=0.13.2,<0.14.0',
    'typer>=0.15.2,<0.16.0',
    'xgboost>=2.1.4,<3.0.0',
]

entry_points = {'console_scripts': ['bitser = bitser.cli:app']}

setup_kwargs = {
    'name': 'bitser',
    'version': '0.2.0',
    'description': '',
    'long_description': '\n<div align="center" style="display: display_block">\n\n# **BITSER**\n\n#### **BI**nary pa**T**tern **S**equenc**E** **R**ecognition\n\n![image_info](https://img.shields.io/badge/bitser-v0.2.0-blue)\n\n</div>\n\n\n<div align="center">\n    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/python/python-original.svg" width="100" height="100" />\n    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/poetry/poetry-original.svg" width="100" height="100" />\n    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/scikitlearn/scikitlearn-original.svg" width="100" height="100" />\n    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/numpy/numpy-original.svg" width="100" height="100" />\n    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/pandas/pandas-original.svg" width="100" height="100" />\n</div>\n\n\n## Overview\n\nBITSER (Binary Pattern Sequence Recognition) is a software tool built with the Python language that extracts features segments of each genetic sequence at a local level.\n\nThe method for feature extraction utilizes the concept of Local Binary Pattern (LBP), as well as adapted versions of the Texture Unit and Texture Unit Number from the field of computer vision, to obtain informative features from sequences organized in FASTA files.\n\nA k-mer window (default size 9) slides over each genetic sequence, comparing the leftmost nucleotide or aminoacid in the window with the 8 other members.\n\nThis tool is targeted for usage by biologists, researchers and other professionals in the field of bioinformatics.\n\n## Installation\n\n\n\n## CLI commands\n\nBITSER offers the following commands:\n\n| COMMAND | FUNCTION                                         |\n|---------|--------------------------------------------------|\n| train   | Train a classification model from sequence data  |\n| predict | Predict classes for new sequences using a trained model |\n\n### `train` command\n\nThis command initiates the feature extraction and model training workflow, and should be used on a training dataset. It has the following parameters:\n\n| PARAMETER | DESCRIPTION                                               | OPTIONAL | DEFAULT VALUE |\n|-----------|-----------------------------------------------------------|:--------:|-----------|\n| ``input`` | Path to the directory containing FASTA files for training |    ❌     |           |\n| ``output`` | Path to save the trained model to a file                  |     ✔️     | model.pkl |\n| ``classifier``| Type of classifier that will be used to train the model   | ✔️ | xgb       |\n| ``flank`` | How many characters in a sequence will be compared to the leftmost member of the sliding window | ✔️ | 8\n| ``translate`` | Whether the sequence should be translated to aminoacids or not | ✔️ | False |\n\n### `predict` command\n\nThis command initiates the feature extraction on the testing dataset, and then predicts classes based on the trained model. It has the following parameters:\n\n| PARAMETER      | DESCRIPTION                                                                                                                                    | OPTIONAL  | DEFAULT VALUE |\n|----------------|------------------------------------------------------------------------------------------------------------------------------------------------|:---------:|--|\n| ``model``      | Path to trained model obtained after training                                                                                                  |     ❌     |  |\n| ``data``       | Path to the directory containing FASTA files for testing                                                                                       |     ❌     |  |\n| ``flank``      | How many characters in a sequence will be compared to the leftmost member of the sliding window (must match value used in the `train` command) |    ✔️     | 8\n| ``translate``  | Whether the sequence should be translated to aminoacids or not (must match value used in the `train`command)                                   |    ✔️     | False |\n\n## Example Usage\n\nConsidering the following example project structure:\n\n```\n──project\n  └───datasets\n      ├───training_data\n      │   ├───class_a.fasta\n      │   └───class_b.fasta\n      └───testing_data\n          ├───class_a.fasta\n          └───class_b.fasta  \n```\n\nBITSER could be run from the project root, with the following command:\n\n`bitser train --input .\\datasets\\training_data\\ --output example_model.pkl`\n\nThe training data would be used to construct a classification model, which are saved to a Pickle file. The accuracy values for cross-validation would be saved to a text file.\n\nThe updated project structure after running the train command:\n\n```\n──project\n  ├───datasets\n  │   ├───training_data\n  │   │   ├───class_a.fasta\n  │   │   └───class_b.fasta\n  │   └───testing_data\n  │       ├───class_a.fasta\n  │       └───class_b.fasta\n  ├───example_model.pkl\n  └───results\n      └───20250401_094215_results_xgb.txt\n```\n\nThe saved model can then be used to predict class values, with the following command:\n\n`bitser predict --model example_model.pkl --data .\\datasets\\testing_data\\`\n\nThe model would be used to evaluate the testing data, and classify sequences accordingly. The classification results, per-class accuracies, and feature importance data are saved to a text file.\n\nThe updated project structure after running the predict command:\n\n```\n──project\n  ├───datasets\n  │   ├───training_data\n  │   │   ├───class_a.fasta\n  │   │   └───class_b.fasta\n  │   └───testing_data\n  │       ├───class_a.fasta\n  │       └───class_b.fasta\n  ├───example_model.pkl\n  └───results\n      ├───20250401_094215_results_xgb.txt\n      └───20250401_094215_results_xgbclassifier.txt\n```\n\n\n##### Acknowledgements\n\n* This study was supported by national funds through the Coordenação de Aperfeiçoamento de Pessoal de Nível Superior (CAPES) - Finance Code 001, Fundação Araucária (Grant number 035/2019, 138/2021 and NAPI - Bioinformática), CNPq 440412/2022-6 and 408312/2023-8.\n',
    'author': 'LucasCFuganti',
    'author_email': 'lucascostafuganti@alunos.utfpr.edu.br',
    'maintainer': 'None',
    'maintainer_email': 'None',
    'url': 'None',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'entry_points': entry_points,
    'python_requires': '>=3.13,<4.0',
}
from build import *

build(setup_kwargs)

setup(**setup_kwargs)
