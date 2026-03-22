
<div align="center" style="display: display_block">

# **BITSER**

#### **BI**nary pa**T**tern **S**equenc**E** **R**ecognition

![image_info](https://img.shields.io/badge/bitser-v0.3.6-blue)

</div>


<div align="center">
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/python/python-original.svg" width="100" height="100" />
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/poetry/poetry-original.svg" width="100" height="100" />
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/scikitlearn/scikitlearn-original.svg" width="100" height="100" />
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/numpy/numpy-original.svg" width="100" height="100" />
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/pandas/pandas-original.svg" width="100" height="100" />
</div>


## Overview

BITSER (Binary Pattern Sequence Recognition) is a software tool built with the Python language that extracts features segments of each genetic sequence at a local level.

The method for feature extraction utilizes the concept of Local Binary Pattern (LBP), as well as adapted versions of the Texture Unit and Texture Unit Number from the field of computer vision, to obtain informative features from sequences organized in FASTA files.

A k-mer window (default size 9) slides over each genetic sequence, comparing the leftmost nucleotide or aminoacid in the window with the 8 other members.

This tool is targeted for usage by biologists, researchers and other professionals in the field of bioinformatics.

## Installation

```bash
pip install bitser
```

After the installation, run `bitser --help` to see all the available commands.

## CLI commands

BITSER offers the following commands:

| COMMAND | FUNCTION                                         |
|---------|--------------------------------------------------|
| `metadata` | Parse FASTA headers and create `metadata.tsv` with train/test splits   |
| `train`    | Extract features and train a classification model                      |
| `predict`  | Load a trained model and predict classes on new sequences              |

### `metadata` command

This command must be run on your training dataset directory.

The dataset directory **must contain a `sequences/` subfolder** with the FASTA files.

The command scans all `.fasta` files in the `sequences/` subfolder, parses headers to extract class labels, and creates a `metadata.tsv` file, which is used for the `train` and `predict` commands.

Example structure:

```
dataset/
├── sequences/
│ ├── class_a.fasta
│ └── class_b.fasta
```

After executing the command:

```
dataset/
├── sequences/
│ ├── class_a.fasta
│ └── class_b.fasta
└──metadata.tsv
```

### `train` command

This command initiates the feature extraction and model training workflow.

Training performs the following steps:

1. Feature extraction using sliding windows;
2. Construction of the training feature matrix;
3. Model training using cross-validation;
4. Saving the trained model.

#### Parameters

| Parameter | Description | Required | Default |
|---|---|:--:|---|
| `--input`, `-i` | Path to the dataset directory containing `metadata.tsv` and `sequences/` | ✔ | |
| `--output`, `-o` | Path to save the trained model | | `model.pkl` |
| `--classifier`, `-c` | Classifier algorithm | | `xgb` |
| `--flank`, `-f` | Number of neighbors compared to the reference character in the sliding window | | `8` |
| `--translate / --no-translate` | Translate nucleotide sequences to proteins before feature extraction | | `False` |
| `--splits`, `-s` | Number of folds used for cross-validation | | `10` |
| `--repeats`, `-r` | Number of cross-validation repetitions | | `10` |
| `--seed` | Random seed for reproducibility | | `7` |

#### Output

- Trained model file (`.pkl`);
- Training evaluation results stored in the `results/` directory.

### `predict` command

Uses a trained model to classify sequences from a testing dataset.

Feature extraction settings must match those used during training.

#### Parameters

| Parameter | Description | Required | Default |
|---|---|:--:|---|
| `--model`, `-m` | Path to the trained model file | ✔ | |
| `--data`, `-d` | Dataset directory containing `metadata.tsv` and `sequences/` | ✔ | |
| `--flank`, `-f` | Number of neighbors compared to the reference character in the sliding window, must match value used during training | | `8` |
| `--translate / --no-translate` | Must match the translation setting used during training | | `False` |

#### Output

The prediction step generates:

- Classification results;
- Per-class performance metrics;
- Feature importance;
- Prediction report (CSV).

##### Acknowledgements

* This study was supported by national funds through the Coordenação de Aperfeiçoamento de Pessoal de Nível Superior (CAPES) - Finance Code 001, Fundação Araucária (Grant number 035/2019, 138/2021 and NAPI - Bioinformática), CNPq 440412/2022-6 and 408312/2023-8.
