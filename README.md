
<div align="center" style="display: display_block">

# **BITSER**

#### **BI**nary pa**T**tern **S**equenc**E** **R**ecognition

![image_info](https://img.shields.io/badge/bitser-v0.4.21-blue)

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

A k-mer window of size 9 slides over each genetic sequence, comparing the leftmost nucleotide or aminoacid in the window with the 8 other members.

This tool is targeted for usage by biologists, researchers and other professionals in the field of bioinformatics.

## Installation

```bash
pip install bitser
```

After the installation, run `bitser --help` to see all the available commands.

## CLI commands

BITSER follows a three-step workflow:

1. `metadata` → generate `metadata.tsv` with train/test split  
2. `train` → extract features from train split and train model  
3. `predict` → evaluate model on test split and generate reports  

| COMMAND   | FUNCTION                                                                 |
|-----------|--------------------------------------------------------------------------|
| `metadata` | Parse FASTA headers and create `metadata.tsv` with train/test splits     |
| `train`    | Extract features from training split and train classification model      |
| `predict`  | Load trained model, evaluate on test split, and generate reports         |

---

### `metadata` command

Generates `metadata.tsv` by parsing FASTA headers and splitting data into train/test sets.

The dataset directory **must contain a `sequences/` subfolder** with FASTA files.

#### Parameters

| Parameter | Description | Required | Default |
|---|---|:--:|---|
| `--dataset`, `-d` | Dataset directory containing `sequences/` | ✔ | |
| `--class-delim`, `-delim` | Delimiter used to extract class label from FASTA headers | ✔ | |
| `--train-count`, `-n` | Number of sequences per class used for training | ✔ | |
| `--class-which`, `-which` | Which occurrence of the delimiter to use (1 = first, -1 = last) | | `1` |
| `--seed` | Random seed for reproducibility | | `7` |

#### Output

- `metadata.tsv` file containing dataset splits

---

### `train` command

Performs feature extraction and trains a classification model using the **training split only**.

#### Parameters

| Parameter | Description | Required | Default |
|---|---|:--:|---|
| `--input`, `-i` | Dataset directory containing `metadata.tsv` and `sequences/` | ✔ | |
| `--output-dir`, `-dir` | Directory where outputs (model + logs) will be saved | ✔ | |
| `--output`, `-o` | Model filename (e.g., `model.pkl`) | ✔ | |
| `--classifier`, `-c` | Classifier: `xgb`, `rf`, `svm`, `mlp`, `nb` | | `xgb` |
| `--flank`, `-f` | Sliding window size for feature extraction | | `8` |
| `--translate / --no-translate` | Translate nucleotide sequences to proteins | | `False` |
| `--splits`, `-s` | Number of cross-validation folds | | `10` |
| `--repeats`, `-r` | Number of cross-validation repetitions | | `10` |
| `--seed` | Random seed for reproducibility | | `7` |

#### Output

- Trained model (`.pkl`) saved inside `--output-dir`
- Training logs and evaluation results saved to `--output-dir`

---

### `predict` command

Loads a trained model and evaluates it on the **test split**, generating predictions and reports.

#### Parameters

| Parameter | Description | Required | Default |
|---|---|:--:|---|
| `--model`, `-m` | Path to trained model file | ✔ | |
| `--output-dir`, `-dir` | Directory where prediction outputs will be saved | ✔ | |
| `--data`, `-d` | Dataset directory containing `metadata.tsv` and `sequences/` | ✔ | |
| `--flank`, `-f` | Sliding window size (must match training) | | `8` |
| `--translate / --no-translate` | Must match training configuration | | `False` |

#### Output

- Classification results
- Per-class performance metrics
- Confusion matrix (if applicable)
- Prediction report (CSV) saved to `--output-dir`

##### Acknowledgements

* This study was supported by national funds through the Coordenação de Aperfeiçoamento de Pessoal de Nível Superior (CAPES) - Finance Code 001, Fundação Araucária (Grant number 035/2019, 138/2021 and NAPI - Bioinformática), CNPq 440412/2022-6 and 408312/2023-8.
