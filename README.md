<div align="center" style="display: display_block">

# **BITSER**

#### **BI**nary pa**T**tern **S**equenc**E** **R**ecognition

![image_info](https://img.shields.io/badge/bitser-v0.5.21-blue)

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

1. `metadata` → parse FASTA headers and generate `metadata.tsv` describing the full dataset (no split is written)
2. `train` → internally split the dataset, run cross-validation + hyperparameter tuning, and train the final model
3. `predict` → reconstruct the exact held-out test split from the seed stored in the model, and evaluate on it

| COMMAND   | FUNCTION                                                                 |
|-----------|--------------------------------------------------------------------------|
| `metadata` | Parse FASTA headers and create `metadata.tsv` describing the full dataset |
| `train`    | Split the dataset, extract features, and train a classification model    |
| `predict`  | Load a trained model, reconstruct its test split, and generate reports   |

Run `bitser --help` or `bitser <command> --help` at any time to see this information from the CLI itself.

---

### `metadata` command

Generates `metadata.tsv` by parsing FASTA headers. The resulting file describes the **full dataset** (columns: `sample-id`, `fasta_path`, `class`, `record_index`).

The dataset directory **must contain a `sequences/` subfolder** with FASTA files.

#### Parameters

| Parameter | Description | Required | Default |
|---|---|:--:|---|
| `--dataset`, `-d` | Dataset directory containing `sequences/` | ✔ | |
| `--class-delim`, `-delim` | Delimiter string before the class label in FASTA headers (e.g. `" "`, `"|"`, `"genotype "`) | ✔ | |
| `--class-which`, `-which` | Which occurrence of the delimiter to use (`1` = first, `-1` = last) | | `1` |

The extracted class token is automatically cleaned to contain only alphanumeric characters.

#### Output

- `metadata.tsv` describing the full dataset (no split included)

#### Example

```bash
bitser metadata -d mydata/ -delim "_"
```

---

### `train` command

Loads the full dataset from `metadata.tsv`, performs a stratified train/test split internally, runs cross-validation with hyperparameter tuning on the training subset, and trains the final classification model. The test subset is never seen during tuning or training, and the split definition (including the seed) is persisted inside the saved model so `predict` can reconstruct it exactly.

#### Parameters

| Parameter | Description | Required | Default |
|---|---|:--:|---|
| `--input`, `-i` | Dataset directory containing `metadata.tsv` and `sequences/` | ✔ | |
| `--output-dir`, `-dir` | Directory where outputs (model + logs) will be saved | ✔ | |
| `--output`, `-o` | Model filename (e.g., `model.pkl`) | ✔ | |
| `--classifier`, `-c` | Classifier: `xgb`, `rf`, `svm`, `mlp`, `nb` | | `xgb` |
| `--flank`, `-f` | Sliding window size for feature extraction | | `8` |
| `--translate / --no-translate` | Translate nucleotide sequences to proteins | | `False` |
| `--splits`, `-s` | Number of cross-validation folds | | `5` |
| `--repeats`, `-r` | Number of cross-validation repetitions (for variance estimation) | | `1` |
| `--test-size` | Fraction of data held out for testing | | `0.20` |
| `--seed` | Random seed for splitting, CV, and training. Auto-generated and logged if omitted | | auto-generated |

#### Output

- Trained model (`.pkl`), including the persisted split definition, saved inside `--output-dir`
- Training logs and cross-validation results saved to `--output-dir`

#### Example

```bash
bitser train -i mydata/ -dir results/ -o model.pkl --seed 42
```

---

### `predict` command

Loads a trained model and reconstructs its **exact held-out test split** from the seed and split definition stored inside the model file, guaranteeing zero leakage from training. It then evaluates the model on that split and generates reports.

`--flank` and `--translate/--no-translate`, if specified, must match the values used during `train`.

#### Parameters

| Parameter | Description | Required | Default |
|---|---|:--:|---|
| `--model`, `-m` | Path to trained model file | ✔ | |
| `--output-dir`, `-dir` | Directory where prediction outputs will be saved | ✔ | |
| `--data`, `-d` | Dataset directory containing `metadata.tsv` and `sequences/` | ✔ | |
| `--flank`, `-f` | Sliding window size (must match training) | | `8` |
| `--translate / --no-translate` | Must match training configuration | | `False` |

#### Output

- Test AUROC and per-class performance metrics
- Confusion matrix (if applicable)
- Per-sample prediction report (CSV) saved to `--output-dir`
- Reference to the run ID used for the reconstructed split

#### Example

```bash
bitser predict -m results/model.pkl -dir results/ -d mydata/
```

## Singularity container

A ready-to-use Singularity (Apptainer) container is available on Zenodo, so BITSER can be run without setting up a local Python/Poetry environment.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21220797.svg)](https://doi.org/10.5281/zenodo.21220797)

Once you have Singularity/Apptainer installed, click below to download the container:

**Click [HERE](https://zenodo.org/records/21220797/files/bitser.sif?download=1) to download the `bitser.sif` container.**

You can also download it directly with `wget`:

```bash
wget "https://zenodo.org/records/21220797/files/bitser.sif?download=1" -O bitser.sif
```

Once downloaded, the container can be run in place of the `bitser` command, for example:

```bash
singularity exec bitser.sif bitser metadata -d mydata/ -delim "_"
```

---

##### Acknowledgements

* This study was supported by national funds through the Coordenação de Aperfeiçoamento de Pessoal de Nível Superior (CAPES) - Finance Code 001, Fundação Araucária (Grant number 035/2019, 138/2021 and NAPI - Bioinformática), CNPq 440412/2022-6 and 408312/2023-8.
