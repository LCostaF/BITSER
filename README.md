
<div align="center" style="display: display_block">

# **BITSER**

#### **BI**nary pa**T**tern **S**equenc**E** **R**ecognition

![image_info](https://img.shields.io/badge/bitser-v0.0.1-blue)

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



## CLI commands

BITSER offers the following commands:

| COMMAND | FUNCTION                                         |
|---------|--------------------------------------------------|
| train   | Train a classification model from sequence data  |
| predict | Predict classes for new sequences using a trained model |

### `train` command

This command initiates the feature extraction and model training workflow, and should be used on a training dataset. It has the following parameters:

| PARAMETER | DESCRIPTION                                               | OPTIONAL | DEFAULT VALUE |
|-----------|-----------------------------------------------------------|:--------:|-----------|
| ``input`` | Path to the directory containing FASTA files for training |    ❌     |           |
| ``output`` | Path to save the trained model to a file                  |     ✔️     | model.pkl |
| ``classifier``| Type of classifier that will be used to train the model   | ✔️ | xgb       |
| ``flank`` | How many characters in a sequence will be compared to the leftmost member of the sliding window | ✔️ | 8
| ``translate`` | Whether the sequence should be translated to aminoacids or not | ✔️ | False |

### `predict` command

This command initiates the feature extraction on the testing dataset, and then predicts classes based on the trained model. It has the following parameters:

| PARAMETER      | DESCRIPTION                                                                                                                                    | OPTIONAL  | DEFAULT VALUE |
|----------------|------------------------------------------------------------------------------------------------------------------------------------------------|:---------:|--|
| ``model``      | Path to trained model obtained after training                                                                                                  |     ❌     |  |
| ``data``       | Path to the directory containing FASTA files for testing                                                                                       |     ❌     |  |
| ``flank``      | How many characters in a sequence will be compared to the leftmost member of the sliding window (must match value used in the `train` command) |    ✔️     | 8
| ``translate``  | Whether the sequence should be translated to aminoacids or not (must match value used in the `train`command)                                   |    ✔️     | False |

## Example Usage

Considering the following example project structure:

```
──project
  └───datasets
      ├───training_data
      │   ├───class_a.fasta
      │   └───class_b.fasta
      └───testing_data
          ├───class_a.fasta
          └───class_b.fasta  
```

BITSER could be run from the project root, with the following command:

`bitser train --input .\datasets\training_data\ --output example_model.pkl`

The training data would be used to construct a classification model, which are saved to a Pickle file. The accuracy values for cross-validation would be saved to a text file.

The updated project structure after running the train command:

```
──project
  ├───datasets
  │   ├───training_data
  │   │   ├───class_a.fasta
  │   │   └───class_b.fasta
  │   └───testing_data
  │       ├───class_a.fasta
  │       └───class_b.fasta
  ├───example_model.pkl
  └───results
      └───20250401_094215_results_xgb.txt
```

The saved model can then be used to predict class values, with the following command:

`bitser predict --model example_model.pkl --data .\datasets\testing_data\`

The model would be used to evaluate the testing data, and classify sequences accordingly. The classification results, per-class accuracies, and feature importance data are saved to a text file.

The updated project structure after running the predict command:

```
──project
  ├───datasets
  │   ├───training_data
  │   │   ├───class_a.fasta
  │   │   └───class_b.fasta
  │   └───testing_data
  │       ├───class_a.fasta
  │       └───class_b.fasta
  ├───example_model.pkl
  └───results
      ├───20250401_094215_results_xgb.txt
      └───20250401_094215_results_xgbclassifier.txt
```


##### Acknowledgements

* This study was supported by national funds through the Coordenação de Aperfeiçoamento de Pessoal de Nível Superior (CAPES) - Finance Code 001, Fundação Araucária (Grant number 035/2019, 138/2021 and NAPI - Bioinformática), CNPq 440412/2022-6 and 408312/2023-8.
