# Doloris
Dual Conditional Diffusion Implicit Bridges with Sparsity Masking Strategy for Unpaired Single-Cell Perturbation Estimation (ICLR2026)

### 1. Dataset Preparation
The dataset can be downloaded from the following URL:
```bash
# Genetic Perturbation Dataset:
# Download data to /Dataset/gene/
wget -O Dataset/gene/Adamson.h5ad https://dataverse.harvard.edu/api/access/datafile/6154417
```
```bash
wget -O Dataset/gene/Norman.h5ad https://dataverse.harvard.edu/api/access/datafile/6154020
```
```bash
# Molecular Perturbation Dataset:
# Download data to /Dataset/molecular/

The original dataset was obtained from [Paper](https://arxiv.org/abs/2204.13545).
Users can follow the splitting described in the paper to create their own train/test partitions. 
We will provide the pre-split data and the corresponding code in the following days.
```


### 2. Environment Requirements

The environment should be configured according to the requirements specified in `requirements.txt`. We recommend first setting up the base environment of **`scGPT`**.

Download files to their corresponding folders according to the instructions in `note.txt` inside folders `/Dataset/scGPT/` and `/Dataset/Uni-Mol/`.


### 3. Training

As an example, using the Adamson dataset, we need to train diffusion models:
```bash
python main_target.py --config config/adamson/adamson.yaml
```
and train mask model:
```bash
python main_mask_model.py --config config/adamson/adamson_mask.yaml
```

### 4. Test

```bash
python test.py --config config/adamson/adamson_test.yaml
```
