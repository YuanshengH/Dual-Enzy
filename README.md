# Dual-Grained Cross-Modal Molecular Representation Learning for Enzymatic Reaction Modeling
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg) <br>
Implementation of A Dual-Grained Molecular Learning Approach Integrating Knowledge Graphs and Contrastive Learning<br>
![model_overview](./figure/model_overview.png)


## Table of contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Setup Environment](#setup-environment)
  - [Clone the repository to your local machine](#clone-the-repository-to-your-local-machine)
  - [Install the environment](#install-the-environment)

- [Reproduce Results](#reproduce-results)

  - [Input formats](#Input-formats)
  - [Supported residues](#supported-residues)
  - [Docking prediction](#docking-prediction)
  - [Visualization](#Visualization)


## Introduction

We introduce a dual-grained , cross-modal enzymatic-reaction-aware molecular learning method named ERAM which is capable of (i) simultaneously predicting the feasibility and the ideal enzymes for a given reaction (and vice versa), by a dual-grained contrastive learning scheme, (ii) capturing the catalytic interactions between the substrates and the enzyme via a joint cross-modal modeling of their individual representations.

## Prerequisites
* Python (version >= 3.10) 
* PyTorch (version >= 2.0.1) 
* RDKit (version >= 2023.9.5)
* fair-esm (version == 2.0.0)
* lmdb 
* loguru 
* scikit-learn (version >= 1.3.2)
* unimol-tools(version == 1.0.0)
* unicore(version == 0.0.1)
* transformers
* wandb

## Setup Environment

### Clone the repository to your local machine

```
git clone https://github.com/YuanshengH/Dual-Enzy.git
```

### Install the environment

```
conda create -n ERAM python=3.10
conda activate ERAM
cd Dual-Enzy
pip install -r requirement.txt
```

## Reproduce Results

### Data process

```
python dataprocess.py --task data_process --data_path ./data/rhea_data.csv
python dataprocess.py --task unimol --data_path ./data/rhea_processed_data.csv
python dataprocess.py --task esm_extract --data_path ./data/rhea_processed_data.csv
```

### Training the model
```
torchrun --nproc_per_node {num_GPU} train_ddp.py --batch_size 64 --wandb_api_key {wandb_api_key}
``` 