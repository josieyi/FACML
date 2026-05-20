# FACML: Foundation-Assisted Cross-Domain Meta-Learning for Few-Shot Molecular Property Prediction

Implementation for paper: **Foundation-Assisted Cross-Domain Meta-Learning for Few-Shot Molecular Property Prediction**

This repository provides the implementation of FACML, a foundation-assisted cross-domain meta-learning framework for few-shot molecular property prediction. The method uses SPMM-derived SMILES features, graph features, and SPMM-predicted Property Vector (PV) representations to support target-domain adaptation under limited supervision.


## Requirements

- Python >= 3.9
- PyTorch
- PyTorch Geometric
- RDKit
- transformers
- numpy
- scikit-learn
- tqdm


## Usage

The main running parameters are as follows:

| Parameter | Description                 | Default Value | Choices                                                                                                                                     |
| --- |-----------------------------|---------------|---------------------------------------------------------------------------------------------------------------------------------------------|
| dataset | name of source dataset      | tox21         | `tox21, sider, toxcast-APR, toxcast-ATG, toxcast-BSK, toxcast-CEETOX, toxcast-CLD, toxcast-NVS, toxcast-OT, toxcast-Tanguay, toxcast-TOX21` |
| test_dataset | name of target dataset      | sider         | `tox21, sider, toxcast-APR, toxcast-ATG, toxcast-BSK, toxcast-CEETOX, toxcast-CLD, toxcast-NVS, toxcast-OT, toxcast-Tanguay, toxcast-TOX21` |
| inner_lr | inner-loop learning rate | `0.1` | - |
| meta_lr | meta learning rate | `1e-3` | - |
| n_support | number of support molecules | `10`            | `1, 10`                                                                                                                                     |
| n_query | number of query molecules   | `16`            | -                                                                                                                                           |
| gpu | which GPU to use            | `0`             | -                                                                                                                                           |



### Step 1: Generate SPMM-derived PV representations

```
python generate_PV.py --dataset {source dataset} --test_dataset {target dataset}
```

### Step 2: Run cross-domain few-shot molecular property prediction

For the 1-shot setting:
```
python run.py --dataset {source dataset} --test_dataset {target dataset} --gpu 0 --n_support 1 --n_query 16 --inner_lr 0.1 --meta_lr 1e-3
```

For the 10-shot setting:
```
python run.py --dataset {source dataset} --test_dataset {target dataset} --gpu 0 --n_support 10 --n_query 16 --inner_lr 0.1 --meta_lr 1e-3
```
**For example, to run the Tox21-to-SIDER 10-shot experiment:**

```
python generate_PV.py --dataset tox21 --test_dataset sider
python run.py --dataset tox21 --test_dataset sider --gpu 0 --n_support 10 --n_query 16 --inner_lr 0.1 --meta_lr 1e-3
```