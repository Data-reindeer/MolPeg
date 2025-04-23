# Beyond Efficiency: Molecular Data Pruning for Enhanced Generalization

This repository provides the source code for the paper **Beyond Efficiency: Molecular Data Pruning for Enhanced Generalization**.

## Environments

```markdown
numpy             1.21.2
scikit-learn      1.0.2
pandas            1.3.4
python            3.7.11
torch             1.10.2+cu113
torch-geometric   2.0.3
transformers      4.17.0
rdkit             2020.09.1.0
ase               3.22.1
descriptastorus   2.3.0.5
ogb               1.3.3
```

## Python environment setup with Conda

```shell
conda create --name MolPeg python=3.7.11
conda activate MolPeg

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.10.2%2Bcu113.html

pip install -r requirements.txt
```



## Experiments 

- **Classification on HIV and PCBA**

Please use the default settings in config.py for unspecified hyperparameters.

```bash
# ratio: [0.1, 0.2, 0.3, 0.4, 0.6, 0.8]
# dataset:       [hiv, pcba]
python main.py --ratio=0.1 --dataset=hiv --pretrain
```
