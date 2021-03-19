# Reward Propagation using Graph Convolutional Networks
The repository contains the code for running the experiments in the paper [Reward Propagation using Graph Convolutional Networks](https://arxiv.org/abs/2010.02474) using the Proto Value Functions. The implementation is based on a few source codes: [gym-miniworld](https://github.com/maximecb/gym-miniworld), a  and Thomas Kipf's [pytorch GCN implementation](https://github.com/tkipf/pygcn).

# Installation

```bash
# PyTorch
conda install pytorch torchvision -c soumith

# Other requirements
pip install -r requirements.txt

#Installing PyGCN
python setup_gcn.py install
```

# Usage

## For GridWorld Implementation

```python main.py --env_dim 5 5 --gcn_epochs 100 --gcn_lambda 10 --gcn_alpha 0.6 --episodes 2000```
