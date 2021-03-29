# Reward Progpagation using Graph Convolutional Networks in GridWorld

The repository contains the code for running the experiments on sparse reward environments in 2D-Gridworld, based on the paper [Reward Propagation using Graph Convolutional Networks](https://arxiv.org/abs/2010.02474) using the Proto Value Functions by [Mahadevan and Maggioni](https://www.jmlr.org/papers/volume8/mahadevan07a/mahadevan07a.pdf) as features to the GCN. The underlying MDP of the Gridworld is captured as a graph whihc is then used to calculate the Proto Value Functions.The implementation is GCN is baseed on Thomas Kipf's [pytorch GCN implementation](https://github.com/tkipf/pygcn). The environment currently is only a GridWorld and all the results have been produced using this environment. The actor critic network implementation was not from any library but our own implementation using linear function approximators. 



## Getting Started

For a quick start clone the repository, and type the following command.
```
$ git clone <repo link>
```

```
$ python main.py 
```


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


## Results

![Reward Propagation](/images/1.png)
![Regret Plot](/images/2.png)
![Loss Plot](/images/3.png)


## Built With

* [Pytorch GCN implementation](https://github.com/tkipf/pygcn)
* [Python](https://python.org)

## Authors

* **Alisetti Sai Vamsi** - [Vamsi995](https://github.com/Vamsi995)

* **Raswanth Murugan** - [RaswanthMurugan20](https://github.com/RaswanthMurugan20)


## Acknowledgments

* **Dr.Chandra Shekar Lakshminarayan, IIT Palakkad**

