from __future__ import division
from __future__ import print_function

from A2C import Environment, Agent, ActorCritic
import torch
import time
import argparse
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from pygcn.models import GCN
from pygcn.utils import normalize,sparse_mx_to_torch_sparse_tensor,load_data,accuracy
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.linalg as la
from pygcn.train import shortest_dist, update_graph
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--gcn_epochs', type=int, default= 20,
                    help='Number of epochs to train.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--gcn_lambda', type=float, default=5,
                        help='Mixing coefficient between GCN losses.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')


parser.add_argument('--env_dim', type=int, nargs="+", default=5,
                    help='Number of rows')

parser.add_argument('--gcn_alpha', type=float, default=0.5,
                    help='Q Comb alpha')

parser.add_argument('--episodes', type=int, default=100,
                    help='Number of Episodes')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


def walk_features(n,m):
    adj = np.zeros((n * m, n * m))
    D = np.zeros((n * m, n * m))

    for i in range(n):
        for j in range(m):
            currcod = i * m + j
            north, east, south, west = m * (i - 1) + j, m * (i) + j + 1, m * (i + 1) + j, m * (i) + j - 1

            if i - 1 >= 0:
                adj[currcod, north] = 1
            if j + 1 <= m - 1:
                adj[currcod, east] = 1
            if i + 1 <= n - 1:
                adj[currcod, south] = 1
            if j - 1 >= 0:
                adj[currcod, west] = 1

            D[currcod, currcod] = sum(adj[currcod, :])

    D_hat = la.fractional_matrix_power(D, -0.5)
    L_norm = np.identity(n * m) - np.dot(D_hat, adj).dot(D_hat)
    eigvals, features = la.eig(L_norm)
    features = normalize(sp.csr_matrix(features))
    features = torch.FloatTensor(np.array(features.todense()))

    adj = sp.coo_matrix(adj)


    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    deg = np.diag(adj.toarray().sum(axis=1))
    laplacian = torch.from_numpy((deg - adj.toarray()).astype(np.float32))
    adj = normalize(sp.csr_matrix(adj) + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    if args.cuda and torch.cuda.is_available():
        features = features.cuda()
        adj = adj.cuda()

    return features, adj

def walk_train_gcn(env_dim, model, optimizer):
    torch.set_num_threads(1)
    n, m = env_dim
    adj, features, _, _ = shortest_dist(n, m, [n, m])
    features = normalize(sp.csr_matrix(features))
    features = torch.FloatTensor(np.array(features.todense()))

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(sp.csr_matrix(adj) + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    for episodes in range(40):
      update_graph(n, m, args, model, optimizer)
    # print("{} episode done :".format(episodes + 1))



def qcomb(phi, ac, env_dim):

    qc = np.zeros((4,env_dim[0], env_dim[1]))
    for i in range(env_dim[0]):

        for j in range(env_dim[1]):
            qc[:, i, j] = ac.find_qvalues((i,j))  - (1 - args.gcn_alpha) * phi[i,j]

    return qc

def main():

    env_dim = tuple(args.env_dim)
    # env = Environment((0,0), (7,7), env_dim)
    # ag = Agent(env.size)
    # ac = ActorCritic(1, env, ag)

    # env_ac = Environment((0,0), (7,7), env_dim)
    # ag_ac = Agent(env_ac.size)
    # ac_t = ActorCritic(1, env_ac, ag_ac)

    num_ep = args.episodes

    # Define the GCN Model
    device = torch.device("cuda:0" if args.cuda else "cpu")
    model = GCN(nfeat=env_dim[0]*env_dim[1], nhid=args.hidden)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    features, adj = walk_features(env_dim[0], env_dim[1])


    walk_train_gcn(env_dim, model, optimizer)
    output = model(features, adj).cpu()
    gcn_phi = torch.exp(output).detach().numpy()
    gcn_phi = gcn_phi[:, 1].reshape((env_dim))



    gcn_ac = np.zeros((args.episodes))
    ac_og = np.zeros((args.episodes))

    for j in range(10):
      env = Environment((0,0), (4,4), env_dim)
      ag = Agent(env.size)
      ac = ActorCritic(1, env, ag)

      env_ac = Environment((0,0), (4,4), env_dim)
      ag_ac = Agent(env_ac.size)
      ac_t = ActorCritic(1, env_ac, ag_ac)

      for i in range(num_ep):
      
      # plt.imshow(gcn_phi, cmap='hot', interpolation='nearest')
      # sns.heatmap(gcn_phi, vmin=0, vmax=1)
      # plt.show()

        qc = qcomb(gcn_phi, ac, env_dim)

        ac.main(1, qc)

        print("Episode no.", i)
        ac.printPolicy()


      ac_t.main_ac(args.episodes)
      # ac.plot_error()
      # plt.plot(ac.regfin, color='red')
      # plt.plot(ac_t.regfin, color='blue')
      
      # plt.show()

      gcn_ac += np.array(ac.regfin)
      ac_og += np.array(ac_t.regfin)

    
    plt.plot(gcn_ac/10, color='red')
    plt.plot(ac_og/10, color='blue')
    plt.show()


    

if __name__ == "__main__":
    main()
