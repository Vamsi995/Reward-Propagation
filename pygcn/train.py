from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from pygcn.models import GCN
from pygcn.utils import normalize, sparse_mx_to_torch_sparse_tensor, load_data, accuracy
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.linalg as la
#
# parser = argparse.ArgumentParser()
# parser.add_argument('--hidden', type=int, default=64,
#                     help='Number of hidden units.')
# parser.add_argument('--lr', type=float, default=0.01,
#                     help='Initial learning rate.')
# parser.add_argument('--weight_decay', type=float, default=0e-4,
#                     help='Weight decay (L2 loss on parameters).')
# parser.add_argument('--no-cuda', action='store_true', default=False,
#                     help='Disables CUDA training.')
# parser.add_argument('--gcn_epochs', type=int, default=20,
#                     help='Number of epochs to train.')
# parser.add_argument('--seed', type=int, default=42, help='Random seed.')
# parser.add_argument('--gcn_lambda', type=float, default=5,
#                     help='Mixing coefficient between GCN losses.')
# parser.add_argument('--dropout', type=float, default=0.5,
#                     help='Dropout rate (1 - keep probability).')
# parser.add_argument('--fastmode', action='store_true', default=False,
#                     help='Validate during training pass.')
#
# args = parser.parse_args()
# args.cuda = not args.no_cuda and torch.cuda.is_available()


def state_selector(n, i, m, j, action):
    x, y = i, j
    if action == 0:
        x = i - 1
    elif action == 1:
        y = j + 1
    elif action == 2:
        x = i + 1
    else:
        y = j - 1

    if x > n - 1:
        x = n - 1
    elif x < 0:
        x = 0

    if y > m - 1:
        y = m - 1
    elif y < 0:
        y = 0

    return int(x), int(y)


def GraphCons(n, m, nt, mt, sam_len):
    i = 0
    j = 0
    states = []
    policy = [0.25, 0.25, 0.25, 0.25]
    a = np.random.choice([0, 1, 2, 3], size=1, p=policy)
    states.append(0)
    epi_len = 0
    labels = np.zeros((n * m))
    states.append(0)

    # epi_len <= sam_len and
    while ((i != nt - 1 or j != mt - 1)):

        i_t, j_t = state_selector(n, i, m, j, a)
        a_t = np.random.choice([0, 1, 2, 3], size=1, p=policy)
        if (i_t == nt - 1 and j_t == mt - 1):
            labels[m * i + j] = 1
        states[1] = m * i + j
        i = i_t
        j = j_t
        a = a_t
        epi_len += 1

    return states, labels


def shortest_dist(n, m, goal_states):
    # Adjacency Matrix
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

    # Labels and index 

    idx_train, labels = GraphCons(n, m, goal_states[0], goal_states[1], 1500)
    labels = torch.LongTensor(labels)
    idx_train = torch.LongTensor(idx_train)

    return adj, features, labels, idx_train


def update_graph(n, m, args, model, optimizer):
    # adj, full_adj, features, labels, idx_train, idx_val, idx_test,edges = load_data()
    adj, features, labels, idx_train = shortest_dist(n, m, [n, m])

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    deg = np.diag(adj.toarray().sum(axis=1))
    laplacian = torch.from_numpy((deg - adj.toarray()).astype(np.float32))
    adj = normalize(sp.csr_matrix(adj) + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    if args.cuda and torch.cuda.is_available():
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        laplacian = laplacian.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()

    t_total = time.time()
    for epoch in range(args.gcn_epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        soft_out = torch.unsqueeze(torch.nn.functional.softmax(output, dim=1)[:, 1], 1)
        loss_reg = torch.mm(torch.mm(soft_out.T, laplacian), soft_out)
        # print('Epoch: {:04d}'.format(epoch + 1),
        #       'loss_train: {:.4f}'.format(loss_train.item()),
        #       'time: {:.4f}s'.format(time.time() - t))
        loss_train += args.gcn_lambda * loss_reg.squeeze()
        loss_train.backward()
        optimizer.step()

#
# torch.set_num_threads(1)
# device = torch.device("cuda:0" if args.cuda else "cpu")
# n, m = 15, 15
# gcn_model = GCN(nfeat=n * m, nhid=args.hidden)
# gcn_model.to(device)
# optimizer = optim.Adam(gcn_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
# adj, features, _, _ = shortest_dist(n, m, [n, m])
#
# features = normalize(sp.csr_matrix(features))
# features = torch.FloatTensor(np.array(features.todense()))
#
# adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
# adj = normalize(sp.csr_matrix(adj) + sp.eye(adj.shape[0]))
# adj = sparse_mx_to_torch_sparse_tensor(adj)
#
# for episodes in range(40):
#     update_graph(n, m, args, gcn_model, optimizer)
#     print("{} episode done :".format(episodes + 1))
#
# if args.cuda and torch.cuda.is_available():
#     features = features.cuda()
#     adj = adj.cuda()
#
# output = gcn_model(features, adj).cpu()
# gcn_phi = torch.exp(output).detach().numpy()
# gcn_phi = gcn_phi[:, 1].reshape(n, m)
# plt.imshow(gcn_phi, cmap='hot', interpolation='nearest')
# sns.heatmap(gcn_phi, vmin=0, vmax=1)
# plt.show()


