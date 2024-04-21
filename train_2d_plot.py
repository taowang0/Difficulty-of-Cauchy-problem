import torch
from torch import nn
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import viz_tools
from model import MLP
# from utils import compute_Q, loss_2norm, get_U_V

dim_width_list = \
    [[200100, 2, 4, 4, 4, 1]]

activation = 'sigmoid' # ['linear', 'relu', 'tanh', 'sigmoid']
name = 'width'

if not os.path.exists('figures'):
    os.mkdir('figures')

U = np.load('data/U.npy')
X = np.load('data/X.npy')
T = np.load('data/T.npy')
SX, ST = np.meshgrid(T,X)
XT = torch.tensor(np.vstack([ST.ravel(), SX.ravel()]), dtype=torch.float32).T
U = torch.tensor(U.flatten(), dtype=torch.float32)[:, None]
# print(XT.shape, U.shape)

# Training for one epoch
def train_epoch(epoch, x_train, y_train):
    model.train()
    optimizer.zero_grad()
    output = model.forward(x_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    return loss

# plt.figure()
lr = 0.01
loss_all = []

for i in range(len(dim_width_list)):
    print(i, dim_width_list[i])
    dim = dim_width_list[i]

    torch.manual_seed(12345*i)
    model = MLP(dim, activation)
    device = torch.device("cpu")
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_arr = []
    nepoch = 5000 if activation == 'sigmoid' else 2000
    for epoch in range(nepoch):
        loss = train_epoch(epoch, XT, U)
        loss_arr.append(loss.detach().numpy())
        if epoch % 200 == 199:
            print(loss)
    
    loss_all.append(loss_arr)
    print(loss_arr[-1])

U = model.forward(XT)
# print(U.size())
U = U.detach().cpu().numpy().reshape((2001, 100))
viz_tools.plot_spatio_temp_3D(X,T,U,'figures/Burgers_NN.pdf')

loss_all = np.array(loss_all)
# np.save('data/loss_all_width_lr{}.npy'.format(lr), loss_all)

# solution slice
def plot_2d(x, u, u_sol, t):
    plt.figure()
    plt.plot(x, u, linewidth=3, label='NN')
    plt.plot(x, u_sol, linewidth=3, label='solution')
    plt.xlabel('$x$', fontsize=26)
    plt.ylabel('$u$', fontsize=26)
    # plt.xscale('log')
    # plt.yscale('log')
    plt.xticks([-1, -0.5, 0, 0.5, 1], fontsize= 20)
    plt.yticks([-1, -0.5, 0, 0.5, 1], fontsize= 20)
    plt.legend(fontsize=17)
    plt.savefig('figures/t_{}.pdf'.format(t), bbox_inches='tight')

U_sol = np.load('data/U.npy')
X = np.load('data/X.npy')
T = np.load('data/T.npy')
for t in [0, 0.25, 0.5, 0.75, 1.0]:
    idx = int(U_sol.shape[1] * t)
    if idx == U_sol.shape[1]:
        idx -= 1
    plot_2d(X, U[:, idx], U_sol[:, idx], t)

viz_tools.plot_spatio_temp_3D(X,T,U_sol,'figures/Burgers_NN.pdf')
