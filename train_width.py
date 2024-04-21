import torch
from torch import nn
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import viz_tools
from model import MLP

dim_width_list = \
    [[200100, 2, 2, 1], \
     [200100, 2, 4, 1], \
     [200100, 2, 8, 1], \
     [200100, 2, 16, 1], \
     [200100, 2, 32, 1], \
     [200100, 2, 64, 1], \
     [200100, 2, 128, 1], \
     [200100, 2, 256, 1]]

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
lr = 0.1
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
    nepoch = 2000 if activation == 'sigmoid' else 2000
    for epoch in range(nepoch):
        loss = train_epoch(epoch, XT, U)
        loss_arr.append(loss.detach().numpy())
        if epoch % 200 == 199:
            print(loss)
    
    loss_all.append(loss_arr)
    print(loss_arr[-1])

U = model.forward(XT)
U = U.detach().cpu().numpy().reshape((2001, 100))
# viz_tools.plot_spatio_temp_3D(X,T,U,'figures/Burgers_NN.pdf')

loss_all = np.array(loss_all)
np.save('data/loss_all_width_lr{}.npy'.format(lr), loss_all)



loss_all_lr = []
for lr in [0.1]: 
    loss_all_lr.append(np.load('data/loss_all_width_lr{}.npy'.format(lr)))
loss_all_lr = np.array(loss_all_lr)

loss_all_01 = np.min(loss_all_lr, axis=0)
print(loss_all_lr.shape, loss_all_01.shape)

# # loss curves
# plt.figure()
# for i in range(8):
#     plt.plot(loss_all_01[i], label=i)
# plt.yscale('log')
# plt.legend()
# plt.savefig('figures/loss_curves_width.pdf')

# L2 loss vs. number of layers
plt.figure()
plt.plot([1.6, 310], [10**(-3/2), 10**(-3/2)], color='black', label=r'$\sqrt{10^{-3}}$')
plt.plot([2, 4, 8, 16, 32, 64, 128, 256], np.sqrt(loss_all_01[:,-1]), linewidth=3)
plt.xlabel('Layer width', fontsize=26)
plt.ylabel(r'$L^2$ error', fontsize=26)
plt.xscale('log')
plt.yscale('log')
plt.xticks([1e1, 1e2], fontsize= 20)
plt.yticks([1e0, 1e-1, 1e-2], fontsize= 20)
plt.xlim([1.6, 310])
plt.legend(fontsize=17)
plt.savefig('figures/loss_layers_width.pdf', bbox_inches='tight')
