from ast import parse
import torch
import torch.nn.functional as F 
from torch.utils import data
from torch import nn
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils import get_model


features = np.load('data/features_0.npy')[-5000:]
labels = np.load('data/labels_0.npy')[-5000:]
for i in range(len(labels)):
    if labels[i] == 2.0:
        labels[i] = 0.0

features = torch.tensor(features.astype(np.float32))
labels = torch.tensor(labels.astype(np.float32))

# ---------------------------------------------------------
# ---------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.0001)
parser.add_argument('-b_s', default=10)
parser.add_argument('--n_epoch', default=10)
args = parser.parse_args()

lr = float(args.lr)
batch_size = int(args.b_s)
num_epoch = int(args.n_epoch)

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, 0, 0.01)
    return None
net = get_model()
net.apply(init_weights)
loss = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(), lr = lr)

def load_array(data_arrays, batch_size, is_train=False):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)
data_iterator = load_array((features, labels), batch_size, is_train=True)

hist = []
for epoch in range(num_epoch):
    for X, y in data_iterator:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    hist.append(l.detach().numpy())
    print(f'epoch {epoch + 1}, loss {l:f}')

X = np.array(range(len(hist)))
Y = np.array(hist).flatten()
with plt.style.context(['science', 'no-latex']):
        fig, ax = plt.subplots()
        ax.plot(X,Y, label='total loss')
        ax.legend(title='')
        # ax.autoscale(tight=True)
        pparam = dict(xlabel='$epoch$', ylabel='$loss$')
        ax.set(**pparam)
        # fig.savefig('figs/fig1.pdf')
        fig.savefig(f'figs/lr_{lr}_bs_{batch_size}.jpg', dpi=300)


torch.save(net.state_dict(), 'model/net.pt')

# net.eval()
# net(features[-3:])