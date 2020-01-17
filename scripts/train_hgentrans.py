import os.path as osp

import torch
import torch.nn.functional as F
import numpy as np
import os, sys

install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(install_path)

from models.hyedge import neighbor_distance
from models.hygraph import hyedge_concat
from models import HGEnTrans
from models.utils.meter import trans_class_acc

# initialize parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load data
features = None


# init H and X
# for each domain, generate a X and a H
# dimension of X is (En + Ba) X (768 * len)
# dimension of H is (En + Ba) X (En + Ba)

# x_ch is the max_len_of_sen X 768
# n_class is the number of intents

def generate_target(en_ba_dim):
    en_target = []
    ba_target = []

    for i in range(en_ba_dim.shape[1]):
        en_target += [i] * en_ba_dim[0, i]
        ba_target += [i] * en_ba_dim[1, i]
    return np.array(en_target + ba_target)


d = '/users6/kyzhang/tk/bert/HG_data/taxi'
H_all = np.load(os.path.join(d, 'H.npy'))
en_ba_dim = np.load(os.path.join(d, 'en_ba_sen_nums_metadata.npy'))
X_1d = np.load(os.path.join(d, 'X_1d.npy'))
n_class = H_all.shape[1]
target = generate_target(en_ba_dim)

en_all_num, ba_all_num = np.cumsum(en_ba_dim, axis=1)
en_nums, ba_nums = np.sum(en_ba_dim, axis=1)
i = 0
en_c_num = en_ba_dim[0, i]
ba_c_num = en_ba_dim[1, i]

X_c = np.row_stack((X_1d[: en_c_num], X_1d[en_nums: en_nums + ba_c_num]))
H_c = np.row_stack((H_all[: en_c_num], H_all[en_nums: en_nums + ba_c_num]))
target_c = np.concatenate((target[: en_c_num], target[en_nums: en_nums + ba_c_num]))
n_class_c = 1

model = HGEnTrans(X_c, n_class_c, hiddens=[8])
# model = HGEnTrans()
model, ft, H, target = model.to(device), X_c.to(device), H_c.to(device), target_c.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train():
    model.train()
    optimizer.zero_grad()
    pred = model(ft, H)
    F.nll_loss(pred, target_c.backward())
    optimizer.step()


def val():
    model.eval()
    pred = model(ft, H)

    _train_acc = trans_class_acc(pred, target, target)
    # _val_acc = trans_class_acc(pred, target, mask_val)

    return _train_acc  # , _val_acc


if __name__ == '__main__':
    best_acc, best_iou = 0.0, 0.0
    for epoch in range(1, 101):
        train()
        # train_acc, val_acc = val()
        train_acc = val()
        # if val_acc > best_acc:
        #     best_acc = val_acc
        print(f'Epoch: {epoch}, Train:{train_acc:.4f},'
              f'Best Val acc:{best_acc:.4f}')
