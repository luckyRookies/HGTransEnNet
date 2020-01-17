import os
import sys

install_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(install_path)

import torch.nn.functional as F
from torch import nn

from hyconv import HyConv


class HGEnTrans(nn.Module):
    def __init__(self, in_ch, n_class, hiddens=(16,), dropout=0.5) -> None:
        super().__init__()
        self.dropout = dropout
        _in = in_ch
        self.hyconvs = []
        for _h in hiddens:
            _out = _h
            self.hyconvs.append(HyConv(_in, _out))
            _in = _out
        self.hyconvs = nn.ModuleList(self.hyconvs)
        self.last_hyconv = HyConv(_in, n_class)

    def forward(self, x, H, hyedge_weight=None):
        for hyconv in self.hyconvs:
            x = hyconv(x, H)
            x = F.leaky_relu(x, inplace=True)
            x = F.dropout(x, self.dropout)
        x = self.last_hyconv(x, H)
        return F.log_softmax(x, dim=1)

def main(learning_rate=0.001, epoch=100):
    '''

    :param learning_rate: default=0.001
    :param epoch: default=100
        loss: 0.2389
        acc: 0.9170
    :return: null
    '''

    import torch
    import torch.nn.functional as F
    from torch.optim import lr_scheduler
    import numpy as np
    import os, sys

    install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(install_path)

    from models.hyedge import neighbor_distance
    from models.hygraph import hyedge_concat
    # from models import HGEnTrans
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

    def H2idx(H):
        node_len, edge_len = H.size()
        node_indices = []
        edge_indices = []
        for i in range(node_len):
            for j in range(edge_len):
                if H[i, j]:
                    node_indices.append(i)
                    edge_indices.append(j)
        return torch.LongTensor([node_indices, edge_indices])

    d = '/users6/kyzhang/tk/bert/HG_data/attraction'
    H_all = np.load(os.path.join(d, 'H.npy'))
    en_ba_dim = np.load(os.path.join(d, 'en_ba_sen_nums_metadata.npy'))
    X_1d = np.load(os.path.join(d, 'X_1d.npy'))
    n_class = H_all.shape[1]
    target = generate_target(en_ba_dim)
    print(en_ba_dim)

    en_cum_num, ba_cum_num = np.cumsum(en_ba_dim, axis=1)
    print(en_cum_num, ba_cum_num)
    en_nums, ba_nums = np.sum(en_ba_dim, axis=1)
    i = 9
    en_c_num = en_cum_num[i]
    ba_c_num = ba_cum_num[i]

    X_c = np.row_stack((X_1d[: en_c_num], X_1d[en_nums: en_nums + ba_c_num]))
    H_c = np.row_stack((H_all[: en_c_num], H_all[en_nums: en_nums + ba_c_num]))
    target_c = np.concatenate((target[: en_c_num], target[en_nums: en_nums + ba_c_num]))

    X_c, H_c, target_c = torch.FloatTensor(X_c), torch.from_numpy(H_c), torch.from_numpy(target_c)
    ft, H, target_c = X_c.to(device), H2idx(H_c.to(device)), target_c.to(device)

    # construct model
    n_class_c = i + 1
    in_ft = ft.size(1)
    model = HGEnTrans(in_ft, n_class_c, hiddens=(8,))
    model = model.to(device)
    # model = HGEnTrans()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

    def train():
        model.train()
        optimizer.zero_grad()
        pred = model(ft, H)
        loss = F.nll_loss(pred, target_c)
        print('loss:', loss)
        loss.backward()
        optimizer.step()

    def val():
        model.eval()
        pred = model(ft, H)

        _train_acc = trans_class_acc(pred, target_c)
        # _val_acc = trans_class_acc(pred, target, mask_val)

        return _train_acc  # , _val_acc

    best_acc, best_iou = 0.0, 0.0
    for epoch in range(1, 100):
        train()
        # train_acc, val_acc = val()
        train_acc = val()
        # scheduler.step()
        print('learning rate:', optimizer.state_dict()['param_groups'][0]['lr'])
        # if val_acc > best_acc:
        #     best_acc = val_acc
        print(f'Epoch: {epoch}, Train:{train_acc:.4f}')


if __name__ == '__main__':
    main()

