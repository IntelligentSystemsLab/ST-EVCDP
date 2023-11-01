import torch
import torch.nn as nn
import torch.nn.functional as F
import functions as fn
import copy

use_cuda = True
device = torch.device("cuda:0" if use_cuda and torch.cuda.is_available() else "cpu")
fn.set_seed(seed=2023, flag=True)


class MultiHeadsGATLayer(nn.Module):
    def __init__(self, a_sparse, input_dim, out_dim, head_n, dropout, alpha):  # input_dim = seq_length
        super(MultiHeadsGATLayer, self).__init__()

        self.head_n = head_n
        self.heads_dict = dict()
        for n in range(head_n):
            self.heads_dict[n, 0] = nn.Parameter(torch.zeros(size=(input_dim, out_dim), device=device))
            self.heads_dict[n, 1] = nn.Parameter(torch.zeros(size=(1, 2 * out_dim), device=device))
            nn.init.xavier_normal_(self.heads_dict[n, 0], gain=1.414)
            nn.init.xavier_normal_(self.heads_dict[n, 1], gain=1.414)
        self.linear = nn.Linear(head_n, 1, device=device)

        # regularization
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=0)

        # sparse matrix
        self.a_sparse = a_sparse
        self.edges = a_sparse.indices()
        self.values = a_sparse.values()
        self.N = a_sparse.shape[0]
        a_dense = a_sparse.to_dense()
        a_dense[torch.where(a_dense == 0)] = -1000000000
        a_dense[torch.where(a_dense == 1)] = 0
        self.mask = a_dense

    def forward(self, x):
        b, n, s = x.shape
        x = x.reshape(b*n, s)

        atts_stack = []
        # multi-heads attention
        for n in range(self.head_n):
            h = torch.matmul(x, self.heads_dict[n, 0])
            edge_h = torch.cat((h[self.edges[0, :], :], h[self.edges[1, :], :]), dim=1).t()  # [Ni, Nj]
            atts = self.heads_dict[n, 1].mm(edge_h).squeeze()
            atts = self.leakyrelu(atts)
            atts_stack.append(atts)

        mt_atts = torch.stack(atts_stack, dim=1)
        mt_atts = self.linear(mt_atts)
        new_values = self.values * mt_atts.squeeze()
        atts_mat = torch.sparse_coo_tensor(self.edges, new_values)
        atts_mat = atts_mat.to_dense() + self.mask
        atts_mat = self.softmax(atts_mat)
        return atts_mat


class MLP(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(in_features=in_channel, out_features=256)
        self.l2 = nn.Linear(in_features=256, out_features=256)
        self.l3 = nn.Linear(in_features=256, out_features=out_channel)
        # self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)
        return x


class PAG(nn.Module):
    def __init__(self, a_sparse, seq=12, kcnn=2, k=6, m=2):
        super(PAG, self).__init__()
        self.feature = seq
        self.seq = seq-kcnn+1
        self.alpha = 0.5
        self.m = m
        self.a_sparse = a_sparse
        self.nodes = a_sparse.shape[0]

        # GAT
        self.conv2d = nn.Conv2d(1, 1, (kcnn, 2))  # input.shape = [batch, channel, width, height]
        self.gat_lyr = MultiHeadsGATLayer(a_sparse, self.seq, self.seq, 4, 0, 0.2)
        self.gcn = nn.Linear(in_features=self.seq, out_features=self.seq)

        # TPA
        self.lstm = nn.LSTM(m, m, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(in_features=self.seq - 1, out_features=k)
        self.fc2 = nn.Linear(in_features=k, out_features=m)
        self.fc3 = nn.Linear(in_features=k + m, out_features=1)
        self.decoder = nn.Linear(self.seq, 1)

        # Activation
        self.dropout = nn.Dropout(p=0.5)
        self.LeakyReLU = nn.LeakyReLU()

        #
        adj1 = copy.deepcopy(self.a_sparse.to_dense())
        adj2 = copy.deepcopy(self.a_sparse.to_dense())
        for i in range(self.nodes):
            adj1[i, i] = 0.000000001
            adj2[i, i] = 0
        degree = 1.0 / (torch.sum(adj1, dim=0))
        degree_matrix = torch.zeros((self.nodes, self.feature), device=device)
        for i in range(12):
            degree_matrix[:, i] = degree
        self.degree_matrix = degree_matrix
        self.adj2 = adj2

    def forward(self, occ, prc):  # occ.shape = [batch,node, seq]
        b, n, s = occ.shape
        data = torch.stack([occ, prc], dim=3).reshape(b*n, s, -1).unsqueeze(1)
        data = self.conv2d(data)
        data = data.squeeze().reshape(b, n, -1)

        # first layer
        atts_mat = self.gat_lyr(data)  # attention matrix, dense(nodes, nodes)
        occ_conv1 = torch.matmul(atts_mat, data)  # (b, n, s)
        occ_conv1 = self.dropout(self.LeakyReLU(self.gcn(occ_conv1)))

        # second layer
        atts_mat2 = self.gat_lyr(occ_conv1)  # attention matrix, dense(nodes, nodes)
        occ_conv2 = torch.matmul(atts_mat2, occ_conv1)  # (b, n, s)
        occ_conv2 = self.dropout(self.LeakyReLU(self.gcn(occ_conv2)))

        occ_conv1 = (1 - self.alpha) * occ_conv1 + self.alpha * data
        occ_conv2 = (1 - self.alpha) * occ_conv2 + self.alpha * occ_conv1
        occ_conv1 = occ_conv1.view(b * n, self.seq)
        occ_conv2 = occ_conv2.view(b * n, self.seq)

        x = torch.stack([occ_conv1, occ_conv2], dim=2)  # best
        lstm_out, (_, _) = self.lstm(x)  # b*n, s, 2

        # TPA
        ht = lstm_out[:, -1, :]  # ht
        hw = lstm_out[:, :-1, :]  # from h(t-1) to h1
        hw = torch.transpose(hw, 1, 2)
        Hc = self.fc1(hw)
        Hn = self.fc2(Hc)
        ht = torch.unsqueeze(ht, dim=2)
        a = torch.bmm(Hn, ht)
        a = torch.sigmoid(a)
        a = torch.transpose(a, 1, 2)
        vt = torch.matmul(a, Hc)
        ht = torch.transpose(ht, 1, 2)
        hx = torch.cat((vt, ht), dim=2)
        y = self.fc3(hx)
        y = y.view(b, n)
        return y
