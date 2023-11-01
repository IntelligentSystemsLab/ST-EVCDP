import torch
import torch.nn as nn
import checkpoints
import torch.nn.functional as F
import functions as fn
import copy

use_cuda = True
device = torch.device("cuda:0" if use_cuda and torch.cuda.is_available() else "cpu")
fn.set_seed(seed=2023, flag=True)


class VAR(nn.Module):
    def __init__(self, node=247, seq=12, feature=2):  # input_dim = seq_length
        super(VAR, self).__init__()
        self.linear = nn.Linear(node*seq*feature, node)

    def forward(self, occ, prc):
        x = torch.cat((occ, prc), dim=2)
        x = torch.flatten(x, 1, 2)
        x = self.linear(x)
        return x


class LSTM(nn.Module):
    def __init__(self, seq, n_fea):
        super(LSTM, self).__init__()
        self.encoder = nn.Conv2d(self.nodes, self.nodes, (n_fea, n_fea))  # input.shape: [batch, channel, width, height]
        self.lstm = nn.LSTM(self.nodes, self.nodes, num_layers=2, batch_first=True)
        self.decoder = nn.Linear(seq-n_fea+1, 1)

    def forward(self, occ, prc):  # occ.shape = [batch, node, seq]
        x = torch.stack([occ, prc], dim=3)
        x = self.encoder(x)
        x = torch.transpose(x.squeeze(), 1, 2)  # shape [batch, seq-n_fea+1, node]
        x, _ = self.lstm(x)
        x = torch.transpose(x, 1, 2)  # shape [batch, node, seq-n_fea+1]
        x = self.decoder(x)
        return x


class GCN(nn.Module):
    def __init__(self, seq, n_fea, adj_dense):
        super(GCN, self).__init__()
        self.encoder = nn.Conv2d(self.nodes, self.nodes, (n_fea, n_fea))
        self.gcn_l1 = nn.Linear(seq-n_fea+1, seq-n_fea+1)
        self.gcn_l2 = nn.Linear(seq-n_fea+1, seq-n_fea+1)
        self.A = adj_dense
        self.act = nn.ReLU()
        self.decoder = nn.Linear(seq-n_fea+1, 1)

        # calculate A_delta matrix
        deg = torch.sum(adj_dense, dim=0)
        deg = torch.diag(deg)
        deg_delta = torch.linalg.inv(torch.sqrt(deg))
        a_delta = torch.matmul(torch.matmul(deg_delta, adj_dense), deg_delta)
        self.A = a_delta

    def forward(self, occ, prc):  # occ.shape = [batch, node, seq]
        x = torch.stack([occ, prc], dim=3)
        x = self.encoder(x)
        #  l1
        x = self.gcn_l1(x)
        x = torch.matmul(self.A, x)
        x = self.act(x)
        #  l2
        x = self.gcn_l2(x)
        x = torch.matmul(self.A, x)
        x = self.act(x)
        x = self.decoder(x)
        return x


class LstmGcn(nn.Module):
    def __init__(self, seq, n_fea, adj_dense):
        super(LstmGcn, self).__init__()
        self.A = adj_dense
        self.nodes = adj_dense.shape[0]
        self.encoder = nn.Conv2d(self.nodes, self.nodes, (n_fea, n_fea), device=device)
        self.gcn_l1 = nn.Linear(seq - n_fea + 1, seq - n_fea + 1, device=device)
        self.gcn_l2 = nn.Linear(seq - n_fea + 1, seq - n_fea + 1, device=device)

        self.act = nn.ReLU()
        self.decoder = nn.Linear(seq - n_fea + 1, 1, device=device)

        # calculate A_delta matrix
        deg = torch.sum(adj_dense, dim=0)
        deg = torch.diag(deg)
        deg_delta = torch.linalg.inv(torch.sqrt(deg))
        a_delta = torch.matmul(torch.matmul(deg_delta, adj_dense), deg_delta)
        self.A = a_delta

    def forward(self, occ, prc):  # occ.shape = [batch, node, seq]
        x = torch.stack([occ, prc], dim=3)
        x = self.encoder(x)
        x = torch.squeeze(x)
        #  l1
        x = self.gcn_l1(x)
        x = torch.matmul(self.A, x)
        x = self.act(x)
        #  l2
        x = self.gcn_l2(x)
        x = torch.matmul(self.A, x)
        x = self.act(x)
        x = self.decoder(x)
        x = torch.squeeze(x)
        return x


class LstmGat(nn.Module):
    def __init__(self, seq, n_fea, adj_dense, adj_sparse):
        super(LstmGat, self).__init__()
        self.A = adj_dense
        self.nodes = adj_dense.shape[0]
        self.gcn = nn.Linear(in_features=seq - n_fea + 1, out_features=seq - n_fea + 1, device=device)
        self.encoder = nn.Conv2d(self.nodes, self.nodes, (n_fea, n_fea), device=device)
        self.gat_l1 = checkpoints.MultiHeadsGATLayer(adj_sparse, seq - n_fea + 1, seq - n_fea + 1, 4, 0, 0.2)
        self.gat_l2 = checkpoints.MultiHeadsGATLayer(adj_sparse, seq - n_fea + 1, seq - n_fea + 1, 4, 0, 0.2)
        self.decoder = nn.Linear(seq - n_fea + 1, 1, device=device)

        # Activation
        self.dropout = nn.Dropout(p=0.5)
        self.LeakyReLU = nn.LeakyReLU()

    def forward(self, occ, prc):  # occ.shape = [batch, node, seq]
        x = torch.stack([occ, prc], dim=3)
        x = self.encoder(x)
        x = torch.squeeze(x)

        # first layer
        atts_mat = self.gat_l1(x)  # attention matrix, dense(nodes, nodes)
        occ_conv1 = torch.matmul(atts_mat, x)  # (b, n, s)
        occ_conv1 = self.dropout(self.LeakyReLU(self.gcn(occ_conv1)))

        # second layer
        atts_mat2 = self.gat_l2(occ_conv1)  # attention matrix, dense(nodes, nodes)
        occ_conv2 = torch.matmul(atts_mat2, occ_conv1)  # (b, n, s)
        occ_conv2 = self.dropout(self.LeakyReLU(self.gcn(occ_conv2)))

        x = self.decoder(occ_conv2)
        x = torch.squeeze(x)
        return x


class TPA(nn.Module):
    def __init__(self, seq, n_fea):
        super(TPA, self).__init__()
        self.encoder = nn.Conv2d(self.nodes, self.nodes, (n_fea, n_fea), device=device)
        # TPA
        self.lstm = nn.LSTM(2, 2, num_layers=2, batch_first=True, device=device)
        self.fc1 = nn.Linear(in_features=self.seq - 1, out_features=2, device=device)
        self.fc2 = nn.Linear(in_features=2, out_features=2, device=device)
        self.fc3 = nn.Linear(in_features=2 + 2, out_features=1, device=device)
        self.decoder = nn.Linear(self.seq, 1, device=device)

    def forward(self, occ, prc):  # occ.shape = [batch, node, seq]
        x = torch.stack([occ, prc], dim=3)
        x = self.encoder(x)
        x = torch.squeeze(x)

        # TPA
        lstm_out, (_, _) = self.lstm(x)  # b*n, s, 2
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
        print(y.shape)
        return y


# Other baselines refer to its own original code.
