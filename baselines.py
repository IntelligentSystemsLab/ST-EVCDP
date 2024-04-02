import torch
import torch.nn as nn
import models
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
    def __init__(self, seq, n_fea, node=247):
        super(LSTM, self).__init__()
        self.nodes = node
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
        x = torch.squeeze(x)
        return x


class GCN(nn.Module):
    def __init__(self, seq, n_fea, adj_dense):
        super(GCN, self).__init__()
        self.nodes = adj_dense.shape[0]
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
        self.lstm = nn.LSTM(self.nodes, self.nodes, num_layers=2, batch_first=True)
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
        # lstm
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = x.transpose(1, 2)
        x = self.decoder(x)
        x = torch.squeeze(x)
        return x


class LstmGat(nn.Module):
    def __init__(self, seq, n_fea, adj_dense, adj_sparse):
        super(LstmGat, self).__init__()
        self.nodes = adj_dense.shape[0]
        self.gcn = nn.Linear(in_features=seq - n_fea + 1, out_features=seq - n_fea + 1, device=device)
        self.encoder = nn.Conv2d(self.nodes, self.nodes, (n_fea, n_fea), device=device)
        self.gat_l1 = models.MultiHeadsGATLayer(adj_sparse, seq - n_fea + 1, seq - n_fea + 1, 4, 0, 0.2)
        self.gat_l2 = models.MultiHeadsGATLayer(adj_sparse, seq - n_fea + 1, seq - n_fea + 1, 4, 0, 0.2)
        self.lstm = nn.LSTM(self.nodes, self.nodes, num_layers=2, batch_first=True)
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

        # lstm
        x = occ_conv2.transpose(1, 2)
        x, _ = self.lstm(x)
        x = x.transpose(x, 1, 2)

        # decode
        x = self.decoder(x)
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


# https://doi.org/10.1016/j.trc.2023.104205
class HSTGCN(nn.Module):
    def __init__(self, seq, n_fea, adj_distance, adj_demand, alpha=0.5):
        super(HSTGCN, self).__init__()
        # hyper-params
        self.nodes = adj_distance.shape[0]
        self.alpha = alpha
        hidden = seq - n_fea + 1

        # network components
        self.encoder = nn.Conv2d(self.nodes, self.nodes, (n_fea, n_fea))
        self.linear = nn.Linear(hidden, hidden)
        self.distance_gcn_l1 = nn.Linear(hidden, hidden)
        self.distance_gcn_l2 = nn.Linear(hidden, hidden)
        self.gru1 = nn.GRU(self.nodes, self.nodes, num_layers=2, batch_first=True)
        self.demand_gcn_l1 = nn.Linear(hidden, hidden)
        self.demand_gcn_l2 = nn.Linear(hidden, hidden)
        self.gru2 = nn.GRU(self.nodes, self.nodes, num_layers=2, batch_first=True)
        self.decoder = nn.Sequential(nn.Linear(hidden, 16),
        nn.ReLU(),
        nn.Linear(16, 1)
        )
        
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

        # calculate A_delta matrix
        deg = torch.sum(adj_distance, dim=0)
        deg = torch.diag(deg)
        deg_delta = torch.linalg.inv(torch.sqrt(deg))
        a_delta = torch.matmul(torch.matmul(deg_delta, adj_distance), deg_delta)
        self.A_dis = a_delta

        deg = torch.sum(adj_demand, dim=0)
        deg = torch.diag(deg)
        deg_delta = torch.linalg.inv(torch.sqrt(deg))
        a_delta = torch.matmul(torch.matmul(deg_delta, adj_demand), deg_delta)
        self.A_dem = a_delta

    def forward(self, occ, prc):  # occ.shape = [batch, node, seq]
        x = torch.stack([occ, prc], dim=3)
        x = self.encoder(x)
        x = torch.squeeze(x)
        x = self.act(self.linear(x))

        # distance-based graph propagation
        #  l1
        x1 = self.distance_gcn_l1(x)
        x1 = torch.matmul(self.A_dis, x1)
        x1 = self.dropout(self.act(x1))
        #  l2
        x1 = self.distance_gcn_l2(x1)
        x1 = torch.matmul(self.A_dis, x1)
        x1 = self.dropout(self.act(x1))
        # gru
        x1 = x1.transpose(1, 2)
        x1, _ = self.gru1(x1)
        x1 = x1.transpose(1, 2)

        # demand-based graph propagation
        #  l1
        x2 = self.demand_gcn_l1(x)
        x2 = torch.matmul(self.A_dem, x2)
        x2 = self.dropout(self.act(x2))
        #  l2
        x2 = self.demand_gcn_l2(x2)
        x2 = torch.matmul(self.A_dem, x2)
        x2 = self.dropout(self.act(x2))
        # gru
        x2 = x2.transpose(1, 2)
        x2, _ = self.gru2(x2)
        x2 = x2.transpose(1, 2)

        # decode
        output = self.alpha * x1 + (1-self.alpha) * x2
        output = self.decoder(output)
        output = torch.squeeze(output)
        return output


# https://arxiv.org/abs/2311.06190
class FGN(nn.Module):
    def __init__(self, pre_length=1, embed_size=64,
                 feature_size=0, seq_length=12, hidden_size=32, hard_thresholding_fraction=1, hidden_size_factor=1, sparsity_threshold=0.01):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.number_frequency = 1
        self.pre_length = pre_length
        self.feature_size = feature_size
        self.seq_length = seq_length
        self.frequency_size = self.embed_size // self.number_frequency
        self.hidden_size_factor = hidden_size_factor
        self.sparsity_threshold = sparsity_threshold
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.scale = 0.02
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))

        self.encoder = nn.Linear(2, 1)
        self.w1 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size, self.frequency_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor, self.frequency_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.frequency_size))
        self.w3 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size,
                                     self.frequency_size * self.hidden_size_factor))
        self.b3 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor))
        self.embeddings_10 = nn.Parameter(torch.randn(self.seq_length, 8))
        self.fc = nn.Sequential(
            nn.Linear(self.embed_size * 8, 64),
            nn.LeakyReLU(),
            nn.Linear(64, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pre_length)
        )
        self.to('cuda:0')

    def tokenEmb(self, x):
        x = x.unsqueeze(2)
        y = self.embeddings
        return x * y

    # FourierGNN
    def fourierGC(self, x, B, N, L):
        o1_real = torch.zeros([B, (N*L)//2 + 1, self.frequency_size * self.hidden_size_factor],
                              device=x.device)
        o1_imag = torch.zeros([B, (N*L)//2 + 1, self.frequency_size * self.hidden_size_factor],
                              device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        o3_real = torch.zeros(x.shape, device=x.device)
        o3_imag = torch.zeros(x.shape, device=x.device)

        o1_real = F.relu(
            torch.einsum('bli,ii->bli', x.real, self.w1[0]) - \
            torch.einsum('bli,ii->bli', x.imag, self.w1[1]) + \
            self.b1[0]
        )

        o1_imag = F.relu(
            torch.einsum('bli,ii->bli', x.imag, self.w1[0]) + \
            torch.einsum('bli,ii->bli', x.real, self.w1[1]) + \
            self.b1[1]
        )

        # 1 layer
        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)

        o2_real = F.relu(
            torch.einsum('bli,ii->bli', o1_real, self.w2[0]) - \
            torch.einsum('bli,ii->bli', o1_imag, self.w2[1]) + \
            self.b2[0]
        )

        o2_imag = F.relu(
            torch.einsum('bli,ii->bli', o1_imag, self.w2[0]) + \
            torch.einsum('bli,ii->bli', o1_real, self.w2[1]) + \
            self.b2[1]
        )

        # 2 layer
        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = x + y

        o3_real = F.relu(
                torch.einsum('bli,ii->bli', o2_real, self.w3[0]) - \
                torch.einsum('bli,ii->bli', o2_imag, self.w3[1]) + \
                self.b3[0]
        )

        o3_imag = F.relu(
                torch.einsum('bli,ii->bli', o2_imag, self.w3[0]) + \
                torch.einsum('bli,ii->bli', o2_real, self.w3[1]) + \
                self.b3[1]
        )

        # 3 layer
        z = torch.stack([o3_real, o3_imag], dim=-1)
        z = F.softshrink(z, lambd=self.sparsity_threshold)
        z = z + x
        z = torch.view_as_complex(z)
        return z

    def forward(self, occ, prc):
        x = torch.stack([occ, prc], dim=3)
        x = self.encoder(x)
        x = torch.squeeze(x)

        B, N, L = x.shape
        # B*N*L ==> B*NL
        x = x.reshape(B, -1)
        # embedding B*NL ==> B*NL*D
        x = self.tokenEmb(x)

        # FFT B*NL*D ==> B*NT/2*D
        x = torch.fft.rfft(x, dim=1, norm='ortho')

        x = x.reshape(B, (N*L)//2+1, self.frequency_size)

        bias = x

        # FourierGNN
        x = self.fourierGC(x, B, N, L)

        x = x + bias

        x = x.reshape(B, (N*L)//2+1, self.embed_size)

        # ifft
        x = torch.fft.irfft(x, n=N*L, dim=1, norm="ortho")

        x = x.reshape(B, N, L, self.embed_size)
        x = x.permute(0, 1, 3, 2)  # B, N, D, L

        # projection
        x = torch.matmul(x, self.embeddings_10)
        x = x.reshape(B, N, -1)
        x = self.fc(x)
        x = torch.squeeze(x)
        return x

# Other baselines refer to its own original code.
