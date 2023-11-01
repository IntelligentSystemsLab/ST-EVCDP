import copy
import baselines
import torch
import pandas as pd
import functions as fn
from torch.utils.data import DataLoader
from tqdm import tqdm
import models


# system configuration
use_cuda = True
device = torch.device("cuda:0" if use_cuda and torch.cuda.is_available() else "cpu")
fn.set_seed(seed=2023, flag=True)

# hyper params
model_name = 'PAG'
seq_l = 12
pre_l = 6
bs = 512
p_epoch = 400
n_epoch = 2000
is_train = True
is_pre_train = False

# input data
occ, prc, adj, col, dis, cap, time, inf = fn.read_dataset()
adj_dense = torch.Tensor(adj)
adj_dense_cuda = adj_dense.to(device)
adj_sparse = adj_dense.to_sparse_coo().to(device)

# dataset division
train_occupancy, valid_occupancy, test_occupancy = fn.division(occ, train_rate=0.6, valid_rate=0.2, test_rate=0.2)
train_price, valid_price, test_price = fn.division(prc, train_rate=0.6, valid_rate=0.2, test_rate=0.2)

# data
train_dataset = fn.CreateDataset(train_occupancy, train_price, seq_l, pre_l, device, adj_dense)
train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, drop_last=True)
valid_dataset = fn.CreateDataset(valid_occupancy, valid_price, seq_l, pre_l, device, adj_dense)
valid_loader = DataLoader(valid_dataset, batch_size=len(valid_occupancy), shuffle=False)
test_dataset = fn.CreateDataset(test_occupancy, test_price, seq_l, pre_l, device, adj_dense)
test_loader = DataLoader(test_dataset, batch_size=len(test_occupancy), shuffle=False)

# training setting
model = models.PAG(a_sparse=adj_sparse).to(device)
# model = baselines.LstmGcn(seq_l, 2, adj_dense_cuda)
# model = baselines.LstmGat(seq_l, 2, adj_dense_cuda, adj_sparse)
optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.00001)
loss_function = torch.nn.MSELoss()
valid_loss = 100

if is_train is True:
    model.train()
    if is_pre_train is True:
        for epoch in tqdm(range(p_epoch), desc='Pre-training'):
            for j, data in enumerate(train_loader):
                '''
                occupancy = (batch, seq, node)
                price = (batch, seq, node)
                label = (batch, node)
                '''
                occupancy, price, label, prc_ch, label_ch = data
                optimizer.zero_grad()
                predict = model(occupancy, prc_ch)
                loss = loss_function(predict, label_ch)
                loss.backward()
                optimizer.step()

    for epoch in tqdm(range(n_epoch), desc='Fine-tuning'):
        for j, data in enumerate(train_loader):
            '''
            occupancy = (batch, seq, node)
            price = (batch, seq, node)
            label = (batch, node)
            '''
            model.train()
            occupancy, price, label, prc_ch, label_ch = data

            optimizer.zero_grad()
            predict = model(occupancy, price)
            loss = loss_function(predict, label)
            loss.backward()
            optimizer.step()

        # validation
        model.eval()
        for j, data in enumerate(valid_loader):
            '''
            occupancy = (batch, seq, node)
            price = (batch, seq, node)
            label = (batch, node)
            '''
            model.train()
            occupancy, price, label, prc_ch, label_ch = data
            predict = model(occupancy, price)
            loss = loss_function(predict, label)
            if loss.item() < valid_loss:
                valid_loss = loss.item()
                torch.save(model, './checkpoints' + '/' + model_name + '_' + str(pre_l) + '_bs' + str(bs) + 'model.pt')

model = torch.load('./checkpoints' + '/' + model_name + '_' + str(pre_l) + '_bs' + str(bs) + 'model.pt')
# test
model.eval()
result_list = []
node = 32
for j, data in enumerate(test_loader):
    if j == 0:
        occupancy, price, label, prc_ch, label_ch = data  # occupancy.shape = [batch, seq, node]
        print('occupancy:', occupancy.shape, 'price:', price.shape, 'label:', label.shape)
        with torch.no_grad():
            predict = model(occupancy, price)
            predict = predict.cpu().detach().numpy()
            label = label.cpu().detach().numpy()

        print('Evaluation results')
        output_no_noise = fn.metrics(test_pre=predict, test_real=label)
        result_list.append(output_no_noise)
        result_df = pd.DataFrame(columns=['MSE', 'RMSE', 'MAPE', 'RAE', 'MAE', 'R2'], data=result_list)
        result_df.to_csv('./results' + '/' + model_name + '_' + str(pre_l) + 'bs' + str(bs) + '.csv', encoding='gbk')

    else:
        break


