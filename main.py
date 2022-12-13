import torch
from torch import nn
from RNNModel import RNNModel, train
from matplotlib import pyplot as plt
from data import dataset_process, dataset_encode_scale, split_train_test, load_data, inverse_transform_col
from predict import pre_seq_data_iter, predict

print(f'-------------------------------\n'
      f'Reading and processing data\n'
      f'-------------------------------')
# # 原生数据存放路径
csv_filepath = 'data/test.csv'
# # 处理后的数据存放目标路径
csv_processed_filepath = 'data/test_processed.csv'
# # 处理数据集（将时间作为index，去掉前24h的数据，用0填充‘PM2.5浓度’中缺失的数据，保存处理后的数据至文件）
dataset = dataset_process(csv_filepath, csv_processed_filepath)
# # 编码和归一化后的数据存放目标路径
csv_scaled_filepath = 'test_scaled.csv'
# # 编码和归一化
dataset_scaled, scaler = dataset_encode_scale(dataset, csv_scaled_filepath)

print(f'-------------------------------\n'
      f'Splitting dataset\n'
      f'-------------------------------')
# # 按比例划分训练集与验证集
test_ratio = 0.2
train_dataset, test_dataset = split_train_test(dataset_scaled, test_ratio)
print(train_dataset)
print(test_dataset)
# train_dataset.to_csv('train_dataset.csv')
# test_dataset.to_csv('test_dataset.csv')

num_layer = 2
batch_size = 72
num_steps = 24
num_hiddens = 50
num_epochs = 100
learning_rate = 0.0001
output_size = 1
input_size = len(train_dataset.columns) - output_size
use_random_iter = True

# 每调用一次返回：X的形状为（batch_size, num_steps, input_size），Y的形状为（batch_size, num_steps, output_size）
train_iter = load_data(train_dataset, batch_size, num_steps, output_size, use_random_iter=use_random_iter)
test_iter = load_data(test_dataset, batch_size, num_steps, output_size, use_random_iter=use_random_iter)

# # 检查torch.cuda是否可用，否则继续使用CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'-------------------------------\n'
      f'Using {device} device\n'
      f'-------------------------------')

RNN_net = RNNModel(nn.RNN, num_layer, input_size, num_hiddens, output_size)
LSTM_net = RNNModel(nn.LSTM, num_layer, input_size, num_hiddens, output_size)
GRU_net = RNNModel(nn.GRU, num_layer, input_size, num_hiddens, output_size)

net = RNN_net.to(device)

train(net, train_iter, test_iter, learning_rate, num_epochs, device, use_random_iter=use_random_iter)
plt.show()

print(f'-------------------------------\n'
      f'Predicting\n'
      f'-------------------------------')
predict_iter = pre_seq_data_iter(test_dataset[500:1000], pre_num_steps=num_steps, output_size=output_size)
Ys, Y_hats, predict_loss = predict(net, predict_iter, loss=nn.MSELoss(reduction='mean'), device=device)
print('Predict_loss:', predict_loss.double())
Ys_t = inverse_transform_col(scaler, Ys, 0)
Y_hats_t = inverse_transform_col(scaler, Y_hats, 0)
plt.plot(Ys, color='tab:blue', label='true')
plt.plot(Y_hats, color='tab:orange', label='predicted')
plt.legend()
plt.grid()
plt.show()


print(f'-------------------------------\n'
      f'Done\n'
      f'-------------------------------')