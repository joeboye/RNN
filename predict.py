# # 定义使用随机抽样生成一个小批量子序列的函数，小批量的形状：（批量大小，时间步数）
import numpy as np
import torch


def pre_seq_data_iter(dataset, pre_num_steps, output_size):
    """使用随机抽样生成一个小批量子序列"""
    # 从随机偏移量开始对序列进行分区，随机范围包括num_steps-1
    dataset_values = dataset.values
    dataset_features = dataset_values[:, 0:-1]
    dataset_labels = dataset_values[:, -1 * output_size:]

    num_batches = len(dataset_values) - pre_num_steps + 1  # batch的个数

    for i in range(0, num_batches):
        # 在这里，initial_indices包含子序列的随机起始索引
        X = np.array([dataset_features[i: i + pre_num_steps]])  # (batch_size=1, num_steps, input_size)
        Y = np.array([dataset_labels[i: i + pre_num_steps]])  # (batch_size=1, num_steps, output_size)
        yield torch.tensor(X), torch.tensor(Y)  # 可迭代对象，每次一个batch


def predict(net, predict_iter, loss, device):
    """预测"""
    net.eval()  # 将模型设置为评估模式
    Ys = []
    Y_hats = []
    # 执行测试
    with torch.no_grad():  # 不计算梯度，只前向传播
        # 循环遍历每个batch
        for X, Y in predict_iter:
            # 这里X的形状为（batch_size=1, num_steps, input_size）
            # 这里Y的形状为（batch_size=1, num_steps, output_size）
            state = net.begin_state(batch_size=X.shape[0], device=device)
            X = X.transpose(0, 1)  # 这里X的形状为（num_steps, batch_size, input_size）
            y = Y.transpose(0, 1).reshape(-1, Y.shape[-1])  # 这里Y的形状为（num_steps * batch_size, output_size）
            Ys.append(np.array(y[-1].cpu()).tolist())

            X, y = X.to(device), y.to(device)
            y_hat, state = net(X, state)  # 这里y_hat的形状为（num_steps * batch_size, output_size）
            Y_hats.append(np.array(y_hat[-1].cpu()).tolist())

        predict_loss = loss(torch.tensor(Y_hats), torch.tensor(Ys))  # 这里y_hat的形状为（num_steps * batch_size, output_size）
    return Ys, Y_hats, predict_loss