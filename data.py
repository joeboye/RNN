# # 对数据集dataset非数值项编码，对所有项归一化处理，另存为csv_scaled_filepath，返回处理后的dataset_scaled
import random

import numpy as np
import torch
from sklearn import preprocessing
import os
import pandas as pd


def dataset_process(csv_filepath, csv_processed_filepath):
    """判断数据是否存在处理数据后生成的文件（是否处理过），如果没有则处理数据"""
    if not os.path.exists(csv_processed_filepath):
        # # 读取csv文件，整合日期数据后将日期放至第1列
        dataset = pd.read_csv(csv_filepath, sep=',', header=0, encoding="gbk")
        # 保存处理后的数据
        dataset.to_csv(csv_processed_filepath, encoding="utf_8_sig")

    # # 读取处理后的数据
    dataset = pd.read_csv(csv_processed_filepath,
                          sep=',', header=0, index_col=0, encoding="utf-8")
    return dataset


def dataset_encode_scale(dataset, csv_scaled_filepath):
    values = dataset.values  # 获取dataset中的数据,values的类型为numpy.ndarray

    # # 将所有数据按列映射至（0,1），实现归一化
    # scaler.inverse_transform(values_scaled)实现逆向映射回原值
    values = values.astype('float32')  # 将所有数据转换为浮点型
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    values_scaled = scaler.fit_transform(values)  # 类型为numpy.ndarray,一行是一个时间步，一列为一个维度，即列数为input_size

    # 将编码和归一化后的数据转换为pandas DataFrame，并写入文件
    dataset_scaled = pd.DataFrame(values_scaled, columns=dataset.columns, index=dataset.index)
    if not os.path.exists(csv_scaled_filepath):
        dataset_scaled.to_csv(csv_scaled_filepath)
    return dataset_scaled, scaler


# # 对归一化的某一列反归一化
def inverse_transform_col(scaler, col_scaled, n_col):
    '''scaler是对包含多个feature的X拟合的,
    col_scaled对应其中一个feature,n_col为col_scaled在X中对应的列编号.返回col_scaled的反归一化结果'''
    col_scaled = col_scaled.copy()
    col_scaled -= scaler.min_[n_col]
    col_scaled /= scaler.scale_[n_col]
    return col_scaled


def split_train_test(dataset_scaled, test_ratio):
    # # 将预测变量PM2.5浓度的真实值，新增至最后一列，构成训练、验证数据集
    train_teat_dataset = dataset_scaled.copy(deep=True)
    train_teat_dataset.drop(index=train_teat_dataset.index[-1], inplace=True)
    train_teat_dataset['真实值'] = dataset_scaled.values[1:dataset_scaled.shape[0], 3]
    # # 划分训练集与验证集
    train_teat_dataset_num = train_teat_dataset.shape[0]
    test_data_num = round(train_teat_dataset_num * test_ratio)
    test_dataset = train_teat_dataset[-test_data_num:train_teat_dataset_num]
    train_dataset = train_teat_dataset[0:-test_data_num]
    return train_dataset, test_dataset


# # 定义使用随机抽样生成一个小批量子序列的函数，小批量的形状：（批量大小，时间步数）
def seq_data_iter(dataset, batch_size, num_steps, output_size, use_random_iter=False):
    """使用随机抽样生成一个小批量子序列"""
    # 从随机偏移量开始对序列进行分区，随机范围包括num_steps-1
    dataset_values = dataset.values[random.randint(0, num_steps - 1):]  # 产生0~num_steps之间的随机整数，丢掉corpus中该数前面的一些元素
    dataset_features = dataset_values[:, 0:-1]
    dataset_labels = dataset_values[:, -1 * output_size:]
    # 减去1，是因为我们需要考虑label
    num_subseqs = (len(dataset_values) - 1) // num_steps  # 计算产生的子序列的个数
    # 长度为num_steps的子序列的起始索引
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))

    if use_random_iter:
        # 在随机抽样的迭代过程中，来自两个相邻的、随机的、小批量中的子序列不一定在原始序列上相邻
        random.shuffle(initial_indices)

    def data(feature_or_label, pos):
        # 返回从pos位置开始的长度为num_steps的序列
        return feature_or_label[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size  # 计算batch的个数
    for i in range(0, batch_size * num_batches, batch_size):
        # 在这里，initial_indices包含子序列的随机起始索引
        initial_indices_per_batch = initial_indices[i: i + batch_size]  # 拿出一个batch的子序列起始索引
        X = np.array([data(dataset_features, j) for j in initial_indices_per_batch])
        Y = np.array([data(dataset_labels, j) for j in initial_indices_per_batch])
        yield torch.tensor(X), torch.tensor(Y)  # 可迭代对象，每次一个batch


# # 定义加载序列数据的迭代器SeqDataLoader类
class SeqDataLoader:
    """加载序列数据的迭代器"""

    def __init__(self, dataset, batch_size, num_steps, output_size, use_random_iter):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.output_size = output_size
        self.use_random_iter = use_random_iter

    def __iter__(self):
        return seq_data_iter(self.dataset, self.batch_size, self.num_steps, self.output_size,
                             use_random_iter=self.use_random_iter)


# # 定义访问数据集，并产生可迭代对象的函数
def load_data(dataset, batch_size, num_steps, output_size, use_random_iter=False):
    """返回数据集的迭代器"""
    data_iter = SeqDataLoader(dataset, batch_size, num_steps, output_size, use_random_iter)
    return data_iter