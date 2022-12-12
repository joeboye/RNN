# # 对数据集dataset非数值项编码，对所有项归一化处理，另存为csv_scaled_filepath，返回处理后的dataset_scaled
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

print(f'-------------------------------\n'
      f'Splitting dataset\n'
      f'-------------------------------')
# # 按比例划分训练集与验证集
test_ratio = 0.2
train_dataset, test_dataset = split_train_test(dataset_scaled, test_ratio)
# print(train_dataset)
# print(test_dataset)
# train_dataset.to_csv('train_dataset.csv')
# test_dataset.to_csv('test_dataset.csv')