import torch
from torch import nn, optim
from d2l_tool import Timer, Accumulator, Animator


class RNNModel(nn.Module):
    def __init__(self, rnn_layer, num_layer, input_size, num_hiddens, output_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        # # 定义RNN层
        # RNN输入的形状为（num_steps, batch_size, input_size）  # input_size 就是 vocab_size
        # RNN输出的形状为（num_steps, batch_size, num_hiddens）和（1, batch_size, num_hiddens）
        self.rnn = rnn_layer(input_size, num_hiddens, num_layer)
        self.input_size = self.rnn.input_size
        self.num_hiddens = self.rnn.hidden_size

        self.output_size = output_size
        self.linear = nn.Linear(self.num_hiddens, self.output_size)

    def forward(self, inputs, state):
        # inputs的形状为（num_steps, num_hiddens, input_size）
        # Y是所有时间步的隐藏状态，state是最后一个时间步的隐藏状态
        # Y的形状为（num_steps, batch_size, num_hiddens），state为（1，batch_size, num_hiddens）
        Y, state = self.rnn(inputs, state)
        # 全连接层首先将Y的形状改为(num_steps*batch_size, num_hiddens)
        # 它的输出形状是(num_steps*batch_size, output_size)。
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # nn.GRU以张量作为隐状态
            return torch.zeros((self.rnn.num_layers, batch_size, self.num_hiddens),
                               device=device)
        else:
            # nn.LSTM以元组作为隐状态
            return (torch.zeros((self.rnn.num_layers, batch_size, self.num_hiddens),
                                device=device),
                    torch.zeros((self.rnn.num_layers, batch_size, self.num_hiddens),
                                device=device))


def train_epoch(net, train_iter, loss, updater, device, use_random_iter):
    net.train()
    state, timer = None, Timer()
    metric = Accumulator(2)
    for X, Y in train_iter:
        if state is None or use_random_iter:
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                state.detach_()
            else:
                for s in state:
                    s.detach_()

        X = X.transpose(0, 1);
        y = Y.transpose(0, 1).reshape(-1, Y.shape[-1])
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            updater.step()
        else:
            pass
        metric.add(1 * y.shape[0], y.shape[0])
    return metric[0] / metric[1], metric[1] / timer.stop()


# 实际上是计算一个epoch中的准确率
def evaluate_accuracy_epoch(net, test_iter, loss, device, use_random_iter):
    """计算在指定数据集上模型的精度"""
    net.eval()  # 将模型设置为评估模式
    state, timer = None, Timer()
    metric = Accumulator(2)  # 训练损失之和, num_steps*batch_size

    # 执行测试
    with torch.no_grad():  # 不计算梯度，只前向传播
        # 循环遍历每个batch
        for X, Y in test_iter:
            # 这里X的形状为（batch_size, num_steps, input_size）
            # 这里Y的形状为（batch_size, num_steps, output_size）
            if state is None or use_random_iter:
                # 在第一次迭代或使用随机抽样时初始化state
                state = net.begin_state(batch_size=X.shape[0], device=device)
            else:
                if isinstance(net, nn.Module) and not isinstance(state, tuple):
                    # state对于nn.GRU是个张量
                    state.detach_()
                else:
                    # state对于nn.LSTM或对于我们从零开始实现的模型是个张量
                    for s in state:
                        s.detach_()
            X = X.transpose(0, 1)  # 这里X的形状为（num_steps, batch_size, input_size）
            y = Y.transpose(0, 1).reshape(-1, Y.shape[-1])  # 这里Y的形状为（num_steps * batch_size, output_size）
            X, y = X.to(device), y.to(device)
            y_hat, state = net(X, state)  # 这里y_hat的形状为（num_steps * batch_size, output_size）
            l = loss(y_hat, y)  # 这里y_hat的形状为（num_steps * batch_size, output_size）
            metric.add(l * y.shape[0], y.shape[0])
    return metric[0] / metric[1]


def train(net, train_iter, test_iter, lr, num_epochs, device, use_random_iter=False):
    """训练模型"""
    loss = nn.MSELoss(reduction='mean')  # 'none'返回对应的tensor，'sum'求和、'mean'求平均
    animator = Animator(xlabel='epoch', ylabel='loss',
                        legend=['train', 'test'], xlim=[0, num_epochs])
    # 优化器
    # updater = torch.optim.SGD(net.parameters(), lr)
    updater = torch.optim.Adam(net.parameters(), lr)

    # 训练和预测
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}\n-------------------------------")
        train_loss, speed = train_epoch(net, train_iter, loss, updater, device, use_random_iter)
        test_loss = evaluate_accuracy_epoch(net, test_iter, loss, device, use_random_iter)
        # if (epoch + 1) % 10 == 0:
        animator.add(epoch + 1, [train_loss, test_loss])
        print(f'{speed}\ttokens/sec {str(device)}\ntrain_loss \t{train_loss}\ntest_loss\t{test_loss}')

# # 梯度裁剪
def grad_clipping(net, theta):
    """裁剪梯度"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm