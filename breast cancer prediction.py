import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd

file = 'E:\实验室培训课件\第四课Pytorch\patients.csv'
data = pd.read_csv(file, engine='python')
data.head()

# 总数据集
data_numpy = np.array(data)
target_numpy = data_numpy[:, 6]

# 分配训练数据，约80%
train_data_tensor = torch.from_numpy(np.vstack((data_numpy[0:30, 0:6], data_numpy[38:56, 0:6]))).type(torch.FloatTensor)
train_target_tensor = torch.from_numpy(np.concatenate((target_numpy[0:30], target_numpy[38:56]))).type(torch.LongTensor)
train_X, train_Y = Variable(train_data_tensor), Variable(train_target_tensor)
# 分配测试数据
test_data_tensor = torch.from_numpy(np.vstack((data_numpy[30:38, 0:6], data_numpy[56:, 0:6]))).type(torch.FloatTensor)
test_target_tensor = torch.from_numpy(np.concatenate((target_numpy[30:38], target_numpy[56:]))).type(torch.LongTensor)
test_X, test_Y = Variable(test_data_tensor), Variable(test_target_tensor)

# ------- 构建神经网络 -------------
class FCNet(torch.nn.Module):
    def __init__(self):
        super(FCNet, self).__init__()
        # 定义一个三层的全连接神经网络，两个隐藏层的神经元个数都为12
        self.hidden1 = torch.nn.Linear(6, 12)
        self.hidden2 = torch.nn.Linear(12, 12)
        self.predict = torch.nn.Linear(12, 2)

    def forward(self, x):
        # 激活函数都用relu
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.predict(x)
        return x

net = FCNet()
print(net)

optimizer = torch.optim.Adam(net.parameters(), lr=0.001) # 定义优化器为Adam优化器，学习率为0.001
loss_func = torch.nn.CrossEntropyLoss() # 由于是分类问题，我们的loss function定义为交叉熵损失

prediction = net(train_X)
loss = loss_func(prediction, train_Y)
print('训练前的训练集合loss为{}'.format(loss))

# 训练1000个step
for t in range(1000):
    prediction = net(train_X) # 网络前向传播（预测）结果
    loss = loss_func(prediction, train_Y) # 计算损失（预测结果与标签的偏差）
    optimizer.zero_grad() # 梯度清零（pytorch中梯度会累加，故在反向传播前要先将梯度清零）
    loss.backward() # 反向传播，计算各个参数的梯度
    optimizer.step() # 更新网络参数

    if t % 10 == 0:
        prediction = torch.max(F.softmax(prediction), 1)[1]
        train_pred = prediction.numpy().squeeze()
        train_acc = sum(train_pred == train_Y.numpy()) / train_X.shape[0]
        print('第{}轮训练的训练集loss为{:.4f}，acc为{:.4f}'.format(t, loss, train_acc))

test_pred = net(test_X)
test_pred = torch.max(F.softmax(test_pred), 1)[1]
test_pred = test_pred.numpy().squeeze()
test_acc = sum(test_pred == test_Y.numpy()) / test_X.shape[0]
print('训练好的网络在测试集上的acc为{:.4f}'.format(test_acc))
