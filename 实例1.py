import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 这是一个二分类任务，我们利用神经网络，将二维坐标图上的两堆点给分开

n_data = torch.ones(1000, 2) # 数据格式为为1000*2的矩阵，相当于有1000个维度为2的样本点
x0 = torch.normal(3*n_data, 2) # 第一堆点：x0是以3为均值，2为标准差生成随机的正态分布数据
y0 = torch.zeros(1000) # x0的标签为0
x1 = torch.normal(-3*n_data, 2) # 第二堆点：x1是以-3为均值，2为标准差生成随机的正态分布数据
y1 = torch.ones(1000) # x1的标签为1
x = torch.cat((x0, x1), 0).type(torch.FloatTensor) # 沿维度0方向（行方向）将x0、x1矩阵拼接到一起，数为Float格式。x为2000*2的矩阵。
y = torch.cat((y0, y1), ).type(torch.LongTensor) # 将y0、y1矩阵拼接到一起
X, Y = Variable(x), Variable(y) # 训练数据需要从Tensor转化为Variable

# 展示我们生成的数据
# 注意matplotlib库只能处理一般形式或numpy格式的数据，所以我们要先将数据转化为numpy数据：X.data.numpy()
plt.scatter(X.data.numpy()[:, 0], X.data.numpy()[:, 1], c=Y.data.numpy(), s=100, lw=0)
plt.show()

# ------- 下面我们构建一个两层的全连接神经网络来分类这些数据 --------------------
# 一.快速构建法
# 使用torch.nn.Sequential方法
net = torch.nn.Sequential(
    torch.nn.Linear(2, 10), # 由于我们数据是二维的，输入填2，另外我们设第一个隐藏层的神经元个数为10
    torch.nn.ReLU(), # 接一个Relu可以增加网络的非线性，提高网络拟合数据的能力
    torch.nn.Linear(10, 2), # 由于第一个隐藏层的神经元个数为10，则第二个隐藏层的输入个数必须是10。
                            # 另外，我们的任务是二分类问题，所以输出填2
)
print(net) # 可以把定义好的网络打印出来看看

# # 二.基本构建法
# class Net(torch.nn.Module):
#     def __init__(self, n_feature, n_hidden, n_output):
#         super(Net, self).__init__()
#         self.hidden = torch.nn.Linear(n_feature, n_hidden)
#         self.predict = torch.nn.Linear(n_hidden, n_output)
#
#     def forward(self, x):
#         x = F.relu(self.hidden(x))
#         x = self.predict(x)
#         return x
#
# net = Net(2, 10, 2)
# print(net)

plt.ion() # 动态打印
plt.show()

optimizer = torch.optim.SGD(net.parameters(), lr=0.001) # 定义优化器为随机梯度下降优化器，学习率为0.001
loss_func = torch.nn.CrossEntropyLoss() # 由于是分类问题，我们的loss function定义为交叉熵损失

# 训练500个step
for t in range(500):
    prediction = net(X) # 网络前向传播（预测）结果
    loss = loss_func(prediction, Y) # 计算损失（预测结果与标签的偏差）
    optimizer.zero_grad() # 梯度清零（pytorch中梯度会累加，故在反向传播前要先将梯度清零）
    loss.backward() # 反向传播，计算各个参数的梯度
    optimizer.step() # 更新网络参数

    # 每2个训练step，打印一次
    if t % 2 == 0:
        plt.cla()
        # softmax的作用是，将网络的k个输出值，变为k个加和为1的概率值
        # torch.max(M,1)的作用是，沿维度1方向（列），求M矩阵每行的最大值
        # torch.max()[1]能够将最大值结果转变为0、1的格式
        prediction = torch.max(F.softmax(prediction), 1)[1] # 得到预测结果（网络输出为两个数字，分别代表0类点和1类点的概率，数字大的那个类我们认为是网络预测的类别）
        pred_y = prediction.data.numpy().squeeze() # numpy.squeeze的作用是删去大小1的维度。即预测的类别不需要区分是行向量或列向量，只需要是一维向量就行
        target_y = Y.data.numpy()
        accuracy = sum(pred_y == target_y) / 2000 # 计算此次step在训练数据上预测正确的概率

        plt.scatter(X.data.numpy()[:, 0], X.data.numpy()[:, 1], c=pred_y, s=100) # 绘制此次step预测的类别图
        plt.text(1.5, -4, 'Accuracy = %.2f' % accuracy) # 将accuracy也打印在图上
        plt.pause(0.1)

plt.ioff()
plt.show()
