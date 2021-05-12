import torch
from torch.autograd import Variable  # 使用variable包住数据
import torch.nn.functional as F  # 激励函数F
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt  # 画图的模块


x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
# unsqueeze将一维的数据变成二维的数据。torch只会处理二维的数据
# x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2 * torch.rand(x.size())
# y是二次方加上一些噪点的影响，后面是噪点
# noisy y data (tensor), shape=(100, 1)
x, y = Variable(x), Variable(y)

# 画图，scatter打印散点图
plt.scatter(x.data.numpy(), y.data.numpy())
plt.show()

import torch
import torch.nn.functional as F  # 激励函数都在这


class Net(torch.nn.Module):  # 继承 torch 的 Module模块
    def __init__(self, n_feature, n_hidden, n_output):
        # init是搭建层需要的信息
        super(Net, self).__init__()  # 继承 __init__ 功能
        # 定义每层用什么样的形式
        self.hidden = torch.nn.Linear(n_feature,n_hidden)  # 隐藏层线性输出，层命名为hidden，n_feature层输入，n_hidden隐藏层的神经元，本函数输出隐藏层神经元的个数。
        self.predict = torch.nn.Linear(n_hidden, n_output)  # 输出层线性输出，预测神经层predict，n_hidden隐藏层神经元个数，n_output输出

    def forward(self, x):  # 这同时也是 Module 中的 forward 功能
        # 正向传播输入值, 神经网络分析出输出值，x输入值，
        x = F.relu(self.hidden(x))  # 激励函数(隐藏层的线性值)
        x = self.predict(x)  # 输出值，这里不用激励函数因为在大多数回归问题中，预测值分布从负无穷到正无穷，用了激励函数，会把取值截断。
        return x


net = Net(n_feature=1, n_hidden=10, n_output=1)
# feature=1，只包含了x一个信息，这里假设隐藏层有10个神经元，输出值y有一个

print(net)  # net 的结构
"""
Net (
  (hidden): Linear (1 -> 10)
  (predict): Linear (10 -> 1)
)
"""

# optimizer 是训练的工具
optimizer = torch.optim.SGD(net.parameters(), lr=0.2)  # 传入 net 的所有参数, 学习效率
loss_func = torch.nn.MSELoss()  # 预测值和真实值的误差计算公式 (均方差处理回归)

for t in range(100):
    prediction = net(x)  # 喂给 net 训练数据 x, 输出预测值

    loss = loss_func(prediction, y)  # 计算两者的误差

    optimizer.zero_grad()  # 清空上一步的残余更新参数值
    loss.backward()  # 误差反向传播, 计算参数更新值
    optimizer.step()  # 将参数更新值施加到 net 的 parameters 上

    if t % 2 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)



plt.ioff()  # 画图
plt.show()

torch.save(net,'net.pkl')
torch.save(net.state_dict(),'net_params.pkl')