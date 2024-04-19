# author:"Tony"
# date: 2024/4/16 14:56
import os
import torch.optim
import torchvision
from torch import nn
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#  显卡只能对网络,损失函数，图片target加速

train_data_set = torchvision.datasets.MNIST(root='dataset', train=True, transform=torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(32, 32)),
    torchvision.transforms.ToTensor()
]), download=True)
# LeNet 输入是32*32而MNIST数据集大小是28*28因此需要进行格式转换，第二个转换是数据类型转换否则运算会出错，download参数True表示自动下载

test_data_set = torchvision.datasets.MNIST(root='dataset', train=False, transform=torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(32, 32)),
    torchvision.transforms.ToTensor()
]), download=True)
#  train参数设置为False表示该文件是测试文件

# 进行筛选
classes = [5, 6, 7, 8, 9]
indices = [i for i, (img, label) in enumerate(train_data_set) if label in classes]
# 对索引重新排序
indices.sort()
train_data_set = torch.utils.data.Subset(train_data_set, indices)

train_data_size = len(train_data_set)
test_data_size = len(test_data_set)
#  计算准确度需要单独用长度变量因此提前用变量承接

print(f'训练集长度为{train_data_set}')
print(f'测试集长度为{test_data_set}')

train_data_loader = DataLoader(dataset=train_data_set, batch_size=64, shuffle=True, drop_last=True)
test_data_loader = DataLoader(dataset=test_data_set, batch_size=64, shuffle=True, drop_last=True)
#  从数据集中加载

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Flatten(),
            nn.Linear(in_features=16 * 5 * 5, out_features=120),
            nn.Linear(in_features=120, out_features=84),
            nn.Linear(in_features=84, out_features=10)
        )

    #  参考LeNet网络结构，Flatten将二维数组展开成一维向量

    def forward(self, x):
        x = self.model(x)
        return x


mynet = MyNet()
#  print(mynet)
mynet = mynet.to(device)

loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)
#  分类问题使用交叉熵损失函数

learning_rate = 1e-2
optim = torch.optim.SGD(mynet.parameters(), learning_rate)

train_step = 0

mini_batch_size = 100
#  每100次输出一次损失函数
epoch = 20

if __name__ == '__main__':
    #  改行代码后面所有内容均不会被导入其他模块，main指文件夹名称
    for i in range(epoch):
        # 不加range就不是可迭代对象
        print(f'----------第{i+1}轮训练----------')
        mynet.train()
        #开始训练标志
        for data in train_data_loader:
            #  读入训练集数据
            imgs,targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            #  print(imgs.shape)
            outputs = mynet(imgs)
            #  print(outputs.shape)

            loss = loss_fn(outputs,targets)
            #  定义损失函数，一个是目标一个是输出
            optim.zero_grad()
            #  梯度清零
            loss.backward()
            #  反向传播
            optim.step()
            #  更新梯度

            train_step += 1
            if train_step % (mini_batch_size) == 0:
                print(f'第{train_step}次训练，loss={loss.item()}')
            #  固定轮数输出损失函数
        mynet.eval()
        accuracy = 0
        total_accuracy = 0
        with torch.no_grad():
            for data in test_data_loader:
                imgs,targets = data
                imgs = imgs.to(device)
                targets = targets.to(device)
                outputs = mynet(imgs)
                # 在0梯度运行可以提高运行速度

                accuracy = (outputs.argmax(axis=1) == targets).sum()
                #  取每一行最大值？？？True和False可以求和python真神奇，计算每一个batch也就是64
                total_accuracy += accuracy
            print(f'第{i+1}轮训练结束，准确率{total_accuracy/test_data_size}')
            #  与第21行呼应
            #  torch.save(mynet,f'MNIST_{i}_acc_{total_accuracy/test_data_size}.pth')
            # 定义文件夹路径
            save_dir = 'C:/Users/HP/Desktop/MNISTProjects/pythonProject1/5-9数据集20轮参数文件'

            # 确保文件夹存在，如果不存在则创建
            os.makedirs(save_dir, exist_ok=True)

            # 使用 os.path.join() 函数拼接文件夹路径和文件名
            save_path = os.path.join(save_dir, f'MNIST_{i}_acc_{total_accuracy / test_data_size}.pth')

            # 保存模型到指定路径
            torch.save(mynet, save_path)


