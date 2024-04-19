# author:"Tony"
# date:2024/4/19 7:32
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

# 定义数据转换
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

# 加载测试集
test_data_set = torchvision.datasets.MNIST(root='dataset', train=False, transform=torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(32, 32)),
    torchvision.transforms.ToTensor()
]), download=True)
#  train参数设置为False表示该文件是测试文件

# 设置数据加载器
test_data_size = len(test_data_set)
print(f'测试集长度为{test_data_set}')
test_data_loader = DataLoader(dataset=test_data_set, batch_size=64, shuffle=True, drop_last=True)


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
        return x  # 导入模型定义

model = MyNet()
# 保存模型参数
torch.save(model.state_dict(), 'MNIST_combined_state_dict.pth')

# 加载模型参数
model.load_state_dict(torch.load('MNIST_combined_state_dict.pth'))

# 设置模型为评估模式
model.eval()

accuracy = 0
total_accuracy = 0

# 禁用梯度计算，加快推理速度
with torch.no_grad():
    for i, data in enumerate(test_data_loader, 1):  # 添加enumerate来获取当前迭代次数i
        imgs, targets = data
        outputs = model(imgs)
        # 在0梯度运行可以提高运行速度

        batch_accuracy = (outputs.argmax(axis=1) == targets).sum().item()
        # 计算当前batch的正确预测数量
        total_accuracy += batch_accuracy

    # 计算整个测试集上的准确率
    accuracy = total_accuracy / test_data_size
    print(f'对测试集准确率: {accuracy}')