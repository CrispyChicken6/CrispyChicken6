# author:"Tony"
# date:2024/4/19 6:58
import torch
import torchvision
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


# 定义模型结构
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

    def forward(self, x):
        x = self.model(x)
        return x


# 加载两个.pth文件
model_path_1 = 'C:/Users/HP/Desktop/MNISTProjects/pythonProject1/0-4数据集20轮参数文件/MNIST_16_acc_0.5105999708175659.pth'
model_path_2 = 'C:/Users/HP/Desktop/MNISTProjects/pythonProject1/5-9数据集20轮参数文件/MNIST_12_acc_0.4788999855518341.pth'
model_1 = torch.load(model_path_1)
model_2 = torch.load(model_path_2)

# 定义数据转换
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

# 加载测试集
test_set = MNIST(root='dataset', train=False, download=True, transform=transform)

# 设置数据加载器
batch_size = 64
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# 定义新模型并初始化参数
new_model = MyNet()

# 遍历两个模型的参数并进行加权平均
t = 0.5  # 权重
for param_new, param_1, param_2 in zip(new_model.parameters(), model_1.parameters(), model_2.parameters()):
    param_new.data = t * param_1.data + (1 - t) * param_2.data

# 保存新模型的参数为.pth文件
output_model_path = 'C:/Users/HP/Desktop/MNISTProjects/pythonProject1/MNIST_combined.pth'
torch.save(new_model, output_model_path)

print(f"新模型的参数已保存到 {output_model_path}")

