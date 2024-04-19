# author:"Tony"
# date:2024/4/17 17:26
import torch
from torch import nn


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


# 加载.pth文件
model_path = 'C:/Users/HP/Desktop/MNISTProjects/pythonProject1/MNIST_combined.pth'  # 将路径替换为你的.pth文件路径
loaded_model = torch.load(model_path)

# 创建一个txt文件来保存参数
output_file = '合并参数可视化.txt'

# 打开文件并将参数写入
with open(output_file, 'w') as f:
    f.write("Loaded Model Parameters:\n\n")
    for name, param in loaded_model.named_parameters():
        f.write(f"Parameter Name: {name}\n")
        f.write(f"Parameter Shape: {param.shape}\n")
        f.write(f"Parameter Values:\n{param}\n\n")

print(f"参数已保存到 {output_file}")

# 查看模型结构
print("Loaded Model Architecture:")
print(loaded_model)

# 查看模型参数
print("\nLoaded Model Parameters:")
for name, param in loaded_model.named_parameters():
    print(f"Parameter Name: {name}")
    print(f"Parameter Shape: {param.shape}")
    print(f"Parameter Values:\n{param}\n")
