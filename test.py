# author:"Tony"
# date:2024/4/16 17:51
import os
import torchvision.transforms
from PIL import Image
from torch import nn
import torch


#  PIL 是 Python Imaging Library 的简称，它是 Python 中用于图像处理的库

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


#  main.py文件里的

root_dir = 'test_number'
number_name = 'test.png'
#  只能一张一张图片导入，可以采用拼接一次性导入多张文件,这步有问题！！！img_path = os.path.join(root_dir, number_name)实现不了

img_path = "C:/Users/HP/Desktop/MNISTProjects/test_number/test-10.png"
img = Image.open(img_path)
img_1 = img.convert('RGB')
#  将RGBA格式转换为RGB格式
img_2 = img.convert('1')
#  转换为单通道,后面要改干净！！！

#  img.show()
#  print(img)
#  print(img_path)
tran_pose = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(32, 32)),
    torchvision.transforms.ToTensor()
])
#  将图片像素值进行转换使之满足LeNet输入端要求，第二个转换是数据类型转换否则运算会出错

mynet = torch.load('MNIST_18_acc_0.982699990272522.pth', map_location=torch.device('cpu'))
#  测试时数据较少可以使用cpu处理
#  print(mynet)

img_2 = tran_pose(img_2)
print(img_2.shape)
#  你会发现神奇的四通道，这样的图片是不可以测试的
img_2 = torch.reshape(img_2, (1, 1, 32, 32))
print(img_2.shape)
#  RGBA格式是四个通道，3通道1通道都会报错

mynet.eval()
#  测试开始的测试标志
with torch.no_grad():
    #  在无梯度情况下节省运算效率
    output = mynet(img_2)
    #  输出样例应该为torch.Size([1, 10])，表示1个数据和10个分类,可以使用print(output.shape)来验证猜想
    number = output.argmax(axis=1).item()
    #  item将我们要找到的最大值转化为数字类型
    print(f'识别的数字是{number}')

# 结论该模型对于具有三通道的黑白图片的拟合效果比较好，而对三通道彩色图片拟合效果较差
