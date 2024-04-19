# author:"Tony"
# date:2024/4/18 16:02
import torch
import torchvision
from torchvision import transforms
import torchvision.datasets as datasets

#  定义数据转换
transform = transforms.Compose([transforms.ToTensor()])
#  加载数据集并转换为pytorch张量
trainset = torchvision.datasets.MNIST(root='dataset', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='dataset', train=False, download=True, transform=transform)
#  分类
classes = [0, 1, 2, 3, 4]
#  对于trainset中的每个数据样本，如果其对应的类别c在classes列表中，则将其索引值i加入到结果列表indices中。
#  其中i表示索引值，(e, c)表示数据样本和对应的类别。
indices = [i for i, (e, c) in enumerate(trainset) if c in classes]
trainset = torch.utils.data.Subset(trainset, indices)

indices = [i for i, (e, c) in enumerate(testset) if c in classes]
testset = torch.utils.data.Subset(testset, indices)

#  torch.save(trainset, 'filtered_trainset5-9.pt')
torch.save(trainset, 'filtered_testset0-4.pt')