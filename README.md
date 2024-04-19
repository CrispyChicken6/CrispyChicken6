# merging的分析与设计
## 本仓库代码的主要功能
1. 是人为对MNIST数据集划分（使用pytorch）
2. 使用pytorch搭建lenet网络结构
3. 将划分后的数据集放入神经网络进行训练，保存两个数据集生成的参数文件
4. 通过python语言实现参数文件的基于插值合并然后评估新参数文件下模型的性能

## pythonProject1文件的介绍
1. main.py对MNIST数据集进行划分并在lenet上进行训练，将每一个epoch保存在对应的参数文件夹下
2. test.py加载训练好的模型并对test_number文件夹下内容进行测试
3. devide.py对数据集进行划分并把划分后的数据集保存为pt文件
4. lerp.py对不同参数文件进行插值合并
5. look parameter对参数文件进行可视化并保存为txt文件
6. eval.py对参数文件进行评估计算准确率

