import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import models

import torchvision.transforms as transforms
import torchvision.datasets as dataset

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from model.res_unet import Resnet34_Unet

# 准备测试所用的模型
modelVGG = Resnet34_Unet(3, 1, False)
modelVGG.load_state_dict(torch.load("res-u-net.pth"))
res = models.resnet34(pretrained=True)  # 采用VGG16的预训练模型
print(modelVGG)

# 准备测试图像
img = cv.imread("rader.png")  # 读取本地一张图片
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()  # 展示原始测试图像

# 准备测试图像转化函数，因为只有将测试图像转化为tensor形式，才可以进行测试
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(512),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 测试图像由3D形式扩展为 [samples, rows, cols, channels]的4D数组
img = np.array(img)

img = transform(img)
print(img.size())
img = img.unsqueeze(0)
print(img.size())

# 接下来就需要访问模型所有的卷积层
no_of_layers = 0
conv_layers = []
re_layers =[]
r_o = 0

model_children = list(modelVGG.children())
model_c = list(res.children())

for child in model_children:
    if type(child) == nn.Conv2d:
        no_of_layers += 1
        conv_layers.append(child)
    elif type(child) == nn.Sequential:
        for layer in child.children():
            if type(layer) == nn.Conv2d:
                no_of_layers += 1
                conv_layers.append(layer)
print(no_of_layers)

for child in model_c:
    if type(child) == nn.Conv2d:
        r_o += 1
        re_layers.append(child)
    elif type(child) == nn.Sequential:
        for layer in child.children():
            if type(layer) == nn.Conv2d:
                r_o += 1
                re_layers.append(layer)
print(r_o)

# 将测试图像作为第一个卷积层的输入，使用for循环，依次将最后一个结果传递给最后一层卷积。
results = [conv_layers[0](img)]
print(conv_layers[1])

for i in range(1, len(re_layers)):
    results.append(re_layers[i][results[-1]])
print(results[-1].shape)

for i in range(1, len(conv_layers)):
    print(conv_layers[i])
    results.append(conv_layers[i](results[-1]))
outputs = results

# 依次展示特征可视化结果
for num_layer in range(len(outputs)):
    plt.figure(figsize=(50, 10))
    layer_viz = outputs[num_layer][0, :, :, :]
    layer_viz = layer_viz.data
    print("Layer ", num_layer + 1)
    for i, filter in enumerate(layer_viz):
        if i == 16:
            break
        plt.subplot(2, 8, i + 1)
        plt.imshow(filter, cmap='jet')  # 如果需要彩色的，可以修改cmap的参数
        plt.axis("off")
    plt.show()
    plt.close()