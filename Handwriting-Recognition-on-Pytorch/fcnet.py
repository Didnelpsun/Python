#!/usr/bin/env python
# encoding: utf-8

"""
代码参考：
全连接神经网络：https://www.cnblogs.com/liualexsone/p/11355217.html
"""

import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data.dataloader as dataloader
import torch.nn as nn
import torch.optim as optim
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


train_set = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    transform=transforms.ToTensor(),
    download=True
)
train_loader = dataloader.DataLoader(
    dataset=train_set,
    batch_size=100,
    shuffle=False,
)

test_set = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    transform=transforms.ToTensor(),
    download=True
)
test_loader = dataloader.DataLoader(
    dataset=test_set,
    batch_size=100,
    shuffle=False,
)


class NeuralNet(nn.Module):

    def __init__(self, input_num, hidden_num, output_num):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_num, hidden_num)
        self.fc2 = nn.Linear(hidden_num, output_num)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        y = self.fc2(x)
        return y


epoches = 20
lr = 0.001
input_num = 784
hidden_num = 500
output_num = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = NeuralNet(input_num, hidden_num, output_num)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)


for epoch in range(epoches):
    for i, data in enumerate(train_loader):
        (images, labels) = data
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        output = model(images)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, epoches, loss.item()))


with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        output = model(images)
        _, predicted = torch.max(output, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print("The accuracy of total {} images: {}%".format(total, 100 * correct/total))
