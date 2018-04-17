# visualize the layers of a CNN

import numpy as np
import os
import cv2

import torch
import torch.nn as nn
from torch.autograd.variable import Variable
from torchvision import models, transforms
import torchvision.datasets as dset
import torch.autograd as autograd
import pandas as pd
import numpy as np 
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim

import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt



root = './data'
download = False  # download MNIST dataset or not

# normalize the data and set up the train and test loaders

trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
train_set = dset.MNIST(root=root, train=True, transform=trans, download=True)
test_set = dset.MNIST(root=root, train=False, transform=trans)

batch_size = 100

train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True)
test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=False)

#########################
# Define the model
#########################

# simple CNN - throughout I use 'same' padding because I don't want the activations to decrease in size 

class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        # conv1
        self.conv1 = nn.Conv2d(1, 16, 5, stride = 1, padding = 2) # this is 'same' padding
        # conv 2: input size = (-1,28,28,16)
        self.conv2 = nn.Conv2d(16, 32, 5, stride = 1, padding = 2) # 'same' padding again
        self.conv2_bn = nn.BatchNorm2d(32) # (-1,28,28,32) out
        # conv 3: input_size = (-1, 28, 28, 32) in
        self.conv3 = nn.Conv2d(32, 64, 5, stride = 1, padding = 2) # (-1,28, 28 ,64) out
        # dropout prob
        self.dropout = nn.Dropout(p=0.5)
        # dense1
        self.fc1 = nn.Linear(28*28*64, 10)
    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2_bn(self.conv2(x1)))
        x3 = F.relu(self.conv3(x2))
        x = x3.view(-1, 28*28*64) # always need to reshape before fc layer!
        x = self.fc1(x)
        return x1, x2, x3, F.log_softmax(x, dim = 1) # output from each layer so that we can visualize what's going on


net = CNN_Model()

optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()

epochs = 5

########################
# Optimization routine
########################

for epoch in range(epochs):
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        o1, o2, o3, outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs.data, 1)
        if (batch_idx % 10) == 0:
            print('Iteration {} finished \nLoss = {} \n'.format(batch_idx, np.round(loss.data[0], 3)))

print('Training finished!')

########################
# Visualizations
########################

# plot an image
# use the same one each time!
im_plot = inputs[1].data.numpy().reshape(28,28)

im = plt.imshow(im_plot, cmap='gray', interpolation='none')
cbar = plt.colorbar(im)
plt.show()

# pass it through the network

o1, o2, o3, outputs = net(inputs[1].view(1,1,28,28))

# have a look at first layer activations

for i in range(o1.shape[1]):
	im = plt.imshow(o1[0][i].data.numpy(), cmap='gray', interpolation='none')
	cbar = plt.colorbar(im)
	plt.show()

# second layer activations

for i in range(o2.shape[1]):
	im = plt.imshow(o2[0][i].data.numpy(), cmap='gray', interpolation='none')
	cbar = plt.colorbar(im)
	plt.show()

# third layer activations

for i in range(o3.shape[1]):
	im = plt.imshow(o3[0][i].data.numpy(), cmap='gray', interpolation='none')
	cbar = plt.colorbar(im)
	plt.show()

# and the weights of the convolution filters themselves

for i in range(net.conv1.weight.shape[0]):
	im = plt.imshow(net.conv1.weight[i].data.numpy().reshape(5,5), cmap='gray', interpolation='none')
	cbar = plt.colorbar(im)
	plt.show()

# it gets a bit complicated here - for every one of the 16 filters in the first layer there are 32 in the second
# even more so for the final layer - for every one of the 32 filters in the second layer there are 64 in the third

for i in range(net.conv2.weight[1].shape[0]):
	im = plt.imshow(net.conv2.weight[1][i].data.numpy().reshape(5,5), cmap='gray', interpolation='none')
	cbar = plt.colorbar(im)
	plt.show()

