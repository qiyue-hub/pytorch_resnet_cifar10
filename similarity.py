# This file is modified from https://pytorch.org/tutorials/beginner/fgsm_tutorial.html .
# Thanks for their work!

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

epsilons = [0, .05, .1, .15, .2, .25, .3]
pretrained_model = "save_resnet20/model.th"
pretrained_model2 = "pretrained_models/resnet20-12fca82f.th"
use_cuda = True
plot_examples = 3

# LeNet Model definition
import resnet as resnet
Net = resnet.resnet20

# MNIST Test dataset and dataloader declaration
test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])),
        batch_size=1, shuffle=True)

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# Initialize the network
model = torch.nn.DataParallel(Net()).to(device)
model2 = torch.nn.DataParallel(Net()).to(device)

# Load the pretrained model
model.load_state_dict(torch.load(pretrained_model, map_location='cpu')['state_dict'])
model2.load_state_dict(torch.load(pretrained_model2, map_location='cpu')['state_dict'])

# Set the model in evaluation mode. In this case this is for the Dropout layers
model.eval()
model2.eval()

def test(model, model2, device, test_loader):
    X, Y = [], []
    i = 0

    # Loop over all examples in test set
    for data, target in test_loader:
        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Forward pass the data through the model
        middle, output = model(data)
        X.append(middle[len(middle) // 2].detach())

        # Re-classify the perturbed image
        middle2, output2 = model2(data)
        Y.append(middle2[len(middle2) // 2].detach())

        i += 1
        if i > 1000:
            break

    X = [x.view(-1) for x in X]
    Y = [y.view(-1) for y in Y]
    i = 0
    for x, y in zip(X, Y):
        if i >= plot_examples:
            break
        i += 1
        x = list(x.numpy())
        y = list(y.numpy())
        x = sorted(x)
        y = sorted(y)
        plt.plot(x)
        plt.plot(y)
        plt.show()

    X = torch.stack(X)
    Y = torch.stack(Y)

    Z = torch.matmul(Y.t(), X)
    X = torch.matmul(X.t(), X)
    Y = torch.matmul(Y.t(), Y)
    z = (Z ** 2).sum()
    x = (X ** 2).sum().sqrt()
    y = (Y ** 2).sum().sqrt()
    print('Similarity: {}'.format((z / (x * y)).item()))

test(model, model2, device, test_loader)
