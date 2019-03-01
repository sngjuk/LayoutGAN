import numpy as np
import torch
from torchvision import datasets
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

from models import *
from datasets import *
import sys

def real_loss(D_out, smooth=False):
    labels = None
    batch_size = D_out.size(0)
    if smooth:
        labels = torch.ones(batch_size)*0.9
    else:
        labels = torch.ones(batch_size)
    
    crit = nn.BCEWithLogitsLoss()
    loss = crit(D_out.squeeze(), labels)
    return loss

def fake_loss(D_out):
    batch_size = D_out.size(0)
    labels = torch.zeros(batch_size)
    crit = nn.BCEWithLogitsLoss()    
    loss = crit(D_out.squeeze(), labels)
    return loss


# Download MNIST data.
_ = datasets.MNIST(root='data', train=True, download=True, transform=None)

lr = 0.00002
beta1 = 1.0
beta2 = 1.0
batch_size = 60
element_num = 128

G = Generator()
D = Discriminator()
print(G)
print(D)

d_optimizer = optim.Adam(D.parameters(), lr)
g_optimizer = optim.Adam(G.parameters(), lr)

num_epochs = 40
train_data = MnistLayoutDataset()
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)

cls_num = 1
geo_num = 2

D.train()
G.train()
for epoch in range(num_epochs):
    for batch_i, real_images in enumerate(train_loader):
        batch_size = real_images.size(0)
        
        # Train Discriminator.
        d_optimizer.zero_grad()
        
        D_real = D(real_images)
        d_real_loss = real_loss(D_real)

        # !Random layout input generation have logic error, should be fixed.
        zlist = []
        for i in range(batch_size):
            cls_z = np.ones((batch_size, cls_num))
            geo_z = np.random.normal(0, 1, size=(batch_size, geo_num))

            z = torch.FloatTensor(np.concatenate((cls_z,geo_z), axis=1))
            zlist.append(z)
        
        fake_images = G(torch.stack(zlist))
        
        D_fake = D(fake_images)
        d_fake_loss = fake_loss(D_fake)

        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        d_optimizer.step()
        
        # Train Generator
        g_optimizer.zero_grad()
        
        # !Random layout input generation have logic error, should be fixed.
        zlist2 = []
        for i in range(batch_size):
            cls_z = np.ones((batch_size, cls_num))
            geo_z = np.random.normal(0, 1, size=(batch_size, geo_num))

            z = torch.FloatTensor(np.concatenate((cls_z,geo_z), axis=1))
            zlist2.append(z)
        
        fake_images2 = G(torch.stack(zlist2))
        D_fake = D(fake_images2)
        g_loss = real_loss(D_fake)
        
        if batch_i % print_every == 0:
            print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                    epoch+1, num_epochs, d_loss.item(), g_loss.item()))
