#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 04:16:14 2019

@author: ankur
"""

import models as m
import densenet as x
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
#import matplotlib.pyplot as plt 
import torch


from sklearn.metrics import confusion_matrix
import numpy as np

tr1=  transforms.Compose([transforms.Resize((224,224))])

train_dataset = ImageFolder('../../cat_dog_dataset/training_set',transform=tr1)
train_loader = DataLoader(train_dataset, 
                          batch_size = 1, 
                          shuffle = True, 
                          num_workers = 1)

test_dataset = ImageFolder('../../cat_dog_dataset/test_set',transform=tr1)
test_loader = DataLoader(test_dataset, 
                          batch_size = 1, 
                          shuffle = False, 
                          num_workers = 1)

# GLOBAL VARIABLES
noe = 25
noc = 2

# MODEL INSTANCE 1
net = m.resnet50()
net.fc = nn.Linear(2048,noc)
#net=net.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

def train(epoch):
    net.train()
    
    train_loss_epoch=0.0
#    running_loss=0
    for batch in train_loader:
        
        inputs, labels = batch
        inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad()
#        print(inputs)
        # FORWARD PASS
            
        out = net (inputs)
        # CALCULATE LOSS
        loss = criterion(out, labels)
        # BACK PROPAGATION
        loss.backward()
        # WEIGHT UPDATION
        optimizer.step()
#        print(loss.data[0])
        
        train_loss_epoch+=loss.item()
#    total_loss_epoch+=t
    print(epoch,train_loss_epoch)
    
def test():
    final_updated_cm=np.zeros((2,2),dtype=int)
#    confusion_matrix = meter.ConfusionMeter(3)
    net.eval()
    test_loss=0.0
    correct=0.0
    for inputs,labels in test_loader:
#        updated_cm=torch.zeros(23,23)
        inputs, labels = Variable(inputs), Variable(labels)
        out = net (inputs)
        #print(out)
        loss=criterion(out,labels)
        test_loss+=loss.item()
        
#        pred=out.data.max(1, keepdim=True)[1]
        pred=torch.max(out.data)
        #print(pred)
#        pred=torch.max(out.data,1)[1]
        correct+=pred.eq(labels.data.view_as(pred)).cpu().sum()
#        confusion_matrix.add(out.data.squeeze(), labels)
        cm=confusion_matrix(labels.data,pred.data,labels=[0,1])
#        print(cm)
        final_updated_cm=final_updated_cm+cm
    test_loss /=len(test_loader.dataset)
    print('\nTest Set: Average loss: {:.4f},Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss,correct,len(test_loader.dataset),
          100.0 * correct / len(test_loader.dataset)))
    
#    t=(100.0*correct/len(test_loader.dataset)).item()
#    f=open('2.txt','w')
#    f.write(str(t))
#    f.close()    
#    print(t)
        
    print(final_updated_cm)
    
for epoch in range(noe):
    train(epoch)
test() 