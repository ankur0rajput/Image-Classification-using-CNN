# -*- coding: utf-8 -*-

import torchvision.models as models
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
import matplotlib.pyplot as plt 
import torch
from sklearn.metrics import confusion_matrix
import numpy as np


from sampler import ImbalancedDatasetSampler
# TRANSFORMS
tr = transforms.Compose([transforms.Resize((224,224)),
                         transforms.ToTensor()])

# DATA LOADERS

train_dataset = ImageFolder('/home/cse/ug/16075044/train/separated_2classes',transform = tr)


train_loader = DataLoader(train_dataset, 
                          batch_size = 6, 
                          shuffle = True, 
                          num_workers = 2)

'''
train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    sampler=ImbalancedDatasetSampler(train_dataset),
    batch_size=6
)
'''

test_dataset = ImageFolder('/home/cse/ug/16075044/test/separated_2classes',transform = tr)
test_loader = DataLoader(test_dataset, 
                          batch_size = 1, 
                          shuffle = True, 
                          num_workers = 2)



# GLOBAL VARIABLES
noe = 100
noc = 2


# MODEL INSTANCE
net = models.resnet50()
net.fc = nn.Linear(2048,noc)
net=net.cuda()

# LOSS FUNCTION AND OPTIMIZER
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

def train(epoch):
    net.train()
    
    train_loss_epoch=0.0
#    running_loss=0
    for batch in train_loader:
        
        inputs, labels = batch
        inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
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
    '''
    plt.plot(epoch,train_loss_epoch)
    plt.title('train_loss_vs_epoch')
    plt.plot(epoch,train_loss_epoch,'r',linewidth=5)
    plt.xlabel('epoch')
    plt.ylabel('train_loss_epoch')
    plt.show()
    plt.savefig(r'/home/nibaran/plot.png')
    '''
   
def test():
    final_updated_cm=np.zeros([2,2],int)
    net.eval()
    test_loss=0.0
    correct=0.0
    
    for inputs,labels in test_loader:
#        updated_cm=torch.zeros(23,23)
        inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
        out = net (inputs)
        loss=criterion(out,labels)
        test_loss+=loss.item()
        
        pred=out.data.max(1, keepdim=True)[1]
        
        correct+=pred.eq(labels.data.view_as(pred)).sum()
        #cm=confusion_matrix(labels,out)
        #final_updated_cm=final_updated_cm+cm
        
        for i,l in enumerate(labels):
            final_updated_cm[l.item(),pred.item()] +=1
            
    test_loss /=len(test_loader.dataset)
    print('\nTest Set: Average loss: {:.4f},Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss,correct,len(test_loader.dataset),
          100.0 * correct / len(test_loader.dataset)))
            
    print(final_updated_cm)        
        
for epoch in range(noe):
    train(epoch)
test()
        
        
        
    
