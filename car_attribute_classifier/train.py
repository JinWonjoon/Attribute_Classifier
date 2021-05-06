import argparse
import os
import numpy as np
import math
import itertools
import sys
from tqdm import tqdm
from time import sleep
from matplotlib import pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image

from models import *
from datasets import *
from utils import *



'''
    Hyper-paramters
'''
pascal3d_root = 'C:\\Users\\cglab\\pytorch-clickhere-cnn-f8988287a9e760a6825f2c91cdf8539fc12a4c6e\\PASCAL3D+_release1.1'
#model_path = 'checkpoint/photo_99.pth'
train_epoch = 60
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32
lr = 1e-3
reg = 1e-4

model = ResNet18().to(device)
# model = ResNet50().to(device)

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = CarAttribute(pascal3d_root, 'train', transform)
valid_dataset = CarAttribute(pascal3d_root, 'val', transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

criterion = MultiTaskLossWrapper(task_num=2).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=reg, amsgrad=False)
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 10)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.5)

loss_train_epoch = []
loss_valid_epoch = []
valid_acc_epoch = []
for epoch in tqdm(range(train_epoch), desc="Total epoch"):
    sleep(0.01)
    print("-"*40)
    loss_train = []
    train_cor = 0
    model.train()
    for batch, dataset in enumerate(tqdm(train_loader, desc="Train epoch{}".format(epoch+1), leave=False), 1):
        sleep(0.01)
        inputs, attributes, landmark, mask = dataset[0], dataset[1], dataset[2], dataset[3]
        landmark = landmark.view(landmark.size(0), -1)
        mask = mask.view(mask.size(0), -1)

        optimizer.zero_grad() # Init gradient
        
        # Forward step
        inputs = inputs.to(device)                
        attributes = attributes.to(device)
        landmark = landmark.to(device)
        mask = mask.to(device)
        pred_attribute, pred_landmark = model(inputs)
        pred_landmark *= mask
        loss = criterion(pred_attribute, pred_landmark, attributes, landmark)
        
        # Backward step
        loss.backward()
        optimizer.step()
    
        # Save loss of mini batch
        loss_train += [loss.item()]

        # Tensor to Numpy
        attributes_np = fn_tonumpy(attributes)
        output_np = fn_tonumpy(pred_attribute)
        output_np[output_np >= 0.5] = 1
        output_np[output_np < 0.5] = 0
    
        # Count correct prediction
        # Add
        train_cor += np.count_nonzero(output_np == attributes_np)
    train_acc = train_cor / len(loss_train)
    loss_train_epoch.append(np.mean(loss_train))
    print(f' Train epoch : {epoch+1} || Loss : {np.mean(loss_train)} || train_acc : {train_acc}')
        
    # Validation
    loss_valid = []
    valid_cor = 0
    model.eval()
    for batch, dataset in enumerate(tqdm(valid_loader, desc="Valid", leave=False), 1):
        sleep(0.01)
        inputs, attributes, landmark, mask = dataset[0], dataset[1], dataset[2], dataset[3]
        landmark = landmark.view(landmark.size(0), -1)
        mask = mask.view(mask.size(0), -1)

        # Forward step
        inputs = inputs.to(device)                
        attributes = attributes.to(device)
        landmark = landmark.to(device)
        mask = mask.to(device)
        pred_attribute, pred_landmark = model(inputs)
        pred_landmark *= mask
        loss = criterion(pred_attribute, pred_landmark, attributes, landmark)

        # Save loss of mini batch
        loss_valid += [loss.item()]
        
        # Tensor to Numpy
        attributes_np = fn_tonumpy(attributes)
        output_np = fn_tonumpy(pred_attribute)
        output_np[output_np >= 0.5] = 1
        output_np[output_np < 0.5] = 0
    
        # Count correct prediction
        # Add
        valid_cor += np.count_nonzero(output_np == attributes_np)
    
    valid_acc = valid_cor / len(loss_valid)
    loss_valid_epoch.append(np.mean(loss_valid))
    valid_acc_epoch.append(valid_acc)
    print(f' Valid epoch : {epoch+1} || Loss : {np.mean(loss_valid)} || valid_acc : {valid_acc}')
    scheduler.step()

save(ckpt_dir=ckpt_dir, net = model, optimizer = optimizer, epoch = epoch, model_name='resnet18')
'''
for epoch in range(train_epochs):
    running_loss = 0.0

    for i, data in enumerate(trainloader):
        images, labels = data[0], data[1]
        images, labels = images.to(device), labels.to(device) 

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        print(f'[{epoch+1}, {i+1}] loss: {running_loss/100:.3f}')
        running_loss = 0.0

    scheduler.step()
    torch.save(model.state_dict(), f'checkpoint/photo_res18_{epoch}.pth')
'''