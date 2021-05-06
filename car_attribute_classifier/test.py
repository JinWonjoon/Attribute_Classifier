import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models import *
from webcari import *

from torch.utils.data import DataLoader

import os
import numpy as np

img_path = '/Kiwi/Data1/Dataset/WebCaricature/WebCaricature_aligned'
label_path = '/Kiwi/Data1/Dataset/WebCariA_Dataset/WebCariA_Dataset/all_photo_data.txt'
label_name_path = '/Kiwi/Data1/Dataset/WebCariA_Dataset/WebCariA_Dataset/labels.txt'
model_path = 'checkpoint/photo_res18_99.pth'
device ='cuda'

with open(label_name_path, 'r') as f:
    lines = f.readlines()
    attributes = [line.rstrip() for line in lines]


model = ResNet18().to(device)
model.load_state_dict(torch.load(model_path))


dataset = ImageDataset(img_path, label_path, mode='test')
testloader = DataLoader(dataset, batch_size=1, shuffle=False)

model.eval()

corrected = np.zeros(50)
print_interval = 100
with torch.no_grad():
    for i, data in enumerate(testloader):
        images, labels = data[0], data[1]
        images, labels = images.to(device), labels.to(device)

        predicted = model(images)
        predicted[predicted > 0.5] = 1
        predicted[predicted <= 0.5] = 0

        corrected += (predicted == labels).sum(dim=0).cpu().numpy()

        if i % print_interval == 0:
            for j, item in enumerate(corrected):
                print(f'[{i}] {attributes[j]}: {100 * item / (i+1)}')

for j, item in enumerate(corrected):
    if 100 * item / (i+1) > 70:
        print(f'[{j}] {attributes[j]}: {(100 * item / (i+1)):.1f}')

print(100 * corrected.sum() / ((i+1)*50))
