from scipy import io
import os
import math
import numpy as np

import torch

pascal3d_root = 'C:\\Users\\cglab\\pytorch-clickhere-cnn-f8988287a9e760a6825f2c91cdf8539fc12a4c6e\\PASCAL3D+_release1.1'
annotation = 'Annotations'
pascal = 'car_imagenet'
mat = 'n02814533_3537.mat'

path = os.path.join(pascal3d_root, os.path.join(annotation, pascal))
mat_file = io.loadmat(os.path.join(path, mat))

anchor_list = []
mask = []

for obj in mat_file['record']['objects'][0][0][0]:
    if not obj['viewpoint']:
        continue
    elif 'distance' not in obj['viewpoint'].dtype.names:
        continue
    elif obj['viewpoint']['distance'][0][0][0][0] == 0:
        continue
    
    cad_index = obj['cad_index'][0][0] - 1
    bbox = obj['bbox'][0]
    anchors = obj['anchors']

    viewpoint = obj['viewpoint']
    azimuth = 360 - viewpoint['azimuth'][0][0][0][0]
    # if azimuth > 180:
    #     azimuth -= 360
    elevation = viewpoint['elevation'][0][0][0][0] 
    distance = viewpoint['distance'][0][0][0][0]
    focal = viewpoint['focal'][0][0][0][0]
    theta = viewpoint['theta'][0][0][0][0] 
    principal = np.array([viewpoint['px'][0][0][0][0],
                            viewpoint['py'][0][0][0][0]])
    viewport = viewpoint['viewport'][0][0][0][0]

    azi = 1 if azimuth <= 180 else 0
    elev = 1 if elevation >= 5.08 else 0
    dist = 1 if distance >= 5.08 else 0


    print(f"Anchor : {anchors}")
    print(f"Azimuth : {azimuth} || Elevation : {elevation} || Distance : {distance}")
    print(f"focal : {focal} || Theta : {theta} || Principal : {principal} || Viewport : {viewport}")


    anchors = obj['anchors'][0][0]
    for name in anchors.dtype.names:
        anchor = anchors[name]
        if anchor['status'] != 1: # Occluded
            print(f'(x,y)=({0},{0})')
            anchor_list.append([0,0])
            mask.append([0,0])
            continue
        x, y = anchor['location'][0][0][0]
        print(f'(x,y)=({x},{y})')
        anchor_list.append([x,y])
        mask.append([1,1])




'''
# print(np.asarray(anchor_list))
print(torch.Tensor(anchor_list))
print(torch.Tensor(mask))
print(torch.Tensor(anchor_list) * torch.Tensor(mask))
# print(torch.Tensor(mask).view(-1,1))
# print(torch.Tensor(anchor_list) * torch.Tensor(mask).view(-1,1))

mk = torch.Tensor(mask)
lm = torch.Tensor(anchor_list)
print(mk.size())
print(lm.size())

mk = torch.stack((mk,mk), 0)
lm = torch.stack((lm,lm), 0)
print(mk.size())
print(lm.size())

lm = lm.view(lm.size(0),-1)
mk = mk.view(mk.size(0),-1)
print(mk.size())
print(lm.size())
print(lm)
print(mk)
print(lm*mk)
# print(lm.view(lm.size(0),-1))
# print(mk.view(mk.size(0),-1))
# print(lm.view(lm.size(0),-1)*mk.view(mk.size(0),-1))
'''