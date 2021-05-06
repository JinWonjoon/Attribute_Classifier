import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from PIL import Image
import os
from scipy import io

class CarAttribute(Dataset):
    def __init__(self, root_path, mode='train', transform = None):
        super().__init__()

        self.mode = mode
        list_path = os.path.join("Image_sets", "car_imagenet_" + self.mode + ".txt")
        img_path = os.path.join("Images", "car_imagenet")
        label_path = os.path.join("Annotations", "car_imagenet")
        
        self.img_list, self.label_list = self.get_list(root_path, list_path, img_path, label_path)

        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])
    
    def get_list(self, root, list_path, img_path, label_path):
        list_path = os.path.join(root, list_path)
        img_path = os.path.join(root, img_path)
        label_path = os.path.join(root, label_path)

        with open(list_path, 'r') as f:
            lines = [line.strip() for line in f.readlines()]

        img_list = [os.path.join(img_path, name + ".jpeg") for name in lines]
        label_list = [os.path.join(label_path, name + ".mat") for name in lines]

        return img_list, label_list
    
    def __getitem__(self, index):
        img = Image.open(self.img_list[index % len(self.img_list)]).convert('RGB')
        img_size = np.shape(img)[:2]
        img = img.resize((500,375), Image.BICUBIC)
        img = self.transform(img)

        mat_file = io.loadmat(self.label_list[index % len(self.label_list)])

        anchor_list = []
        mask = []
        azi, elev, dist = None, None, None
        for obj in mat_file['record']['objects'][0][0][0]:
            if not obj['viewpoint']:
                continue
            elif 'distance' not in obj['viewpoint'].dtype.names:
                continue
            elif obj['viewpoint']['distance'][0][0][0][0] == 0:
                continue
            
            anchors = obj['anchors'][0][0]
            for name in anchors.dtype.names:
                if len(mask) >= 12: # Ignore other cars
                    break
                anchor = anchors[name]
                if anchor['status'] != 1: # Occluded
                    anchor_list.append([0,0]) # How to handle occluded landmarks
                    mask.append([0,0])
                    continue
                x, y = anchor['location'][0][0][0]
                x = int(x * 500/img_size[1])
                y = int(y * 375/img_size[0])
                anchor_list.append([x,y])
                mask.append([1,1])

            viewpoint = obj['viewpoint']
            azimuth = 360 - viewpoint['azimuth'][0][0][0][0]
            elevation = viewpoint['elevation'][0][0][0][0]
            distance = viewpoint['distance'][0][0][0][0]

            azi = 1 if azimuth <= 180 else 0
            elev = 1 if elevation >= 5.08 else 0
            dist = 1 if distance >= 5.08 else 0

        # Deal with else-case
        if azi == None or elev == None or dist == None:
            azi, elev, dist = 0.5, 0.5, 0.5
        if len(anchor_list) == 0:
            anchor_list = np.zeros((12,2))
            mask = np.zeros((12,2))
        
        attributes = torch.Tensor([azi, elev, dist])
        landmark = torch.Tensor(anchor_list)
        mask = torch.Tensor(mask)

        return (img, attributes, landmark, mask)
    
    def __len__(self):
        return len(self.img_list)



class ImageDataset(Dataset):
    def __init__(self, img_path, label_path, mode='train', transform = None):
        super().__init__()
        self.img_list = self.get_images(img_path)
        self.attributes, self.label_list = self.get_labels(label_path)
        self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                ]
            )
        if mode == 'train':
            self.img_list = self.img_list[:150000]
            self.label_list = self.label_list[:150000]
        elif mode == 'test':
            self.img_list = self.img_list[150000:]
            self.label_list = self.label_list[150000:]
            

    def get_images(self, root):
        img_list = [os.path.join(root, name) for name in os.listdir(root)]

        return img_list

    def get_labels(self, root):
        with open(root, 'r') as f:
            lines = f.readlines()

        attributes = lines[1].split()

        data_label = []
        for i, line in enumerate(lines[2:]):
            split = line.split()
            data = [int(item.replace('-1', '0')) for item in split[1:]]
            data_label.append(data)

        return attributes, data_label


    def __getitem__(self, index):
        img = Image.open(self.img_list[index % len(self.img_list)]).convert('RGB')
        img = self.transform(img)
        label = torch.Tensor(self.label_list[index % len(self.label_list)])

        return img, label

    def __len__(self):
        return len(self.img_list)

if __name__ == '__main__':
    pascal3d_root = 'C:\\Users\\cglab\\pytorch-clickhere-cnn-f8988287a9e760a6825f2c91cdf8539fc12a4c6e\\PASCAL3D+_release1.1'
    train_dataset = CarAttribute(pascal3d_root, 'train', transform)



