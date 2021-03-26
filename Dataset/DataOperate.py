""""
we define the data set and data operation in this file.
"""
import glob
import os
from torch.utils.data import Dataset
import numpy as np
import random
import torch
import SimpleITK as sitk
def get_data_list(data_path, ratio=0.8):
    """
    this function is create the data list and the data is set as follow:
    --data
        --name_1
            image.nii.gz
            label.nii.gz
            label.txt
        --name_2
            image.nii.gz
            label.nii.gz
            label.txt

        ...
    if u use your own data, u can rewrite this function
    """
    train_list = os.listdir(data_path)
    train_list.sort()
    train_list_all = [{'data': os.path.join(data_path, name), 'name': name}for name in train_list]
    #random.shuffle(train_list_all)
    return train_list_all


class MySet(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
    def __getitem__(self, item):
        data_dict = self.data_list[item]
        image_path = data_dict["data"] + '/image.nii.gz'
        mask_path = data_dict["data"]+'/label.nii.gz'
        label_path = data_dict["data"] + '/label.txt'
        name = data_dict["name"]
        image = sitk.ReadImage(image_path)
        image = sitk.GetArrayFromImage(image)
        image = image.astype(np.float32)

        with open(label_path, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline()  # 整行读取数据
                if not lines:
                    break
                    pass
                x = lines.split()
                m = int(x[0])
                pass
            pass
        if m ==0:
            label = torch.zeros(1, dtype=torch.long)
        elif m ==1:
            label = torch.ones(1, dtype=torch.long)
        else:
            label = torch.ones(1, dtype=torch.long)
            label = 2*label
        mask = sitk.ReadImage(mask_path)
        mask = sitk.GetArrayFromImage(mask)
        #print(data.shape)
        image = self.normalize(image)
        image = image[np.newaxis, :, :, :]
        mask = mask.astype(np.float32)
        mask = mask[np.newaxis, :, :, :]
        mask_tensor = torch.from_numpy(mask)
        image_tensor = torch.from_numpy(image)

        return image_tensor, mask_tensor,label,name

    @staticmethod
    def normalize(data):
        data = data.astype(np.float32)
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
        return data
    def __len__(self):
        return len(self.data_list)

class MySet_npy(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
    def __getitem__(self, item):
        data_dict = self.data_list[item]
        data_path = data_dict["data"] + '/image.npy'
        mask_path = data_dict["data"]+'/label.npy'
        label_path = data_dict["data"] + '/label.txt'
        name = data_dict["name"]
        data = np.load(data_path)
        data = data.astype(np.float32)

        with open(label_path, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline()  # 整行读取数据
                if not lines:
                    break
                    pass
                x = lines.split()
                m = int(x[0])
                pass
            pass

        if m ==0:
            label = torch.zeros(1, dtype=torch.long)
        elif m ==1:
            label = torch.ones(1, dtype=torch.long)
        else:
            label = torch.ones(1, dtype=torch.long)
            label = 2*label

        mask = np.load(mask_path)
        data = self.normalize(data)
        data = data[np.newaxis, :, :, :]
        mask = mask.astype(np.float32)
        mask = mask[np.newaxis, :, :, :]
        mask_tensor = torch.from_numpy(mask)
        data_tensor = torch.from_numpy(data)

        return data_tensor, mask_tensor,label, name

    @staticmethod
    def normalize(data):
        data = data.astype(np.float32)
        data = (data - np.min(data)) / (np.max(data) - np.min(data)+1e-8)
        return data

    def __len__(self):
        return len(self.data_list)