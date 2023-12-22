import os
import torch
import torchvision
import torchvision.transforms as transforms
import PIL
from torchvision.transforms import *
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.utils.data as data
import torchxrayvision as xrv
from torch.utils.data.dataset import Subset
from torchvision.datasets import CIFAR10

def _convert_image_to_rgb(image):
    return image.convert("RGB")

class Transforms:

    class CIFAR10:

        class VGG:

            train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        class ResNet:
            print('yesssssssss resnet')
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            normalize = transforms.Normalize(mean=mean, std=std)
            train = transforms.Compose([
                transforms.Resize((224, 224), interpolation=PIL.Image.NEAREST),
                transforms.ToTensor(),
                normalize,
            ])
            test = transforms.Compose([
                transforms.Resize((224, 224), interpolation=PIL.Image.NEAREST),
                transforms.ToTensor(),
                normalize,
            ])
        class ResNet_RSNA:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            normalize = transforms.Normalize(mean=mean, std=std)
            train = transforms.Compose([

                transforms.RandomRotation(degrees=15, fill=(0,0,0)),  # Random rotation with 15 degrees
                transforms.RandomVerticalFlip(p=0.3),  # Random vertical flip with 50% probability
                transforms.RandomHorizontalFlip(p=0.3),  # Random horizontal flip with 50% probability
                transforms.Resize((224, 224), interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ])

            test = transforms.Compose([
                transforms.Resize((224, 224), interpolation=PIL.Image.NEAREST),
                transforms.ToTensor(),
                normalize,
            ])  
        class densenet_Chexpert:
            mean = [0.5,]
            std = [0.5, ]
            normalize = transforms.Normalize(mean=mean, std=std)
            train = transforms.Compose([
                # transforms.Grayscale(num_output_channels=1),
                transforms.RandomRotation(degrees=15, fill=(0,0,0)),  # Random rotation with 15 degrees
                transforms.RandomVerticalFlip(p=0.3),  # Random vertical flip with 50% probability
                transforms.RandomHorizontalFlip(p=0.3),  # Random horizontal flip with 50% probability
                transforms.Resize((224, 224), interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ])

            test = transforms.Compose([
                # transforms.Grayscale(num_output_channels=1),
                transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ])  
        class CLIP:
            train = transforms.Compose([ transforms.Resize(224, interpolation=BICUBIC),
                                        transforms.CenterCrop(224),
                                        _convert_image_to_rgb,
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                                        ])
            test = transforms.Compose([ transforms.Resize(224, interpolation=BICUBIC),
                                        transforms.CenterCrop(224),
                                        _convert_image_to_rgb,
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                                        ])



    CIFAR100 = CIFAR10

class RSNADataset(Dataset):
    def __init__(self, csv_file, data_folder, split, transform=None):
        print('hiiiiiii',csv_file, data_folder,split)
        self.data = pd.read_csv(csv_file)
        self.data_folder = data_folder
        self.transform = transform
        self.split = split
        # self.data = self.data[self.data['class'] != 'No Lung Opacity / Not Normal']

        # Filter the data based on the specified split
        self.data = self.data[self.data['split'] == split]
        print(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # print('trainnnnnnnnnnnnnn',self.data.iloc[idx, 8])
        image_folder = os.path.join(self.data_folder, 'train')
        # print('imageeeeeeeeeee', image_folder, idx, self.data.iloc[0, 1])
        image_path = os.path.join(image_folder, self.data.iloc[idx, 1] + '.jpg')
        # print(image_path)
        image = Image.open(image_path).convert('L')
        target = int(self.data.iloc[idx, 7])
        # print(target)
        if self.transform:
            image = self.transform(image)
        return image, target

    
def loaders(dataset, path, batch_size, num_workers, transform_name, use_test=False,
            shuffle_train=True, dataset_selection=None):
    print('dataset_selection')
    if dataset_selection=='cifar10':
        print('yesssssssss')
        transform = getattr(getattr(Transforms, dataset), transform_name)

        train_set = RSNADataset(csv_file = '/home/santoshsanjeev/dnn-mode-connectivity/data/rsna_18/csv/final_dataset_wo_not_normal_cases.csv', data_folder = '/home/santoshsanjeev/dnn-mode-connectivity/data/rsna_18/', split='train', transform=transform.train)
        val_set = RSNADataset(csv_file = '/home/santoshsanjeev/dnn-mode-connectivity/data/rsna_18/csv/final_dataset_wo_not_normal_cases.csv', data_folder = '/home/santoshsanjeev/dnn-mode-connectivity/data/rsna_18/', split='val', transform=transform.test)
        test_set = RSNADataset(csv_file = '/home/santoshsanjeev/dnn-mode-connectivity/data/rsna_18/csv/final_dataset_wo_not_normal_cases.csv', data_folder = '/home/santoshsanjeev/dnn-mode-connectivity/data/rsna_18/', split='test', transform=transform.test)

        # train_loader = data.DataLoader(dataset=train_set, batch_size=128, shuffle=True, num_workers=16)
        # val_loader = data.DataLoader(dataset=val_set, batch_size=128, shuffle=False, num_workers=16)
        # test_loader = data.DataLoader(dataset=test_set, batch_size=128, shuffle=False, num_workers=16)
        # return {'train':train_loader, 'test':test_loader},2
    elif dataset_selection=='rsna':
        print('cifarrrrrrrrrrrrrrrrrrrrr')
        ds = getattr(torchvision.datasets, dataset)
        path = os.path.join(path, dataset.lower())
        transform = getattr(getattr(Transforms, dataset), transform_name)
        train_set = ds(path, train=True, download=True, transform=transform.train)

        if use_test:
            print('You are going to run models on the test set. Are you sure?')
            test_set = ds(path, train=False, download=True, transform=transform.test)
            classnames = test_set.classes
        else:
            print("Using train (45000) + validation (5000)")
            train_set.data = train_set.data[:-5000]
            train_set.targets = train_set.targets[:-5000]

            test_set = ds(path, train=True, download=True, transform=transform.test)
            test_set.train = False
            test_set.data = test_set.data[-5000:]
            test_set.targets = test_set.targets[-5000:]
            # delattr(test_set, 'data')
            # delattr(test_set, 'targets')
    elif dataset_selection=='clip_cifar10':
        import numpy as np
        idxs = np.load('/share/nvmedata/santosh/model_soups/models_model_soups/cifar10_model_soups_models/cifar1098_idxs.npy').astype('int')
        ds = getattr(torchvision.datasets, dataset)
        path = os.path.join(path, dataset.lower())
        transform = getattr(getattr(Transforms, dataset), transform_name)

        indices_val = []
        indices_train = []
        for i in range(len(idxs)):
            if idxs[i]:
                indices_val.append(i)
            else:
                indices_train.append(i)
        
        total = ds(path, train=True, download=True, transform=transform.train)
        val = Subset(total, indices_val)
        train = Subset(total, indices_train)
        print(len(val), len(train))

        test = ds(path, train=False, download=True, transform=transform.test)
        return {
                'train': torch.utils.data.DataLoader(
                    train,
                    batch_size=batch_size,
                    shuffle=shuffle_train,
                    num_workers=num_workers,
                    pin_memory=True
                ),
                'test': torch.utils.data.DataLoader(
                    test,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True
                ),
                'val': torch.utils.data.DataLoader(
                    val,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True
                ),
        }, 10, test.classes
    return {
        'train': torch.utils.data.DataLoader(
            train,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=True
        ),
        'test': torch.utils.data.DataLoader(
            test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )}, max(train_set.targets) + 1, test_set.classes
