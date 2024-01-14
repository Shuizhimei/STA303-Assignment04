import os
import pathlib

import torchvision.datasets as dset
import torchvision.transforms as trn
from PIL import Image
from torch.utils.data import Dataset


def build_dataset(dataset_name, transform=None, mode="train"):
    #  path of usr
    usr_dir = os.path.expanduser('~')
    data_dir = os.path.join(usr_dir, "data")

    if dataset_name == 'imagenet':
        if transform == None:
            transform = trn.Compose([
                trn.Resize(256),
                trn.CenterCrop(224),
                trn.ToTensor(),
                trn.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
            ])

        dataset = dset.ImageFolder(data_dir + "/imagenet/val",
                                   transform)
    elif dataset_name == 'mnist':
        if transform == None:
            transform = trn.Compose([
                trn.ToTensor(),
                trn.Normalize((0.1307,), (0.3081,))
            ])
        if mode == "train":
            dataset = dset.MNIST(data_dir, train=True, download=True, transform=transform)
        elif mode == "test":
            dataset = dset.MNIST(data_dir, train=False, download=True, transform=transform)
            
    elif dataset_name == 'flowers102':
        if transform is None:
            transform = {
                'train': trn.Compose([
                    trn.RandomResizedCrop(224),
                    trn.RandomHorizontalFlip(),
                    trn.ToTensor(),
                    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
                'test': trn.Compose([
                    trn.Resize(256),
                    trn.CenterCrop(224),
                    trn.ToTensor(),
                    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
            }
        dataset = dset.Flowers102(root="/shareddata", split=mode, 
                                download=False, transform=transform[mode])    
    elif dataset_name == 'GTSRB':
        if transform is None:
            transform = {
                'train': trn.Compose([
                    
                    trn.RandomResizedCrop(224),
                    trn.RandomHorizontalFlip(),
                    trn.ToTensor(),
                    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
                'test': trn.Compose([
                    # Adjust the transformations based on your requirements
                    trn.Resize(256),
                    trn.CenterCrop(224),
                    trn.ToTensor(),
                    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
            }
        dataset = dset.GTSRB(root="/shareddata", split=mode, download=True, transform=transform[mode])
    elif dataset_name == 'DTD':
        if transform is None:
            transform = {
                'train': trn.Compose([
                    
                    trn.RandomResizedCrop(224),
                    trn.RandomHorizontalFlip(),
                    trn.ToTensor(),
                    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
                'test': trn.Compose([
                    # Adjust the transformations based on your requirements
                    trn.Resize(256),
                    trn.CenterCrop(224),
                    trn.ToTensor(),
                    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
            }
        dataset = dset.DTD(root="/shareddata", split=mode, download=False, transform=transform[mode])
    else:
        raise NotImplementedError

    return dataset