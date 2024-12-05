import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision.datasets as datasets
import os

def get_cifar10_transforms():

    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    return transform_train, transform_test

def get_alternative_cifar100_transforms():

    # taken from https://jovian.com/tessdja/resnet-practice-cifar100-resnet
    stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4, padding_mode='reflect'), 
                         transforms.RandomHorizontalFlip(), 
                         transforms.ToTensor(), 
                         transforms.Normalize(*stats,inplace=True)])
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats)])
    
    return transform_train, transform_test


def get_cifar100_transforms():
        transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
        return transform, transform


def get_imagenet_transforms():
    pass


def get_tiny_imagenet_dataloaders(batch_size):
    # from https://github.com/tjmoon0104/pytorch-tiny-imagenet/tree/master
    data_transforms = {
    "train": transforms.Compose(
        [
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ]
    ),
    "test": transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ]
    ),
    }


    data_dir = "~/datasets/tiny-imagenet-200/"
    num_workers = {"train": 1, "val": 1, "test": 1}

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ["train", "val", "test"]
    }
    dataloaders = {
        x: data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers[x])
        for x in ["train", "val", "test"]
    }
    return dataloaders
