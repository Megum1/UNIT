import os
import torch
import random
import numpy as np
from torchvision import datasets, transforms

from models import resnet18, resnet34, resnet50
from backdoors import Badnet, Reflection, WaNet


# Set random seed
def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# Print configurations
def print_args(opt):
    message = ''
    message += '='*46 +' Options ' + '='*46 +'\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''

        message += '{:>25}: {:<30}\n'.format(str(k), str(v))
    message += '='*48 +' End ' + '='*47 +'\n'
    print(message)


# Some fixed parameters
EPSILON = 1e-7

_dataset_name = ['cifar10']

_mean = {
    'cifar10':  [0.4914, 0.4822, 0.4465],
}

_std = {
    'cifar10':  [0.2023, 0.1994, 0.2010],
}

_size = {
    'cifar10':  (32, 32),
}

_num = {
    'cifar10':  10,
}


def get_config(dataset):
    assert dataset in _dataset_name, _dataset_name
    config = {}
    config['mean'] = _mean[dataset]
    config['std']  = _std[dataset]
    config['size'] = _size[dataset]
    config['num_classes'] = _num[dataset]
    return config


def get_norm(dataset):
    assert dataset in _dataset_name, _dataset_name
    mean = torch.FloatTensor(_mean[dataset])
    std  = torch.FloatTensor(_std[dataset])
    normalize   = transforms.Normalize(mean, std)
    unnormalize = transforms.Normalize(- mean / std, 1 / std)
    return normalize, unnormalize


def get_transform(args, augment=False, tensor=False):
    resize_size = _size[args.dataset]
    padding = 4

    transforms_list = []
    if augment:
        transforms_list.append(transforms.Resize(resize_size))
        transforms_list.append(transforms.RandomCrop(resize_size, padding=padding))
        
        # Horizontal Flip
        transforms_list.append(transforms.RandomHorizontalFlip())
    else:
        transforms_list.append(transforms.Resize(resize_size))
    
    # To Tensor
    if not tensor:
        transforms_list.append(transforms.ToTensor())

    transform = transforms.Compose(transforms_list)
    return transform


def get_dataset(args, train=True, augment=True):
    transform = get_transform(args, augment=train & augment)
    
    if args.dataset == 'cifar10':
        dataset = datasets.CIFAR10(args.datadir, train, download=True, transform=transform)

    return dataset


def get_model(args):
    num_classes = _num[args.dataset]

    if args.network == 'resnet18':
        model = resnet18(num_classes=num_classes)
    elif args.network == 'resnet34':
        model = resnet34(num_classes=num_classes)
    elif args.network == 'resnet50':
        model = resnet50(num_classes=num_classes)
    else:
        raise NotImplementedError

    return model


def get_backdoor(attack, shape, device=None):
    if attack == 'badnet':
        backdoor = Badnet(shape, device=device)
    elif attack == 'reflection':
        backdoor = Reflection(shape, device=device)
    elif attack == 'wanet':
        backdoor = WaNet(shape, device=device)
    else:
        raise NotImplementedError
    
    return backdoor


class PoisonDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, backdoor, target):
        self.dataset = dataset
        self.backdoor = backdoor
        self.target = target

        # Remember to remove the images that are already from the target class
        self.images = []
        for img, label in self.dataset:
            if label != self.target:
                self.images.append(img)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        device = self.backdoor.device
        poison_img = self.backdoor.inject(img.unsqueeze(0).to(device)).squeeze(0)
        return poison_img, self.target
