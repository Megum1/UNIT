import os
import sys
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from utils import seed_torch, print_args, get_norm, get_dataset, get_backdoor, get_config, PoisonDataset

import warnings
warnings.filterwarnings("ignore")


# Visualize the poisoned samples
def visualize(image_batch, attack, backdoor):
    device = backdoor.device
    image_batch = image_batch.to(device)
    # Apply the backdoor
    poison_image_batch = backdoor.inject(image_batch)
    # Calculate the difference
    diff = torch.abs(poison_image_batch - image_batch)
    if attack == 'wanet':
        diff *= 3.
    diff = torch.clamp(diff, 0., 1.)
    # Concatenate the images
    save_images = torch.cat([image_batch, poison_image_batch, diff], dim=0)
    # Save the images
    save_image(save_images, f'demo_{attack}.png', nrow=image_batch.size(0))


# Evaluate the model on the (benign/poisoned) test set
def eval_acc(model, loader, preprocess):
    model.eval()
    n_sample = 0
    n_correct = 0
    with torch.no_grad():
        for _, (x_batch, y_batch) in tqdm(enumerate(loader), total=len(loader)):
            x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

            output = model(preprocess(x_batch))
            pred = output.max(dim=1)[1]

            n_sample  += x_batch.size(0)
            n_correct += (pred == y_batch).sum().item()

    acc = n_correct / n_sample
    return acc


# Main function: evaluate the model performance
def main(args):
    # Load the model
    model_filepath = f'ckpt/{args.dataset}_{args.network}_{args.attack}.pt'
    model = torch.load(model_filepath, map_location='cpu').cuda()
    model.eval()

    # Get the normalization function
    preprocess, _ = get_norm(args.dataset)

    # Get the image shape
    shape = get_config(args.dataset)['size']

    # Get the backdoor
    backdoor = get_backdoor(args.attack, shape, torch.device('cuda'))

    # Benign test set
    test_set = get_dataset(args, train=False)
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size)

    # Visualize the poisoned samples
    image_batch = next(iter(test_loader))[0][:16]
    visualize(image_batch, args.attack, backdoor)

    # Poisoned test set
    poison_set = PoisonDataset(test_set, backdoor, args.target)
    poison_loader = DataLoader(dataset=poison_set, batch_size=args.batch_size)

    # Measure the clean accuracy
    acc = eval_acc(model, test_loader, preprocess)
    # Measure the ASR
    asr = eval_acc(model, poison_loader, preprocess)

    print(f'Benign accuarcy: {acc*100:.2f}%, ASR: {asr*100:.2f}%')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate the model performance.')
    parser.add_argument('--datadir',     default='./data',    help='root directory of data')
    parser.add_argument('--dataset',     default='cifar10',   help='dataset')
    parser.add_argument('--network',     default='resnet18',  help='network structure')
    parser.add_argument('--attack',      default='badnet',    help='attack type')

    parser.add_argument('--seed',        type=int, default=1024, help='seed index')
    parser.add_argument('--batch_size',  type=int, default=128,  help='attack size')
    parser.add_argument('--target',      type=int, default=0,    help='target label')

    args = parser.parse_args()

    # Print the arguments
    print_args(args)

    # Set the random seed
    seed_torch(args.seed)

    # Main function
    main(args)
