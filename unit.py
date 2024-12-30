import os
import sys
import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils import seed_torch, print_args, EPSILON, get_norm, get_dataset, get_backdoor, get_config, PoisonDataset
from evaluate import eval_acc

import warnings
warnings.filterwarnings("ignore")


def test(args):
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

    # Load the dataset
    test_set = get_dataset(args, train=False)
    poison_set = PoisonDataset(test_set, backdoor, args.target)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)
    poison_loader = DataLoader(poison_set, batch_size=args.batch_size)

    acc = eval_acc(model, test_loader, preprocess)
    asr = eval_acc(model, poison_loader, preprocess)
    print(f'Benign accuracy: {acc*100:.2f}%, ASR: {asr*100:.2f}%')


###############################################################################
# Backdoor mitigation through UNIT
###############################################################################
def prepare_defense_samples(args, train=True, augment=False):
    # Load the dataset
    init_dataset = get_dataset(args, train=train, augment=augment)

    # Collect partial data for defense
    n_sample = int(len(init_dataset) * args.data_rate)

    # Averagely sample data from each class
    num_classes = get_config(args.dataset)['num_classes']
    n_sample_per_class = n_sample // num_classes

    # Collect data
    cnt_per_class = [0 for _ in range(num_classes)]
    x_collect, y_collect = {}, {}
    for x, y in init_dataset:
        if cnt_per_class[y] < n_sample_per_class:
            if y not in x_collect.keys():
                x_collect[y] = []
                y_collect[y] = []
            x_collect[y].append(x)
            y_collect[y].append(y)
            cnt_per_class[y] += 1
        if np.sum(cnt_per_class) >= n_sample:
            break

    # 80% for training, 20% for validation
    x_train, y_train = [], []
    x_valid, y_valid = [], []
    for y in range(num_classes):
        n_train = int(len(x_collect[y]) * 0.8)
        x_train.extend(x_collect[y][:n_train])
        y_train.extend(y_collect[y][:n_train])
        x_valid.extend(x_collect[y][n_train:])
        y_valid.extend(y_collect[y][n_train:])

    x_train, y_train = torch.stack(x_train), torch.tensor(y_train)
    x_valid, y_valid = torch.stack(x_valid), torch.tensor(y_valid)

    return x_train, y_train, x_valid, y_valid


# Reconstruct the model
def activation_clip(x, clip_bound):
    # x: (N, C, H, W) or (N, C)
    # Clip the activation at the channel level
    # clip_bound: (C, H, W) or (C,)
    max_value = clip_bound.unsqueeze(0)
    output = torch.clamp(x, max=max_value)
    return output


class ResNet18_unit:
    def __init__(self, model):
        self.model = model
        self.collect_bounds = None
    
    def get_activation(self, x):
        # x: (N, C, H, W)
        # Collect activation for each activation function
        acti_dict = {}

        # Pre layer
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = F.relu(x)
        acti_dict['pre_layer'] = x

        # Traverse all the layers
        for layer_id in range(1, 5):
            cur_layer = getattr(self.model, f'layer{layer_id}')
            for block_id in range(len(cur_layer)):
                block = cur_layer[block_id]
                out = block.conv1(x)
                out = block.bn1(out)
                out = F.relu(out)
                acti_dict[f'layer{layer_id}_block{block_id}_0'] = out

                out = block.conv2(out)
                out = block.bn2(out)
                out += block.shortcut(x)
                x = F.relu(out)
                acti_dict[f'layer{layer_id}_block{block_id}_1'] = x
        
        # Post layer
        out = F.avg_pool2d(x, 4)
        out = out.view(out.size(0), -1)
        out = self.model.linear(out)

        return out, acti_dict

    def forward(self, x, clip_bounds):
        # Pre layer
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = F.relu(x)
        x = activation_clip(x, clip_bounds['pre_layer'])

        # Traverse all the layers
        for layer_id in range(1, 5):
            cur_layer = getattr(self.model, f'layer{layer_id}')
            for block_id in range(len(cur_layer)):
                block = cur_layer[block_id]
                out = block.conv1(x)
                out = block.bn1(out)
                out = F.relu(out)
                out = activation_clip(out, clip_bounds[f'layer{layer_id}_block{block_id}_0'])

                out = block.conv2(out)
                out = block.bn2(out)
                out += block.shortcut(x)
                x = F.relu(out)
                x = activation_clip(x, clip_bounds[f'layer{layer_id}_block{block_id}_1'])
        
        # Post layer
        out = F.avg_pool2d(x, 4)
        out = out.view(out.size(0), -1)
        out = self.model.linear(out)

        return out

    def eval(self):
        self.model.eval()

    def __call__(self, x):
        if self.collect_bounds is None:
            raise ValueError('Collect bounds first!')
        return self.forward(x, self.collect_bounds)


def validate_acc(model, x, y, preprocess):
    model.eval()
    with torch.no_grad():
        output = model(preprocess(x))
        pred = output.max(dim=1)[1]
        acc = (pred == y).float().mean().item()
    return acc


def unit(args, verbose=True):
    # Load the model
    model_filepath = f'ckpt/{args.dataset}_{args.network}_{args.attack}.pt'
    model = torch.load(model_filepath, map_location='cpu').cuda()
    model.eval()

    # Customize a UNIT model
    model_clip = ResNet18_unit(model)

    # Get the normalization function
    preprocess, _ = get_norm(args.dataset)

    # Get the image shape
    shape = get_config(args.dataset)['size']

    # Get the backdoor
    backdoor = get_backdoor(args.attack, shape, torch.device('cuda'))

    # Load the dataset
    test_set = get_dataset(args, train=False)
    poison_set = PoisonDataset(test_set, backdoor, args.target)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)
    poison_loader = DataLoader(poison_set, batch_size=args.batch_size)

    # Prepare data for defense
    x_train, y_train, x_valid, y_valid = prepare_defense_samples(args, train=True, augment=True)
    x_train, y_train = x_train.cuda(), y_train.cuda()
    x_valid, y_valid = x_valid.cuda(), y_valid.cuda()
    print(f'Data for clipping --- Train: {x_train.size(0)}, Valid: {x_valid.size(0)}')

    time_start = time.time()

    # Collect activation
    output, acti_dict = model_clip.get_activation(preprocess(x_train))

    # Optimize the clip bound
    params = {}
    for name in acti_dict.keys():
        cur_acti = acti_dict[name].detach().clone().cuda()
        # Average over the batch dimension
        cur_mean = cur_acti.mean(dim=0)
        cur_std = cur_acti.std(dim=0)
        param_init = cur_mean + 4 * cur_std
        params[name] = param_init
        params[name].requires_grad = True

    # Optimize the clip bound
    init_lr = args.lr
    optimizer = torch.optim.Adam(params.values(), lr=init_lr, betas=(0.5, 0.9))
    criterion = torch.nn.CrossEntropyLoss()

    # Initialize best bound
    best_reg = 1 / EPSILON
    best_bounds = None

    # Threshold for accuracy
    acc_init = validate_acc(model, x_valid, y_valid, preprocess)
    if verbose:
        print(f'Initial accuracy: {acc_init*100:.2f}%')
    acc_threshold = acc_init - args.acc_degrade

    # Initial cost for bound
    init_cost = 1e-3
    cost = init_cost
    cost_multiplier_up = 2
    cost_multiplier_down = cost_multiplier_up ** 1.5

    # Counters for adjusting balance cost
    cost_set_counter = 0
    cost_up_counter = 0
    cost_down_counter = 0

    # Patience
    patience = 5

    # Total optimization steps
    steps = args.n_steps

    for step in range(steps):
        optimizer.zero_grad()

        # Clip the bound
        clip_bounds = {}
        for name in params.keys():
            clip_bounds[name] = torch.clamp(params[name], min=0)

        # Forward pass
        output = model_clip.forward(preprocess(x_train), clip_bounds)
        ce_loss = criterion(output, y_train)

        # Regularization
        reg_loss = 0
        for name in clip_bounds.keys():
            reg_loss += clip_bounds[name].mean()

        # Total loss
        loss = ce_loss + cost * reg_loss

        loss.backward()
        optimizer.step()

        eval_ce_loss = ce_loss.item()
        eval_reg_loss = reg_loss.item()

        # Evaluate the accuracy
        temp_clip_bounds = {}
        for name in params.keys():
            temp_clip_bounds[name] = torch.clamp(params[name].detach().clone().cuda(), min=0)
        model_clip.collect_bounds = temp_clip_bounds
        acc = validate_acc(model_clip, x_valid, y_valid, preprocess)

        # Print log
        if (step + 1) % 10 == 0 and verbose:
            print(f'Step [{step+1}/{steps}], CE Loss: {eval_ce_loss:.4f}, Reg Loss: {eval_reg_loss:.4f}, Acc: {acc*100:.2f}%')

        if acc >= acc_threshold and eval_reg_loss < best_reg:
            best_reg = eval_reg_loss
            best_bounds = clip_bounds
            if verbose:
                print(f'Update best bound at step {step}')

        # Adjust the cost
        if cost < init_cost and acc >= acc_threshold:
            cost_set_counter += 1
            if cost_set_counter >= patience:
                cost = init_cost
                cost_up_counter = 0
                cost_down_counter = 0
                if verbose:
                    print(f'Cost reset to {cost:.4f}')
        else:
            cost_set_counter = 0
        
        if acc >= acc_threshold:
            cost_up_counter += 1
            cost_down_counter = 0
        else:
            cost_up_counter = 0
            cost_down_counter += 1
        
        if cost_up_counter >= patience:
            cost_up_counter = 0
            cost *= cost_multiplier_up
            if verbose:
                print(f'Cost up to {cost:.4f}')

        if cost_down_counter >= patience:
            cost_down_counter = 0
            cost /= cost_multiplier_down
            if verbose:
                print(f'Cost down to {cost:.4f}')
    
    # If the best bound is not found
    if best_bounds is None:
        best_bounds = clip_bounds
        if verbose:
            print(f'Best bound not found, use the last one')

    # Save the clip bound
    print('=' * 80)
    clip_bounds = {}
    for name in params.keys():
        clip_bounds[name] = best_bounds[name].detach().cuda()
    model_clip.collect_bounds = clip_bounds

    time_end = time.time()

    # Test on holdout data
    init_acc = eval_acc(model, test_loader, preprocess)
    init_asr = eval_acc(model, poison_loader, preprocess)
    print(f'Initial --- Accuracy: {init_acc*100:.2f}%, ASR: {init_asr*100:.2f}%')
    acc = eval_acc(model_clip, test_loader, preprocess)
    asr = eval_acc(model_clip, poison_loader, preprocess)
    print(f'After UNIT --- Accuracy: {acc*100:.2f}%, ASR: {asr*100:.2f}%')
    print(f'Running time: {time_end - time_start:.2f}s')


###############################################################################
# Main function
###############################################################################
def main():
    if args.phase == 'test':
        test(args)
    elif args.phase == 'unit':
        unit(args)
    else:
        print('Option [{}] is not supported!'.format(args.phase))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UNIT defense for backdoor mitigation')

    parser.add_argument('--datadir', default='./data',    help='root directory of data')
    parser.add_argument('--phase',   default='unit',      help='phase of framework')
    parser.add_argument('--dataset', default='cifar10',   help='dataset')
    parser.add_argument('--network', default='resnet18',  help='network structure')
    parser.add_argument('--attack',  default='badnet',    help='attack type')

    parser.add_argument('--seed',       type=int, default=1024, help='seed index')
    parser.add_argument('--batch_size', type=int, default=128,  help='attack size')
    parser.add_argument('--target',     type=int, default=0,    help='target label')

    # UNIT parameters
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--n_steps', type=int, default=300, help='number of steps')
    parser.add_argument('--data_rate', type=float, default=0.05, help='ratio of training data for defense')
    parser.add_argument('--acc_degrade', type=float, default=0.03, help='tolerance of accuracy degradation')

    args = parser.parse_args()

    # Print the arguments
    print_args(args)

    # Set the random seed
    seed_torch(args.seed)

    # Run the main function
    main()
