"""
Code for "i-RevNet: Deep Invertible Networks"
https://openreview.net/pdf?id=HJsjkMb0Z
ICLR 2018
"""

import json
import numpy as np
import os

from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import torch
from torch.autograd import Variable
import torch.nn as nn

from torch.nn import Parameter

from src.metrics import scores_per_class
from src.viz_utils import line_plot, scatter_plot, images_plot, line_plot_per_dim, line_plot_mean

def save_model(model, optimizer, train_cm, val_cm, save_dict, save_dir, **kwargs):

    state = save_dict.copy()

    try:
        state['model'] = model.module
        state_dict = model.module.state_dict()

    except AttributeError:
        state['model'] = model
        state_dict = model.state_dict()

    state['model-statedict'] = state_dict
    state['optimizer-statedict'] = optimizer.state_dict()

    for k, value in kwargs.items():
        state[k] = value

    np.save(os.path.join(save_dir, "train-confusion-matrix.npy"), train_cm)
    np.save(os.path.join(save_dir, "val-confusion-matrix.npy"), val_cm)

    torch.save(state, os.path.join(save_dir, 'model.t7'))

def get_init_batch(dataloader, batch_size):
    """
    gets a batch to use for init
    """
    batches = []
    for i, x in enumerate(dataloader):
        
        _, tiles, *_ = x
        batches.append(tiles)
        if i == (batch_size - 1):
            break
    
    batch = torch.cat(batches).float()

    return batch

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def bits_per_dim(logpx, inputs):
    return -logpx / float(np.log(2.) * np.prod(inputs.shape[1:])) + 8.

def split(x):
    n = int(x.size(1)/2)
    x1 = x[:, :n, :, :].contiguous()
    x2 = x[:, n:, :, :].contiguous()
    return x1, x2

def merge(x1, x2):
    return torch.cat((x1, x2), 1)

class injective_pad(nn.Module):
    def __init__(self, pad_size):
        super(injective_pad, self).__init__()
        self.pad_size = pad_size
        self.pad = nn.ZeroPad2d((0, 0, 0, pad_size))

    def forward(self, x):
        x = x.permute(0, 2, 1, 3)
        x = self.pad(x)
        return x.permute(0, 2, 1, 3)

    def inverse(self, x):
        return x[:, :x.size(1) - self.pad_size, :, :]


class Split(nn.Module):
    def __init__(self):
        super(Split, self).__init__()

    def forward(self, x):
        n = int(x.size(1) / 2)
        x1 = x[:, :n, :, :].contiguous()
        x2 = x[:, n:, :, :].contiguous()
        return x1, x2

    def inverse(self, x1, x2):
        return torch.cat((x1, x2), 1)

class squeeze(nn.Module):
    def __init__(self, block_size):
        super(squeeze, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def inverse(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, d_height, d_width, d_depth) = output.size()
        s_depth = int(d_depth / self.block_size_sq)
        s_width = int(d_width * self.block_size)
        s_height = int(d_height * self.block_size)
        t_1 = output.contiguous().view(batch_size, d_height, d_width, self.block_size_sq, s_depth)
        spl = t_1.split(self.block_size, 3)
        stack = [t_t.contiguous().view(batch_size, d_height, s_width, s_depth) for t_t in spl]
        output = torch.stack(stack, 0).transpose(0, 1).permute(0, 2, 1, 3, 4).contiguous().view(batch_size, s_height, s_width, s_depth)
        output = output.permute(0, 3, 1, 2)
        return output.contiguous()

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, s_height, s_width, s_depth) = output.size()
        d_depth = s_depth * self.block_size_sq
        d_height = int(s_height / self.block_size)
        t_1 = output.split(self.block_size, 2)
        stack = [t_t.contiguous().view(batch_size, d_height, d_depth) for t_t in t_1]
        output = torch.stack(stack, 1)
        output = output.permute(0, 2, 1, 3)
        output = output.permute(0, 3, 1, 2)
        return output.contiguous()


class ListModule(object):
    def __init__(self, module, prefix, *args):
        self.module = module
        self.prefix = prefix
        self.num_module = 0
        for new_module in args:
            self.append(new_module)

    def append(self, new_module):
        if not isinstance(new_module, nn.Module):
            raise ValueError('Not a Module')
        else:
            self.module.add_module(self.prefix + str(self.num_module), new_module)
            self.num_module += 1

    def __len__(self):
        return self.num_module

    def __getitem__(self, i):
        if i < 0 or i >= self.num_module:
            raise IndexError('Out of bound')
        return getattr(self.module, self.prefix + str(i))


def get_all_params(var, all_params):
    if isinstance(var, Parameter):
        all_params[id(var)] = var.nelement()
    elif hasattr(var, "creator") and var.creator is not None:
        if var.creator.previous_functions is not None:
            for j in var.creator.previous_functions:
                get_all_params(j[0], all_params)
    elif hasattr(var, "previous_functions"):
        for j in var.previous_functions:
            get_all_params(j[0], all_params)


class Layer(nn.Module):
    def __init__(self):
        super(Layer, self).__init__()

    def forward_(self, x, objective, z_list, labels=None):
        raise NotImplementedError

    def reverse_(self, y, objective, labels=None):
        raise NotImplementedError


class ActNorm(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super(ActNorm, self).__init__()
        self.eps = eps
        self.num_channels = num_channels
        self._log_scale = Parameter(torch.Tensor(num_channels))
        self._shift = Parameter(torch.Tensor(num_channels))
        self._init = False

    def log_scale(self):
        return self._log_scale[None, :]

    def shift(self):
        return self._shift[None, :]

    def forward(self, x):
        if not self._init:
            with torch.no_grad():
                # initialize params to input stats
                assert self.num_channels == x.size(1)
                mean = torch.transpose(x, 0, 1).contiguous().view(self.num_channels, -1).mean(dim=1)
                zero_mean = x - mean[None, :]
                var = torch.transpose(zero_mean ** 2, 0, 1).contiguous().view(self.num_channels, -1).mean(dim=1)
                std = (var + self.eps) ** .5
                log_scale = torch.log(1. / std)
                self._shift.data = - mean * torch.exp(log_scale)
                self._log_scale.data = log_scale
                self._init = True
        log_scale = self.log_scale()
        logdet = log_scale.sum() 
        return x * torch.exp(log_scale) + self.shift(), logdet

    def inverse(self, x):
        return (x - self.shift()) * torch.exp(-self.log_scale())


class ActNorm2D(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super(ActNorm2D, self).__init__()
        self.eps = eps
        self.num_channels = num_channels
        self._log_scale = Parameter(torch.Tensor(num_channels))
        self._shift = Parameter(torch.Tensor(num_channels))
        self._init = False

    def log_scale(self):
        return self._log_scale[None, :, None, None]

    def shift(self):
        return self._shift[None, :, None, None]

    def forward(self, x):
        if not self._init:
            with torch.no_grad():
                # initialize params to input stats
                assert self.num_channels == x.size(1)
                mean = torch.transpose(x, 0, 1).contiguous().view(self.num_channels, -1).mean(dim=1)
                zero_mean = x - mean[None, :, None, None]
                var = torch.transpose(zero_mean ** 2, 0, 1).contiguous().view(self.num_channels, -1).mean(dim=1)
                std = (var + self.eps) ** .5
                log_scale = torch.log(1. / std)
                self._shift.data = - mean * torch.exp(log_scale)
                self._log_scale.data = log_scale
                self._init = True
        log_scale = self.log_scale()
        logdet = log_scale.sum() * x.size(2) * x.size(3)
        return x * torch.exp(log_scale) + self.shift(), logdet

    def inverse(self, x):
        return (x - self.shift()) * torch.exp(-self.log_scale())


class MaxMinGroup(nn.Module):
    def __init__(self, group_size, axis=-1):
        super(MaxMinGroup, self).__init__()
        self.group_size = group_size
        self.axis = axis

    def forward(self, x):
        maxes = maxout_by_group(x, self.group_size, self.axis)
        mins = minout_by_group(x, self.group_size, self.axis)
        maxmin = torch.cat((maxes, mins), dim=1)
        return maxmin

    def extra_repr(self):
        return 'group_size: {}'.format(self.group_size)
    
def process_maxmin_groupsize(x, group_size, axis=-1):
    size = list(x.size())
    num_channels = size[axis]

    if num_channels % group_size:
        raise ValueError('number of features({}) is not a '
                         'multiple of group_size({})'.format(num_channels, num_channels))
    size[axis] = -1
    if axis == -1:
        size += [group_size]
    else:
        size.insert(axis+1, group_size)
    return size


def maxout_by_group(x, group_size, axis=-1):
    size = process_maxmin_groupsize(x, group_size, axis)
    sort_dim = axis if axis == -1 else axis + 1
    return torch.max(x.view(*size), sort_dim)[0]


def minout_by_group(x, group_size, axis=-1):
    size = process_maxmin_groupsize(x, group_size, axis)
    sort_dim = axis if axis == -1 else axis + 1
    return torch.min(x.view(*size), sort_dim)[0]

def batch_class_weights(labels, nb_classes):
    classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes, labels)
    
    class_weights = np.zeros(nb_classes)

    for c, w in zip(classes, weights):
        class_weights[c] = w

    return class_weights    

def train(model, optimizer, epoch, lr, trainloader, viz, train_log, class_weights, use_cuda=False, classification_weight=1, nb_classes=8):

    model.train()

    params = sum([np.prod(p.size()) for p in model.parameters()])
    print('|  Number of Trainable Parameters: ' + str(params))
    print('\n=> Training Epoch #%d, LR=%.4f' % (epoch, lr))
    
    objective = 0
    conf_matrix = np.zeros((nb_classes, nb_classes))
    
    if use_cuda:
        class_weights = class_weights.cuda()
    
    superv_criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    for batch_idx, (_, inputs, *_, labels) in enumerate(trainloader):
        
        cur_iter = (epoch - 1) * len(trainloader) + batch_idx

        # if first epochs use warmup
        if epoch < 10:
            this_lr = lr * float(cur_iter) / (10 * len(trainloader))
            update_lr(optimizer, this_lr)

        optimizer.zero_grad()
        
        if use_cuda:
            inputs = inputs.cuda() # GPU settings
            labels = labels.cuda()
        
        logits, _, logpz, trace = model(inputs)  # Forward Propagation

        # compute loss
        logpx = logpz + trace

        # apply on all the tiles
        loss = bits_per_dim(logpx, inputs).mean()

        mean_entropy = superv_criterion(logits, labels.long())

        loss += classification_weight * mean_entropy

        objective -= loss.cpu().sum().item()

        loss.backward()  # Backward Propagation
        optimizer.step()  # Optimizer update

        mean_trace = trace.mean().item()
        mean_logpz = logpz.mean().item()

        _, predicted = torch.max(logits.data, 1)
        
        conf_matrix += confusion_matrix(labels.data.cpu().numpy(), predicted.cpu().numpy(), labels=np.arange(nb_classes))

        line_plot(viz, "loss", cur_iter, loss.item())
        line_plot(viz, "logp(z)", cur_iter, mean_logpz)
        line_plot(viz, "log|df/dz|", cur_iter, mean_trace)
        line_plot(viz, "logp(y|z)", cur_iter, - mean_entropy.item())

        # file logging
        log_dict = {"iter": cur_iter, "loss": loss.item(), "logpz": mean_logpz, "logdet": mean_trace, "logp(y|z)": -mean_entropy.item(), "epoch": epoch}
        train_log.write("{}\n".format(json.dumps(log_dict)))
        train_log.flush()

        del logpz, trace, logpx, loss, mean_trace, mean_entropy, mean_logpz, inputs, labels, predicted
        
    accuracy_per_class, f1_per_class = scores_per_class(conf_matrix)

    line_plot_per_dim(viz, "train accuracy", epoch, accuracy_per_class)

    line_plot_mean(viz, "train accuracy", epoch, accuracy_per_class)
    line_plot_mean(viz, "train f1", epoch, f1_per_class)

    line_plot(viz, "objective", epoch, objective) 

    # file logging
    log_dict = {"epoch": epoch, "accuracy per class": accuracy_per_class.tolist(), "f1 per class": f1_per_class.tolist(), "objective": objective}

    train_log.write("{}\n".format(json.dumps(log_dict)))
    train_log.flush()

    return conf_matrix, np.mean(accuracy_per_class)

def test(model, epoch, testloader, viz, test_log, use_cuda=False, flag="validation", classification_weight=1, nb_classes=8):

    model.eval()

    criterion = nn.CrossEntropyLoss()
    
    test_loss = 0.
    test_rec_err = 0.
    nb_tiles = 0.

    conf_matrix = np.zeros((nb_classes, nb_classes))
    for batch_idx, (inputs, labels) in enumerate(testloader):
        
        nb_tiles += len(labels)

        if use_cuda:
            inputs = inputs.cuda() # GPU settings
            labels = labels.cuda()
        
        logits, zs, logpz, trace = model(inputs)  # Forward Propagation

        # compute loss
        logpx = logpz + trace
        loss = bits_per_dim(logpx, inputs).mean()

        mean_entropy = criterion(logits, labels)
        loss += classification_weight * mean_entropy

        x_re = model.module.inverse(zs) if use_cuda else model.inverse(zs)

        _, predicted = torch.max(logits.data, 1)

        conf_matrix += confusion_matrix(labels.data.cpu().numpy(), predicted.cpu().numpy(), np.arange(nb_classes))

        test_loss += loss.item()
        test_rec_err += (inputs - x_re).abs().sum().item()

        del logpz, trace, logpx, loss, mean_entropy, inputs, x_re, labels, predicted

    line_plot(viz, flag + " recons err", epoch, test_rec_err / nb_tiles)

    accuracy_per_class, f1_per_class = scores_per_class(conf_matrix)

    line_plot_per_dim(viz, flag + " accuracy", epoch, accuracy_per_class)
    # line_plot_per_dim(viz, flag + " f1", epoch, f1_per_class)

    line_plot_mean(viz, flag + " accuracy", epoch, accuracy_per_class)
    line_plot_mean(viz, flag + " f1", epoch, f1_per_class)

    # file logging
    log_dict = {"recons err": test_rec_err / nb_tiles, "epoch": epoch, "accuracy per class": accuracy_per_class.tolist(), "f1 per class": f1_per_class.tolist()}
    test_log.write("{}\n".format(json.dumps(log_dict)))
    test_log.flush()

    return conf_matrix, np.mean(accuracy_per_class)
