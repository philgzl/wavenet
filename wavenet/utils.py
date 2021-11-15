import math
import json

import torch
import yaml


def load_yaml(path):
    with open(path) as f:
        output = yaml.safe_load(f)
    return output


def dump_yaml(dict_, path):
    with open(path, 'w') as f:
        yaml.dump(dict_, f)


def load_json(path):
    with open(path) as f:
        output = json.load(f)
    return output


def dump_json(dict_, path):
    with open(path, 'w') as f:
        json.dump(dict_, f)


def mu_law_compress(x, mu):
    return torch.sign(x)*torch.log(1+mu*torch.abs(x))/math.log(1+mu)


def mu_law_expand(x, mu):
    return torch.sign(x)*((1+mu)**torch.abs(x)-1)/mu


def one_hot_encode(x, quantization_levels):
    n = len(x)
    mu = quantization_levels-1
    x = mu_law_compress(x, mu)
    x = 0.5*mu*(x+1)
    x = x.long()
    one_hot = torch.zeros((quantization_levels, n))
    one_hot[x, torch.arange(n)] = 1
    return one_hot


def one_hot_decode(one_hot, quantization_levels):
    mu = quantization_levels-1
    x = one_hot.argmax(axis=1)
    x = 2*x/mu - 1
    return mu_law_expand(x, quantization_levels)


def zero_pad(x, n_pad=None, n_out=None, where='start'):
    if n_pad is None and n_out is None:
        raise ValueError('must provide n_pad or n_out')
    elif n_pad is not None and n_out is not None:
        raise ValueError('cannot provide both n_pad and n_out')
    if n_out is not None:
        if n_out < len(x):
            raise ValueError('n_out must be greater than input tensor length')
        n_pad = len(x) - n_out
    zeros = torch.zeros(n_pad)
    if where == 'start':
        output = torch.cat(zeros, x)
    elif where == 'end':
        output = torch.cat(x, zeros)
    else:
        raise ValueError(f'where must be start or end, got {where}')
    return output
