import math

import torch


def mu_law_compress(x, mu):
    return torch.sign(x)*torch.log(1+mu*torch.abs(x))/math.log(1+mu)


def one_hot_encode(x, quantization_levels):
    n = len(x)
    mu = quantization_levels-1
    x = mu_law_compress(x, mu)
    x = 0.5*mu*(x+1)
    x = x.long()
    one_hot = torch.zeros((quantization_levels, n))
    one_hot[x, torch.arange(n)] = 1
    return one_hot
