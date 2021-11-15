import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import one_hot_encode, mu_law_expand, zero_pad


class ResidualBlock(nn.Module):
    def __init__(self, dilation, kernel_size, residual_channels,
                 dilation_channels, skip_channels, bias):
        super().__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.filter_conv = nn.Conv1d(
            in_channels=residual_channels,
            out_channels=dilation_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            bias=bias,
        )
        self.gate_conv = nn.Conv1d(
            in_channels=residual_channels,
            out_channels=dilation_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            bias=bias,
        )
        self.residual_conv = nn.Conv1d(
            in_channels=dilation_channels,
            out_channels=residual_channels,
            kernel_size=1,
            bias=bias,
        )
        self.skip_conv = nn.Conv1d(
            in_channels=dilation_channels,
            out_channels=skip_channels,
            kernel_size=1,
            bias=bias,
        )

    def forward(self, x, output_length):
        filter_ = self.filter_conv(x)
        filter_ = torch.tanh(filter_)
        gate = self.gate_conv(x)
        gate = torch.sigmoid(gate)
        z = filter_*gate
        residual = self.residual_conv(z)
        x = x[:, :, self.dilation*(self.kernel_size-1):]
        x = x + residual
        z = z[:, :, -output_length:]
        skip = self.skip_conv(z)
        return x, skip


class WaveNet(nn.Module):
    def __init__(self, layers=10, blocks=4, kernel_size=2, input_channels=256,
                 residual_channels=32, dilation_channels=32, skip_channels=256,
                 end_channels=256, output_channels=256,
                 initial_filter_width=1, bias=False):
        super().__init__()
        self.layers = layers
        self.blocks = blocks
        self.kernel_size = kernel_size
        self.input_channels = input_channels
        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.skip_channels = skip_channels
        self.end_channels = end_channels
        self.output_channels = output_channels
        self.initial_filter_width = initial_filter_width
        self.bias = bias

        self.residual_blocks = nn.ModuleList()
        for b in range(blocks):
            for i in range(layers):
                dilation = 2**i
                self.residual_blocks.append(
                    ResidualBlock(
                        dilation=dilation,
                        kernel_size=kernel_size,
                        residual_channels=residual_channels,
                        dilation_channels=dilation_channels,
                        skip_channels=skip_channels,
                        bias=bias,
                    )
                )
        self.initial_conv = nn.Conv1d(
            in_channels=input_channels,
            out_channels=residual_channels,
            kernel_size=initial_filter_width,
            bias=bias,
        )
        self.end_conv_1 = nn.Conv1d(
            in_channels=skip_channels,
            out_channels=end_channels,
            kernel_size=1,
            bias=bias,
        )
        self.end_conv_2 = nn.Conv1d(
            in_channels=end_channels,
            out_channels=output_channels,
            kernel_size=1,
            bias=bias,
        )

    def __repr__(self):
        kwargs = [
            'layers',
            'blocks',
            'kernel_size',
            'input_channels',
            'residual_channels',
            'dilation_channels',
            'skip_channels',
            'end_channels',
            'output_channels',
            'initial_filter_width',
            'bias',
        ]
        kwargs = [f'{kwarg}={getattr(self, kwarg)}' for kwarg in kwargs]
        kwargs = ', '.join(kwargs)
        module_name = self.__class__.__module__
        class_name = self.__class__.__name__
        return f'{module_name}.{class_name}({kwargs})'

    def forward(self, x):
        n_in, c_in, l_in = x.shape
        l_out = l_in - self.receptive_field + 1
        skip_sum = torch.zeros((n_in, self.skip_channels, l_out))
        if x.is_cuda:
            skip_sum = skip_sum.cuda()
        x = self.initial_conv(x)
        for res_block in self.residual_blocks:
            x, skip = res_block(x, l_out)
            skip_sum += skip
        x = torch.relu(skip_sum)
        x = self.end_conv_1(x)
        x = torch.relu(x)
        x = self.end_conv_2(x)
        return x

    @property
    def receptive_field(self):
        dilations = [2**i for i in range(self.layers)]
        block_field = (self.kernel_size - 1)*sum(dilations) + 1
        stack_field = self.blocks*(block_field - 1) + 1
        total_field = stack_field + self.initial_filter_width - 1
        return total_field

    def generate(self, n_samples, init_samples=None):
        if init_samples is None:
            init_samples = torch.zeros(self.receptive_field)
        elif len(init_samples) < self.receptive_field:
            init_samples = zero_pad(init_samples, n_out=self.receptive_field)
        else:
            init_samples = init_samples[-self.receptive_field:]
        input_ = one_hot_encode(init_samples, self.input_channels)
        input_ = input_.unsqueeze(0)
        waveform = torch.zeros(n_samples)
        old_progress = -1
        for i in range(n_samples):
            progress = int((i+1)/n_samples*100)
            if progress != old_progress:
                print(f'{progress}%')
            old_progress = progress
            output = self(input_).flatten()
            prob = F.softmax(output, dim=0).numpy()
            sample = np.random.choice(self.input_channels, p=prob)
            input_[:, :, :-1] = input_[:, :, 1:]
            input_[:, :, -1] = 0
            input_[:, sample, -1] = 1
            waveform[i] = sample
        waveform = 2*waveform/(self.output_channels - 1) - 1
        return mu_law_expand(waveform, self.output_channels)
