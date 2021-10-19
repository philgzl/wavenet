import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, bias):
        super(CausalConv1d, self).__init__()
        self.pad = (kernel_size - 1)*dilation
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, dilation, kernel_size, residual_channels,
                 dilation_channels, skip_channels, bias):
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
            dilation=dilation,
            bias=bias,
        )
        self.skip_conv = nn.Conv1d(
            in_channels=dilation_channels,
            out_channels=skip_channels,
            kernel_size=1,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        filter_ = F.tanh(self.filter_conv(x))
        filter_ = F.tanh(filter_)
        gate = F.sigmoid(self.gate_conv(x))
        gate = F.sigmoid(gate)
        z = filter_*gate
        residual = x + self.residual_conv(z)
        skip = self.skip_conv(z)
        return residual, skip


class WaveNet(nn.Module):
    def __init__(
        self,
        layers=10,
        blocks=4,
        kernel_size=2,
        residual_channels=32,
        dilation_channels=32,
        skip_channels=128,
        bias=False,
    ):
        self.residual_blocks = nn.ModuleList()
        for b in range(blocks):
            for i in range(layers):
                self.residual_blocks.append(
                    ResidualBlock(
                        
                    )
                )