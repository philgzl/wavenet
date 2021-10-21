import itertools

import torch

from wavenet.model import WaveNet


def test_forward():
    blocks = [1, 2, 3]
    layers = [1, 2, 3, 4, 5]
    kernel_size = [1, 2, 3]
    initial_filter_width = [1, 2, 3]

    for params in itertools.product(
                blocks,
                layers,
                kernel_size,
                initial_filter_width,
            ):
        model = WaveNet(
            blocks=params[0],
            layers=params[1],
            kernel_size=params[2],
            initial_filter_width=params[3],
        )
        x = torch.rand((1, 256, model.receptive_field))
        y = model(x)
        assert y.shape == (1, 256, 1)
