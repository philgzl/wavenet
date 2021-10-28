import torch

from wavenet.dataset import one_hot_encode


def test_mu_lay():
    x = torch.tensor([-1, 0, 1])
    x = one_hot_encode(x, 8)

    assert (
        x == torch.tensor([
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 1],
        ])
    ).all()
