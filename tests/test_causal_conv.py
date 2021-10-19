import torch

from wavenet.wavenet import CausalConv1d


dilation_factors = [1, 2, 4]
input_signal = [1, 2, 3, 4, 5, 6, 7]
ouput_signals = [
    [1, 3, 6, 9, 12, 15, 18],
    [1, 2, 4, 6, 9, 12, 15],
    [1, 2, 3, 4, 6, 8, 10],
]

x = torch.tensor(input_signal, dtype=torch.float32).reshape(1, 1, -1)
for dilation, ouput_signal in zip(dilation_factors, ouput_signals):
    conv = CausalConv1d(1, 1, 3, dilation=dilation)
    conv.conv.weight.data.fill_(1)
    y = conv(x)
    y_ref = torch.tensor(ouput_signal, dtype=torch.float32).reshape(1, 1, -1)
    assert (y == y_ref).all()
