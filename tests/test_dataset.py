from wavenet.config import get_config
from wavenet.dataset import WaveNetDataset


config = get_config()
dirpath = config.DATASET.DIRPATH
target_length = config.DATASET.TARGET_LENGTH
quantization_levels = config.DATASET.QUANTIZATION_LEVELS
receptive_field = 4093
input_length = receptive_field + target_length - 1


def test_int_indexing():
    # test individually selecting the first 100 elements only
    dataset = WaveNetDataset(
        dirpath=dirpath,
        target_length=target_length,
        receptive_field=receptive_field,
        quantization_levels=quantization_levels,
    )
    n = 100
    for i in range(n):
        x, y = dataset[i]
        assert x.shape == (quantization_levels, input_length)
        assert y.shape == (target_length, )


def test_slice_indexing():
    # test slicing the first 100 elements only
    dataset = WaveNetDataset(
        dirpath=dirpath,
        target_length=target_length,
        receptive_field=receptive_field,
        quantization_levels=quantization_levels,
    )
    n = 100
    x, y = dataset[:n]
    assert x.shape == (n, quantization_levels, input_length)
    assert y.shape == (n, target_length)
