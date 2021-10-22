from wavenet.dataset import WaveNetDataset


def test_dataset():
    path = 'data/TIMIT/'
    dataset = WaveNetDataset(
        dirpath=path,
        output_length=1,
        receptive_field=5000,
        quantization_levels=256,
    )
    n = 100
    for i in range(n):
        x, y = dataset[i]
