from wavenet.dataset import Dataset


def test_dataset():
    path = 'data/TIMIT/'
    dataset = Dataset(
        dirpath=path,
        output_length=1,
        receptive_field=5000,
        quantization_levels=256,
    )
    n = 100
    for i in range(n):
        x, y = dataset[i]
