import argparse

from .config import get_config


class WaveNetArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        group = self.add_argument_group('model')
        group.add_argument('--layers', type=int)
        group.add_argument('--blocks', type=int)
        group.add_argument('--kernel-size', type=int)
        group.add_argument('--residual-channels', type=int)
        group.add_argument('--dilation-channels', type=int)
        group.add_argument('--skip-channels', type=int)
        group.add_argument('--end-channels', type=int)
        group.add_argument('--initial-filter_width', type=int)
        group.add_argument('--bias', type=lambda x: bool(int(x)))

        group = self.add_argument_group('dataset')
        group.add_argument('--dirpath', type=str)
        group.add_argument('--target-length', type=int)
        group.add_argument('--quantization-levels', type=int)

        group = self.add_argument_group('training')
        group.add_argument('--batch-size', type=int)
        group.add_argument('--workers', type=int)
        group.add_argument('--shuffle', type=lambda x: bool(int(x)))
        group.add_argument('--epochs', type=int)
        group.add_argument('--learning_rate', type=float)
        group.add_argument('--weight_decay', type=float)
        group.add_argument('--train_val_split', type=float)
        group.add_argument('--cuda', type=lambda x: bool(int(x)))

        config = get_config()
        self.set_defaults(
            layers=config.MODEL.LAYERS,
            blocks=config.MODEL.BLOCKS,
            kernel_size=config.MODEL.KERNEL_SIZE,
            residual_channels=config.MODEL.RESIDUAL_CHANNELS,
            dilation_channels=config.MODEL.DILATION_CHANNELS,
            skip_channels=config.MODEL.SKIP_CHANNELS,
            end_channels=config.MODEL.END_CHANNELS,
            initial_filter_width=config.MODEL.INITIAL_FILTER_WIDTH,
            bias=config.MODEL.BIAS,
            dirpath=config.DATASET.DIRPATH,
            target_length=config.DATASET.TARGET_LENGTH,
            quantization_levels=config.DATASET.QUANTIZATION_LEVELS,
            batch_size=config.TRAINING.BATCH_SIZE,
            shuffle=config.TRAINING.SHUFFLE,
            workers=config.TRAINING.WORKERS,
            epochs=config.TRAINING.EPOCHS,
            learning_rate=config.TRAINING.LEARNING_RATE,
            weight_decay=config.TRAINING.WEIGHT_DECAY,
            train_val_split=config.TRAINING.TRAIN_VAL_SPLIT,
            cuda=config.TRAINING.CUDA,
        )
