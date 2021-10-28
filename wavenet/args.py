import argparse

from .config import get_config


class WaveNetArgParser(argparse.ArgumentParser):

    arg_map = {
        'layers': ['MODEL', 'LAYERS'],
        'blocks': ['MODEL', 'BLOCKS'],
        'kernel_size': ['MODEL', 'KERNEL_SIZE'],
        'residual_channels': ['MODEL', 'RESIDUAL_CHANNELS'],
        'dilation_channels': ['MODEL', 'DILATION_CHANNELS'],
        'skip_channels': ['MODEL', 'SKIP_CHANNELS'],
        'end_channels': ['MODEL', 'END_CHANNELS'],
        'initial_filter_width': ['MODEL', 'INITIAL_FILTER_WIDTH'],
        'bias': ['MODEL', 'BIAS'],

        'dirpath': ['DATASET', 'DIRPATH'],
        'target_length': ['DATASET', 'TARGET_LENGTH'],
        'quantization_levels': ['DATASET', 'QUANTIZATION_LEVELS'],

        'batch_size': ['TRAINING', 'BATCH_SIZE'],
        'shuffle': ['TRAINING', 'SHUFFLE'],
        'epochs': ['TRAINING', 'EPOCHS'],
        'learning_rate': ['TRAINING', 'LEARNING_RATE'],
        'weight_decay': ['TRAINING', 'WEIGHT_DECAY'],
        'train_val_split': ['TRAINING', 'TRAIN_VAL_SPLIT'],
    }

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
        group.add_argument('--shuffle', type=lambda x: bool(int(x)))
        group.add_argument('--epochs', type=int)
        group.add_argument('--learning_rate', type=float)
        group.add_argument('--weight_decay', type=float)
        group.add_argument('--train_val_split', type=float)

        config = get_config()
        defaults = {}
        for arg_name, key_list in self.arg_map.items():
            defaults[arg_name] = config.get_field(key_list)
        self.set_defaults(**defaults)
