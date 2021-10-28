import argparse
import logging
import os
import sys

import torch

from wavenet.config import get_config
from wavenet.dataset import WaveNetDataset
from wavenet.model import WaveNet
from wavenet.training import WaveNetTrainer


def main():
    parser = argparse.ArgumentParser(description='model training')
    parser.add_argument('input',
                        help='input config file')
    parser.add_argument('--ignore-checkpoint', action='store_true',
                        help='ignore checkpoint')
    parser.add_argument('--cuda', action='store_true',
                        help='train on gpu')
    parser.add_argument('--workers', type=int, default=0,
                        help='number of workers')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    config = get_config(args.input)

    torch.manual_seed(config.SEED)

    logging.info('Initializing model')
    model = WaveNet(
        layers=config.MODEL.LAYERS,
        blocks=config.MODEL.BLOCKS,
        kernel_size=config.MODEL.KERNEL_SIZE,
        input_channels=config.DATASET.QUANTIZATION_LEVELS,
        residual_channels=config.MODEL.RESIDUAL_CHANNELS,
        dilation_channels=config.MODEL.DILATION_CHANNELS,
        skip_channels=config.MODEL.SKIP_CHANNELS,
        end_channels=config.MODEL.END_CHANNELS,
        output_channels=config.DATASET.QUANTIZATION_LEVELS,
        initial_filter_width=config.MODEL.INITIAL_FILTER_WIDTH,
        bias=config.MODEL.BIAS,
    )
    logging.info(repr(model))

    logging.info('Initializing dataset')
    dataset = WaveNetDataset(
        dirpath=config.DATASET.DIRPATH,
        receptive_field=model.receptive_field,
        target_length=config.DATASET.TARGET_LENGTH,
        quantization_levels=config.DATASET.QUANTIZATION_LEVELS,
    )
    logging.info(repr(dataset))

    model_dir = os.path.dirname(args.input)
    checkpoint_path = os.path.join(model_dir, 'checkpoint.pt')

    logging.info('Initializing trainer')
    trainer = WaveNetTrainer(
        model=model,
        dataset=dataset,
        checkpoint_path=checkpoint_path,
        batch_size=config.TRAINING.BATCH_SIZE,
        shuffle=config.TRAINING.SHUFFLE,
        workers=args.workers,
        epochs=config.TRAINING.EPOCHS,
        learning_rate=config.TRAINING.LEARNING_RATE,
        weight_decay=config.TRAINING.WEIGHT_DECAY,
        train_val_split=config.TRAINING.TRAIN_VAL_SPLIT,
        cuda=args.cuda,
        ignore_checkpoint=args.ignore_checkpoint,
    )
    logging.info(repr(trainer))

    logging.info('Launching trainer')
    trainer.train()


if __name__ == '__main__':
    main()
