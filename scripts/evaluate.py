import argparse
import logging
import os
import sys

import torch
import torch.nn as nn

from wavenet.config import get_config
from wavenet.dataset import WaveNetDataset
from wavenet.model import WaveNet


def main():
    parser = argparse.ArgumentParser(description='model training')
    parser.add_argument('input',
                        help='model config file or directory')
    parser.add_argument('database',
                        help='path to database to evaluate on')
    parser.add_argument('--cuda', action='store_true',
                        help='evaluate on gpu')
    parser.add_argument('--workers', type=int, default=0,
                        help='number of workers')
    parser.add_argument('--mixed-precision', action='store_true',
                        help='use mixed precision')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    if os.path.isdir(args.input):
        args.input = os.path.join(args.input, 'config.yaml')
    config = get_config(args.input)

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
        dirpath=args.database,
        receptive_field=model.receptive_field,
        target_length=config.DATASET.TARGET_LENGTH,
        quantization_levels=config.DATASET.QUANTIZATION_LEVELS,
    )
    logging.info(repr(dataset))

    logging.info('Initializing dataset')
    dataset = WaveNetDataset(
        dirpath=args.database,
        receptive_field=model.receptive_field,
        target_length=config.DATASET.TARGET_LENGTH,
        quantization_levels=config.DATASET.QUANTIZATION_LEVELS,
    )
    logging.info(repr(dataset))

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=config.TRAINING.BATCH_SIZE,
        num_workers=args.workers,
    )

    model_dir = os.path.dirname(args.input)
    checkpoint_path = os.path.join(model_dir, 'checkpoint.pt')
    state = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state['model'])

    criterion = nn.CrossEntropyLoss()

    model.eval()
    old_progress = -1
    loss = 0
    n = len(dataloader)

    with torch.no_grad():
        for i, item in enumerate(dataloader):

            progress = int((i+1)/n*100)
            if progress != old_progress:
                logging.info(f'{progress}%')
            old_progress = progress

            input_, target = item
            if args.cuda:
                input_, target = input_.cuda(), target.cuda()

            with torch.cuda.amp.autocast(enabled=args.mixed_precision):
                output = model(input_)
                loss += criterion(output, target).item()

    loss /= len(dataloader)

    logging.info(loss)


if __name__ == '__main__':
    main()
