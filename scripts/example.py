import argparse
import json
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torchaudio

from wavenet.config import get_config
from wavenet.model import WaveNet
from wavenet.utils import one_hot_encode


def main():
    parser = argparse.ArgumentParser(description='model training')
    parser.add_argument('input',
                        help='model config file or directory')
    parser.add_argument('database',
                        help='path to database to evaluate on')
    parser.add_argument('output',
                        help='output file name without extension')
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

    model_dir = os.path.dirname(args.input)
    checkpoint_path = os.path.join(model_dir, 'checkpoint.pt')
    state = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state['model'])

    paths = []
    for root, folders, files in os.walk(args.database):
        for file in files:
            if file.lower().endswith(('.wav', '.flac')):
                path = os.path.join(root, file)
                paths.append(path)
    random.seed(0)
    path = random.choice(paths)

    waveform, _ = torchaudio.load(path)
    waveform = waveform[0, :]
    quantization_levels = config.DATASET.QUANTIZATION_LEVELS
    one_hot = one_hot_encode(waveform, quantization_levels)
    input_ = one_hot[:, :-1]
    target = one_hot[:, -(len(waveform)-model.receptive_field):]
    target = target.argmax(dim=0)

    input_ = input_.unsqueeze(0)
    target = target.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        if args.cuda:
            model.cuda()
            input_, target = input_.cuda(), target.cuda()
        with torch.cuda.amp.autocast(enabled=args.mixed_precision):
            output = model(input_)

    criterion = nn.CrossEntropyLoss()

    losses = []

    for i in range(output.shape[-1]):

        output_, target_ = output[:, :, i], target[:, i]

        with torch.cuda.amp.autocast(enabled=args.mixed_precision):
            loss = criterion(output_, target_).item()
            losses.append(loss)

    np.save(f'{args.output}.npy', np.array(losses))

    metadata = {'model': args.input, 'file': path}
    with open(f'{args.output}_metadata.json', 'w') as f:
        json.dump(metadata, f)


if __name__ == '__main__':
    main()
