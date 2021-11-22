import argparse
import logging
import os
import sys

import numpy as np
import torch
import torchaudio
import torch.nn.functional as F

from wavenet.config import get_config
from wavenet.model import WaveNet
from wavenet.utils import one_hot_encode, mu_law_expand, zero_pad


def main():
    parser = argparse.ArgumentParser(description='model training')
    parser.add_argument('input',
                        help='input config file or directory')
    parser.add_argument('-n', '--n-samples', type=int, default=160000,
                        help='number of samples to generate')
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

    init_samples = None
    if init_samples is None:
        init_samples = torch.zeros(model.receptive_field)
    elif len(init_samples) < model.receptive_field:
        init_samples = zero_pad(init_samples, n_out=model.receptive_field)
    else:
        init_samples = init_samples[-model.receptive_field:]

    input_ = one_hot_encode(init_samples, model.input_channels).unsqueeze(0)
    waveform = torch.zeros(args.n_samples)

    logging.info('Generating')
    model.eval()
    with torch.no_grad():

        old_progress = -1
        for i in range(args.n_samples):
            progress = int((i+1)/args.n_samples*100)
            if progress != old_progress:
                logging.info(f'{progress}%')
            old_progress = progress

            output = model(input_).flatten()
            prob = F.softmax(output, dim=0).numpy()
            sample = np.random.choice(model.input_channels, p=prob)
            input_[:, :, :-1] = input_[:, :, 1:]
            input_[:, :, -1] = 0
            input_[:, sample, -1] = 1
            waveform[i] = sample

        waveform = 2*waveform/(model.output_channels - 1) - 1
        waveform = mu_law_expand(waveform, model.output_channels)

    torchaudio.save('temp.wav', waveform.unsqueeze(0), 16000)


if __name__ == '__main__':
    main()
