import logging
import time

import torch

from wavenet.model import WaveNet
from wavenet.dataset import WaveNetDataset
from wavenet.args import WaveNetArgParser


def main():
    logging.basicConfig(level=logging.INFO)

    parser = WaveNetArgParser(description='dataloader benchmarking')
    args = parser.parse_args()

    logging.info('Initializing model')
    model = WaveNet(
        layers=args.layers,
        blocks=args.blocks,
        kernel_size=args.kernel_size,
        input_channels=args.quantization_levels,
        residual_channels=args.residual_channels,
        dilation_channels=args.dilation_channels,
        skip_channels=args.skip_channels,
        end_channels=args.end_channels,
        output_channels=args.quantization_levels,
        initial_filter_width=args.initial_filter_width,
        bias=args.bias,
    )

    logging.info('Initializing dataset')
    dataset = WaveNetDataset(
        dirpath=args.dirpath,
        receptive_field=model.receptive_field,
        target_length=args.target_length,
        quantization_levels=args.quantization_levels,
    )

    logging.info('Initializing dataloader')
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.workers,
    )

    logging.info('Starting benchmark')
    total_time = time.time()
    for i in range(args.epochs):
        epoch_time = time.time()
        for item in dataloader:
            pass
        epoch_time = time.time() - epoch_time
        logging.info(f'Epoch {i+1} time: {epoch_time:.2f}')
    total_time = time.time() - total_time
    logging.info(f'Total time: {total_time:.2f}')
    logging.info(f'Averate time per epoch: {total_time/args.epochs:.2f}')


if __name__ == '__main__':
    main()
