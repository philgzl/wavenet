import logging

import torch
import torch.nn as nn
import torch.optim as optim

from wavenet.model import WaveNet
from wavenet.dataset import WaveNetDataset
from wavenet.args import WaveNetArgParser


def main():
    logging.basicConfig(level=logging.INFO)

    parser = WaveNetArgParser(description='model training')
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

    logging.info('Initializing criterion')
    criterion = nn.CrossEntropyLoss()

    logging.info('Initializing optimizer')
    optimizer = optim.Adam(
        params=model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    logging.info('Starting training loop')
    for epoch in range(args.epochs):

        model.train()
        for i, item in enumerate(dataloader):
            input_, target = item
            optimizer.zero_grad()
            output = model(input_)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            log = f'Epoch {epoch} item {i}: train loss: {loss.item():.2f}'
            logging.info(log)


if __name__ == '__main__':
    main()
