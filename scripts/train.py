import logging

from wavenet.args import WaveNetArgParser
from wavenet.model import WaveNet
from wavenet.dataset import WaveNetDataset
from wavenet.training import WaveNetTrainer


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
    logging.info(repr(model))

    logging.info('Initializing dataset')
    dataset = WaveNetDataset(
        dirpath=args.dirpath,
        receptive_field=model.receptive_field,
        target_length=args.target_length,
        quantization_levels=args.quantization_levels,
    )
    logging.info(repr(dataset))

    logging.info('Initializing trainer')
    trainer = WaveNetTrainer(
        model=model,
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        workers=args.workers,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    logging.info(repr(trainer))

    logging.info('Starting training loop')
    trainer.train()


if __name__ == '__main__':
    main()
