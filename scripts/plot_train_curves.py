import argparse
from glob import glob
import logging
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch


def main():
    parser = argparse.ArgumentParser(description='model training')
    parser.add_argument('input', nargs='+',
                        help='model config files or directories')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    models = []
    for input_ in args.input:
        if not glob(input_):
            logging.info(f'Model not found: {input_}')
        models += glob(input_)

    fig, ax = plt.subplots()
    for model in models:
        checkpoint = os.path.join(model, 'checkpoint.pt')
        state = torch.load(checkpoint, map_location='cpu')
        losses = state['losses']
        train, val = losses['train'], losses['val']
        line, = ax.plot(train, label=model)
        ax.plot(val, '--', color=line.get_color())

    lines = [
        Line2D([], [], color='k', linestyle='-'),
        Line2D([], [], color='k', linestyle='--'),
    ]
    lh = ax.legend(loc=1)
    ax.legend(lines, ['train', 'val'], loc=2)
    ax.add_artist(lh)

    ax.grid()

    plt.show()


if __name__ == '__main__':
    main()
