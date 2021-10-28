import os

from wavenet.args import WaveNetArgParser
from wavenet.config import get_config
from wavenet.utils import dump_yaml


def main():
    parser = WaveNetArgParser(description='model training')
    parser.add_argument('-f', '--force', action='store_true',
                        help='overwrite config file if it already exists')
    args = parser.parse_args()

    config = get_config()
    config.update_from_args(args, parser.arg_map)
    model_id = config.get_hash()

    model_dir = os.path.join('models', model_id)
    output_path = os.path.join(model_dir, 'config.yaml')
    if not args.force and os.path.exists(output_path):
        raise OSError(f'{output_path} already exists')
    resp = input(f'File {output_path} will be initialized. Continue? y/n')
    if resp == 'y':
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        dump_yaml(config.to_dict(), output_path)
        print(f'Initialized {output_path}')
    else:
        print('Aborted')


if __name__ == '__main__':
    main()
