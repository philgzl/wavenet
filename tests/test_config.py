from wavenet.args import WaveNetArgParser
from wavenet.config import get_config


def test_args():
    parser = WaveNetArgParser(description='model training')
    args = parser.parse_args('')

    assert len(args.__dict__) == len(parser.arg_map)

    config = get_config()
    config.update_from_args(args, parser.arg_map)


def test_hash():
    config_1, config_2 = get_config(), get_config()
    config_1.set_field(['MODEL', 'LAYERS'], 8)
    config_2.set_field(['MODEL', 'LAYERS'], 10)
    assert config_1.get_hash() != config_2.get_hash()
