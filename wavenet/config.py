import yaml


class WaveNetConfig:
    def __init__(self, dict_):
        for key, value in dict_.items():
            if isinstance(value, dict):
                super().__setattr__(key, WaveNetConfig(value))
            else:
                super().__setattr__(key, value)

    def __setattr__(self, attr, value):
        class_name = self.__class__.__name__
        raise AttributeError(f'{class_name} instances are immutable')


def get_config():
    with open('config.yaml') as f:
        config_dict = yaml.safe_load(f)
    config = WaveNetConfig(config_dict)
    return config
