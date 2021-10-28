import json

import yaml


def load_yaml(path):
    with open(path) as f:
        output = yaml.safe_load(f)
    return output


def dump_yaml(dict_, path):
    with open(path, 'w') as f:
        yaml.dump(dict_, f)


def load_json(path):
    with open(path) as f:
        output = json.load(f)
    return output


def dump_json(dict_, path):
    with open(path, 'w') as f:
        json.dump(dict_, f)
