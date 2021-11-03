# wavenet
Yet another WaveNet implementation on PyTorch

# Installation

1. Clone and change directory
```
git clone https://github.com/philgzl/wavenet.git
cd wavenet
```

2. Create a virtual environment and activate
```
python -m venv venv
source venv/bin/activate
```

3. Install requirements
```
pip install -r requirements.txt
```

4. Install project
```
python setup.py develop
```

# How to use

## Initializing a model

Models are initialized with the `scripts/init_model.py` script.
This creates a new directory under `models/` containing the configuration file for the model.
The model directory is named after a unique ID generated from the configuration file.

The script takes as arguments the hyperparameters for the model.
You can use `python scripts/init_model.py --help` for a list of available hyperparameters.

Example:
```
python scripts/init_model.py --layers 10
```

## Training a model

To train an initialized model, simply call the `scripts/train.py` script with the model path as argument:
```
python scripts/train.py models/{model_id}/
```
A checkpoint file `checkpoint.pt` will be created in the model directory, next to the configuration file.

## Submitting jobs to GPU cluster

If you are on a server equipped with GPUs and able to send jobs through the `bsub` command, you can submit a training job using:
```
bash jobs/train.sh models/{model_id}/
```
Logs will be stored under `jobs/logs/`.

# To do
* Evaluation script
* Global/local conditioning
* Speaker selection
