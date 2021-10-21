import torch
import torch.nn as nn
import torch.optim as optim

from wavenet.model import WaveNet
from wavenet.dataset import Dataset


TIMIT_DIR = 'data/TIMIT/'
OUTPUT_LENGTH = 5000
BATCH_SIZE = 32
EPOCHS = 1
SHUFFLE = True
WORKERS = 0
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0


model = WaveNet()
print(f'Receptive field: {model.receptive_field}')

dataset = Dataset(
    dirpath=TIMIT_DIR,
    output_length=OUTPUT_LENGTH,
    receptive_field=model.receptive_field,
)

dataloader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=BATCH_SIZE,
    shuffle=SHUFFLE,
    num_workers=WORKERS,
)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    params=model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

for epoch in range(EPOCHS):

    model.train()
    for i, data in enumerate(dataloader):
        item, target = data
        optimizer.zero_grad()
        output = model(item)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        print(f'Epoch {epoch} item {i}: train loss: {loss.item():.2f}')
