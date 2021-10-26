import os
import logging
import time

import torch
import torch.nn as nn
import torch.optim as optim


class TrainingTimer:
    def __init__(self, total_steps):
        self.total_steps = total_steps
        self.steps_taken = None
        self.start_time = None

    def start(self):
        self.start_time = time.time()
        self.steps_taken = 0

    def step(self):
        self.steps_taken += 1

    def log(self):
        log = f'Estimated time left: {self.etl_fmt()}'
        logging.info(log)

    def etl_fmt(self):
        etl = self.estimated_time_left
        h, m, s = int(etl//3600), int((etl % 3600)//60), int(etl % 60)
        return f'{h} h {m} m {s} s'

    @property
    def elapsed_time(self):
        return time.time() - self.start_time

    @property
    def time_per_step(self):
        return self.elapsed_time/self.steps_taken

    @property
    def steps_left(self):
        return self.total_steps - self.steps_taken

    @property
    def estimated_time_left(self):
        return self.time_per_step*self.steps_left


class WaveNetTrainer:
    def __init__(self, model, dataset, batch_size=32, shuffle=True,
                 workers=0, epochs=1, learning_rate=1e-3, weight_decay=0.0,
                 train_val_split=0.8):
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.workers = workers
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.train_val_split = train_val_split

        train_length = int(len(self.dataset)*train_val_split)
        val_length = len(self.dataset) - train_length
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            dataset, [train_length, val_length])

        self.train_dataloader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=workers,
        )
        self.val_dataloader = torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=workers,
        )

        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(
            params=model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    def __repr__(self):
        kwargs = [
            'batch_size',
            'shuffle',
            'workers',
            'epochs',
            'learning_rate',
            'weight_decay',
            'train_val_split',
        ]
        kwargs = [f'{kwarg}={getattr(self, kwarg)}' for kwarg in kwargs]
        kwargs = ', '.join(kwargs)
        module_name = self.__class__.__module__
        class_name = self.__class__.__name__
        return f'{module_name}.{class_name}({kwargs})'

    def train(self, checkpoint_path='checkpoint.pt'):
        if os.path.exists(checkpoint_path):
            logging.info('Checkpoint found')
            epoch = self.load_checkpoint(checkpoint_path)
            if epoch+1 < self.epochs:
                logging.info(f'Resuming training at epoch {epoch+1}')
            else:
                logging.info('Model is already trained')
                return
        else:
            epoch = -1

        epoch_timer = TrainingTimer(self.epochs - epoch - 1)
        step_timer = TrainingTimer(len(self.train_dataloader))

        epoch_timer.start()

        for epoch in range(epoch+1, self.epochs):
            self.model.train()
            step_timer.start()

            for i, item in enumerate(self.train_dataloader):
                input_, target = item
                self.optimizer.zero_grad()
                output = self.model(input_)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                logging.info(f'Epoch {epoch+1}/{self.epochs}, '
                             f'step {i+1}/{len(self.train_dataloader)}, '
                             f'train loss: {loss.item():.2f}')

                step_timer.step()
                logging.info(f'Estimated time left on epoch '
                             f'{epoch+1}/{self.epochs} :'
                             f'{step_timer.etl_fmt()}')

            logging.info('Evaluating')
            train_loss = self.evaluate(self.val_dataloader)
            val_loss = self.evaluate(self.val_dataloader)
            logging.info(f'Epoch {epoch+1}/{self.epochs}, '
                         f'train loss: {train_loss:.2f}, '
                         f'val loss: {val_loss:.2f}')

            logging.info('Saving checkpoint')
            self.save_checkpoint(epoch, checkpoint_path)

            epoch_timer.timer.step()
            logging.info(f'Estimated time left: {epoch_timer.etl_fmt()}')

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for i, item in enumerate(dataloader):
                input_, target = item
                output = self.model(input_)
                loss = self.criterion(output, target)
                total_loss += loss.item()
        total_loss /= len(dataloader)
        return total_loss

    def save_checkpoint(self, epoch, path):
        state = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, path)

    def load_checkpoint(self, path):
        state = torch.load(path)
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        return state['epoch']
