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
        etl = self.estimated_time_left
        h, m, s = int(etl//3600), int((etl % 3600)//60), int(etl % 60)
        log = f'Estimated time left: {h} h {m} m {s} s'
        logging.info(log)

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


class LossLogger:
    def __init__(self, epochs):
        self.epochs = epochs
        self.losses = {'train': [], 'val': []}

    def add(self, train_loss, val_loss):
        self.losses['train'].append(train_loss)
        self.losses['val'].append(val_loss)

    def log(self, epoch):
        train_loss = self.losses['train'][-1]
        val_loss = self.losses['val'][-1]
        logging.info(f'Epoch {epoch+1}/{self.epochs}, '
                     f'train loss: {train_loss:.2f}, '
                     f'val loss: {val_loss:.2f}')


class WaveNetTrainer:
    def __init__(self, model, dataset, batch_size=32, shuffle=True,
                 workers=8, epochs=10, learning_rate=1e-3, weight_decay=0.0,
                 train_val_split=0.8, cuda=True):
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.workers = workers
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.train_val_split = train_val_split
        self.cuda = cuda

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

        self.logger = LossLogger(epochs)

    def __repr__(self):
        kwargs = [
            'batch_size',
            'shuffle',
            'workers',
            'epochs',
            'learning_rate',
            'weight_decay',
            'train_val_split',
            'cuda',
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

        if self.cuda:
            self.model.cuda()

        timer = TrainingTimer(self.epochs - epoch - 1)

        for epoch in range(epoch+1, self.epochs):
            self.model.train()

            logging.info(f'Training on epoch {epoch+1}/{self.epochs}')
            for i, item in enumerate(self.train_dataloader):
                input_, target = item
                if self.cuda:
                    input_, target = input_.cuda(), target.cuda()
                self.optimizer.zero_grad()
                output = self.model(input_)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

            logging.info('Evaluating')
            train_loss = self.evaluate(self.val_dataloader)
            val_loss = self.evaluate(self.val_dataloader)
            self.logger.add(train_loss, val_loss)
            self.logger.log()

            logging.info('Saving checkpoint')
            self.save_checkpoint(epoch, checkpoint_path)

            timer.step()
            timer.log()

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
            'losses': self.logger.losses,
        }
        torch.save(state, path)

    def load_checkpoint(self, path):
        state = torch.load(path)
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.logger.losses = state['losses']
        return state['epoch']
