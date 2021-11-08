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

    def log(self, epochs):
        train_loss = self.losses['train'][-1]
        val_loss = self.losses['val'][-1]
        if epochs == 0:
            logging.info(f'Initial state '
                         f'train loss: {train_loss:.2f}, '
                         f'val loss: {val_loss:.2f}')
        else:
            logging.info(f'Epoch {epochs}/{self.epochs} '
                         f'train loss: {train_loss:.2f}, '
                         f'val loss: {val_loss:.2f}')


class WaveNetTrainer:
    def __init__(self, model, dataset, checkpoint_path, batch_size=32,
                 shuffle=True, workers=0, epochs=10, learning_rate=1e-3,
                 weight_decay=0.0, train_val_split=0.8, cuda=False,
                 ignore_checkpoint=False):
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
        self.checkpoint_path = checkpoint_path
        self.ignore_checkpoint = ignore_checkpoint

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

        self.scaler = torch.cuda.amp.GradScaler()

        self.logger = LossLogger(epochs)

    def __repr__(self):
        kwargs = [
            'checkpoint_path',
            'batch_size',
            'shuffle',
            'workers',
            'epochs',
            'learning_rate',
            'weight_decay',
            'train_val_split',
            'cuda',
            'checkpoint_path',
            'ignore_checkpoint',
        ]
        kwargs = [f'{kwarg}={getattr(self, kwarg)}' for kwarg in kwargs]
        kwargs = ', '.join(kwargs)
        module_name = self.__class__.__module__
        class_name = self.__class__.__name__
        return f'{module_name}.{class_name}({kwargs})'

    def train(self):
        # check for a checkpoint
        if not self.ignore_checkpoint and os.path.exists(self.checkpoint_path):
            logging.info('Checkpoint found')

            # get number of epochs model was trained on
            epochs = self.load_checkpoint()

            # if training was interrupted then resume training
            if epochs < self.epochs:
                logging.info(f'Resuming training at epoch {epochs+1}')
            else:
                logging.info('Model is already trained')
                return
        else:
            # if no checkpoint then model was trained on 0 epochs
            epochs = 0

            # cast to cuda if requested
            if self.cuda:
                self.model.cuda()

            # evaluate before training
            logging.info('Evaluating before first epoch')
            train_loss = self.evaluate(self.train_dataloader)
            val_loss = self.evaluate(self.val_dataloader)
            self.logger.add(train_loss, val_loss)
            self.logger.log(epochs)

        # initialize timer
        timer = TrainingTimer(self.epochs - epochs)
        timer.start()

        # start main loop
        for epoch in range(epochs, self.epochs):
            logging.info(f'Training on epoch {epoch+1}/{self.epochs}')

            # training routine
            self.model.train()
            for i, item in enumerate(self.train_dataloader):
                # get inputs and labels
                input_, target = item

                # cast to cuda if requested
                if self.cuda:
                    input_, target = input_.cuda(), target.cuda()

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # run the forward past with autocasting
                with torch.cuda.amp.autocast():
                    output = self.model(input_)
                    loss = self.criterion(output, target)

                # compute gradients on a scaled loss
                self.scaler.scale(loss).backward()

                # update parameters
                self.scaler.step(self.optimizer)

                # update the scale
                self.scaler.update()

            # evaluate
            logging.info('Evaluating')
            train_loss = self.evaluate(self.train_dataloader)
            val_loss = self.evaluate(self.val_dataloader)
            self.logger.add(train_loss, val_loss)
            self.logger.log(epoch+1)

            # save checkpoint
            logging.info('Saving checkpoint')
            self.save_checkpoint(epoch+1)

            # estimate time left
            timer.step()
            timer.log()

        # log end of training
        logging.info('Training over')

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for i, item in enumerate(dataloader):
                input_, target = item
                if self.cuda:
                    input_, target = input_.cuda(), target.cuda()
                with torch.cuda.amp.autocast():
                    output = self.model(input_)
                    loss = self.criterion(output, target)
                    total_loss += loss.item()
        total_loss /= len(dataloader)
        return total_loss

    def save_checkpoint(self, epochs):
        state = {
            'epochs': epochs,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'losses': self.logger.losses,
        }
        torch.save(state, self.checkpoint_path)

    def load_checkpoint(self):
        state = torch.load(self.checkpoint_path)
        self.model.load_state_dict(state['model'])
        if self.cuda:
            self.model.cuda()
            # if the model was moved to cuda then the optimizer needs to be
            # reinitialized before loading the optimizer state dictionary
            # see https://github.com/pytorch/pytorch/issues/2830
            self.optimizer.__init__(
                params=self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        self.optimizer.load_state_dict(state['optimizer'])
        self.logger.losses = state['losses']
        return state['epochs']
