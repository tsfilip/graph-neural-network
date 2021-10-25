import torch
from torch.utils.tensorboard import SummaryWriter


class EarlyStopping:
    """Early stopping callback. Early stop property is set to False when the training loss is not improving
    for patience training steps."""

    def __init__(self, patience, delta=0):
        """Args:
        delta: Minimal value between best_loss and loss for which we consider loss improvement.
        patience: Number of steps before we set early_stop to True if there is not a loss improvements.
        """
        self.min_delta = delta
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, loss):
        if self.best_loss is None:
            self.best_loss = loss
        elif self.best_loss - loss > self.min_delta:
            self.best_loss = loss
            self.counter = 0
        elif self.best_loss - loss < self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


class TBLogs:
    def __init__(self, log_path):
        self.writter = SummaryWriter(log_path)

    def __call__(self, epoch, *args, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, float):
                self.writter.add_scalar(key, value, epoch)
            if isinstance(value, tuple):
                train, validation = value
                self.writter.add_scalars(key, {'train': train, 'validation': validation}, epoch)

    def save_graph(self, model):
        self.writter.add_graph(model, torch.tensor([3, 4, 5]))

