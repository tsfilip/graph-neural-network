import torch
import torch.nn as nn
import torch.optim as optim
from utils import EarlyStopping, TBLogs
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import trange

from dataset import read_dataset
from model import GNNClassifier

from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string("train", "/media/tom/HDD-HARD-DISK-1/datasets/CORA/", "Path to dataset directory.")
flags.DEFINE_string("model_name", "gcn.pt", "Model name.")
flags.DEFINE_string("logs", None, "Path to logging directory.")
flags.DEFINE_string("aggregation", "sum", "Aggregation type for neighbours node representations (sum or mean).")
flags.DEFINE_string("combination", "concat", "Combination type for final representation update (sum or concat).")
flags.DEFINE_integer("batch_size", 256, "Batch size for training.")
flags.DEFINE_integer("n_epochs", 500, "Number of training epochs.")
flags.DEFINE_integer("patience", 300, "Number of patience steps for early stopping.")
flags.DEFINE_float("learning_rate", 0.01, "Learning rate for training.")
flags.DEFINE_float("dropout_rate", 0.5, "dropout_rate.")
flags.DEFINE_bool("save_model", False, "Save model graph to tensorboard.")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Dataset:
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        return x, y

    def __len__(self):
        return self.target.shape[0]


def train_loop(dataloader, model, loss_fn, optimizer):
    epoch_loss = 0.
    accuracy = 0.
    n_samples = len(dataloader.dataset)
    model.train()

    for step, (samples, target) in enumerate(dataloader):
        samples, target = samples.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(samples)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * samples.size(0)
        accuracy += torch.sum(output.argmax(1) == target)

    epoch_loss /= n_samples
    accuracy /= n_samples
    return epoch_loss, float(accuracy)


def evaluate(dataloader, model, loss_fn):
    n_samples = len(dataloader.dataset)
    val_loss = 0.
    val_accuracy = 0.
    model.eval()
    with torch.no_grad():
        for samples, target in dataloader:
            samples, target = samples.to(device), target.to(device)
            model.eval()
            output = model(samples)
            loss = loss_fn(output, target)

            val_loss += loss.item() * samples.size(0)
            val_accuracy += torch.sum(output.argmax(1) == target)

    val_loss /= n_samples
    val_accuracy /= n_samples

    return val_loss, float(val_accuracy)


def main(argv):
    hidden_units = [32, 32]

    train, test, graph_info, class_names = read_dataset(FLAGS.train)
    train = DataLoader(Dataset(*train), batch_size=FLAGS.batch_size, shuffle=True)
    test = DataLoader(Dataset(*test), batch_size=FLAGS.batch_size)
    graph_info = tuple(torch.from_numpy(i).to(device) for i in graph_info)

    model = GNNClassifier(graph_info, len(class_names), hidden_units, aggregation_type=FLAGS.aggregation,
                          combination_type=FLAGS.combination, dropout_rate=FLAGS.dropout_rate)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), FLAGS.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1,
                                  patience=max(FLAGS.patience // 2, 1), threshold_mode='abs')
    loss_fn = nn.CrossEntropyLoss()
    p_bar = trange(FLAGS.n_epochs)
    early_stopping = EarlyStopping(FLAGS.patience)
    tb_logs = TBLogs(FLAGS.logs)
    if FLAGS.save_model:
        tb_logs.save_graph(model)
    minimal_loss = float('inf')

    for epoch in p_bar:
        epoch_loss, epoch_accuracy = train_loop(train, model, loss_fn, optimizer)
        val_loss, val_accuracy = evaluate(test, model, loss_fn)

        p_bar.set_description("Epoch: {0} - loss: {1:.4f}, accuracy: {2:.0f}%, val_loss: {3:.4f}, " 
                              "val_accuracy: {4:.0f}%".format(epoch, epoch_loss, epoch_accuracy * 100, val_loss,
                                                              val_accuracy * 100))

        tb_logs(epoch,
                loss=(epoch_loss, val_loss),
                accuracy=(epoch_accuracy, val_accuracy),
                learning_rate=optimizer.param_groups[0]["lr"])
        scheduler.step(val_loss)
        early_stopping(val_loss)

        if val_loss < minimal_loss:  # Save model with the lowest val_loss value
            minimal_loss = val_loss
            torch.save(model.state_dict(), FLAGS.model_name)
        if early_stopping.early_stop:
            print("Early stopping at {0} epoch.".format(epoch))
            break


if __name__ == '__main__':
    app.run(main)
