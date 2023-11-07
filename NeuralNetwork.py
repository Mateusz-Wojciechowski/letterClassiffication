import numpy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net_structure = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 26),
        )

    def forward(self, x):
        x = self.flatten(x)
        output = self.net_structure(x)
        return output


def back_pass(model, train_loader, loss_fn, optimizer):
    model.train()

    for (X, y) in train_loader:
        pred = model.forward(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def test(model, test_loader, loss_fn, get_y_values=False):
    model.eval()

    correct = 0
    samples_amount = 0
    total_loss = 0.0

    y_pred, y_real = [], []

    with torch.no_grad():
        for x, y in test_loader:
            pred = model(x)
            loss = loss_fn(pred, y)
            total_loss += loss.item() * y.size(0)
            pred_classes = torch.argmax(pred, dim=1)
            correct += torch.eq(torch.argmax(pred, dim=1), y).sum().item()
            samples_amount += y.size(0)

            if get_y_values:
                y_pred.append(pred_classes.numpy())
                y_real.append(y.numpy())

    accuracy = correct / samples_amount
    avg_loss = total_loss / samples_amount

    if get_y_values:
        y_pred = np.concatenate(y_pred)
        y_real = np.concatenate(y_real)
        return round(accuracy * 100, 3), round(avg_loss, 3), y_pred, y_real

    return round(accuracy * 100, 3), round(avg_loss, 3)

