import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from CustomDataset import CustomDataset
import numpy as np

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 26)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


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


if torch.cuda.is_available():
    print("Use GPU")

else:
    print("No GPU")

X_train = np.load('X_train.npy/X_train.npy')
y_train = np.load('y_train.npy/y_train.npy')
X_valid = np.load('X_val.npy/X_val.npy')
y_valid = np.load('y_val.npy')

amounts_in_category = np.sum(y_train, axis=0)

y_train = np.argmax(y_train, axis=1)
y_valid = np.argmax(y_valid, axis=1)
y_train = torch.from_numpy(y_train)
y_valid = torch.from_numpy(y_valid)

X_train = torch.unsqueeze(torch.from_numpy(X_train), 1).float()
X_valid = torch.unsqueeze(torch.from_numpy(X_valid), 1).float()

class_sample_counts = np.array([amounts_in_category[i] for i in range(len(amounts_in_category))])
weights = 1. / class_sample_counts
samples_weights = np.array([weights[t] for t in y_train.numpy()])

batch_size = 32
epochs = 30
learning_rate = 0.0001

samples_weights = torch.from_numpy(samples_weights)
samples_weights = samples_weights.double()

sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weights, len(samples_weights))

train_loader = DataLoader(dataset=CustomDataset(X_train, y_train, True), batch_size=batch_size, sampler=sampler)
test_loader = DataLoader(dataset=CustomDataset(X_valid, y_valid, True), batch_size=batch_size, shuffle=True)

model = CNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

train_accu, test_accu, train_loss, test_loss = [], [], [], []


for i in range(epochs):
    print(f"Epoch: {i+1}")
    back_pass(model, train_loader, loss_fn, optimizer)
    test_accuracy, test_l = test(model, test_loader, loss_fn)
    train_accuracy, train_l = test(model, train_loader, loss_fn)
    print(f"Accuracy on train data: {train_accuracy}%")
    print(f"Accuracy on test data: {test_accuracy}%")
    train_accu.append(train_accuracy)
    test_accu.append(test_accuracy)
    test_loss.append(test_l)
    train_loss.append(train_l)


test_accuracy, test_l, y_pred_test, y_real_test = test(model, test_loader, loss_fn, True)
train_accuracy, train_l, y_pred_train, y_real_train = test(model, train_loader, loss_fn, True)
