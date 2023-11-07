import numpy
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from NeuralNetwork import NeuralNetwork
from CustomDataset import CustomDataset
from NeuralNetwork import back_pass, test
from sklearn.metrics import confusion_matrix, classification_report

X_train = np.load('X_train.npy/X_train.npy')
y_train = np.load('y_train.npy/y_train.npy')
X_valid = np.load('X_val.npy/X_val.npy')
y_valid = np.load('y_val.npy')

amounts_in_category = np.sum(y_train, axis=0)
print("Amount of data in each category:")
print(amounts_in_category)

## transforming to index from one-hot
y_train = numpy.argmax(y_train, axis=1)
y_valid = numpy.argmax(y_valid, axis=1)
y_train = torch.from_numpy(y_train)
y_valid = torch.from_numpy(y_valid)

print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of X_valid: {X_valid.shape}")
print(f"Shape of y_valid: {y_valid.shape}")

print(f"Data type of X_train {X_train.dtype}")
print(f"Data type of y_train {y_train.dtype}")

batch_size = 32
epochs = 120
learning_rate = 0.0001

class_sample_counts = np.array([amounts_in_category[i] for i in range(len(amounts_in_category))])
weights = 1. / class_sample_counts
samples_weights = np.array([weights[t] for t in y_train.numpy()])

samples_weights = torch.from_numpy(samples_weights)
samples_weights = samples_weights.double()

sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weights, len(samples_weights))

train_loader = DataLoader(dataset=CustomDataset(X_train, y_train, True), batch_size=batch_size, sampler=sampler)
test_loader = DataLoader(dataset=CustomDataset(X_valid, y_valid, True), batch_size=batch_size, shuffle=True)

model = NeuralNetwork()
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

torch.save(model, 'model_fine_tuned.pth')

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_accu, label='Training Accuracy')
plt.plot(test_accu, label='Test Accuracy')
plt.title('Accuracy(epoch)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_loss, label='Training Loss')
plt.plot(test_loss, label='Test Loss')
plt.title('Loss(epoch)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


confusion_matrix_train = confusion_matrix(y_real_train, y_pred_train)
confusion_matrix_test = confusion_matrix(y_real_test, y_pred_test)

plt.figure(figsize=(12, 12))
sns.heatmap(confusion_matrix_train, annot=True, fmt='d', cmap='Blues')
plt.ylabel('Actual results')
plt.xlabel('Predicted results')
plt.title('Train set heatmap')
plt.show()

plt.figure(figsize=(12, 12))
sns.heatmap(confusion_matrix_test, annot=True, fmt='d', cmap='Blues')
plt.ylabel('Actual results')
plt.xlabel('Predicted results')
plt.title('Test set heatmap')
plt.show()

print("Classification report for train set")
print(classification_report(y_real_train, y_pred_train))

print("Classification report for test set")
print(classification_report(y_real_test, y_pred_test))





