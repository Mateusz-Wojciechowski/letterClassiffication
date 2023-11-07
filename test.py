import numpy as np
import matplotlib.pyplot as plt
import torch

def test_single_image(model, x_test):
    x_test_copy = x_test.flatten()
    x_test_copy = torch.from_numpy(x_test_copy).float().unsqueeze(0)

    model.eval()
    with torch.no_grad():
        pred = model(x_test_copy)
        _, predicted_label = torch.max(pred, 1)
        predicted_label = predicted_label.item()

        print(f"This image is likely {chr(ord('A') + predicted_label)}")
        plt.imshow(x_test, cmap='grey')
        plt.show()


X_test = np.load('X_train.npy/X_train.npy')
model = torch.load('model_fine_tuned.pth')

for x in X_test:
    test_single_image(model, x)
    input()
