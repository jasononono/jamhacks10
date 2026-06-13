import torch
import matplotlib.pyplot as plt
from torch import optim, nn
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sb

from consts import *


def train(model, dataset, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
    progress.log("training", epochs)
    progress.add_entry("loss")

    cost = []

    try:

        for e in range(epochs):
            total_loss = 0
            for i, o in dataset:
                optimizer.zero_grad()
                prediction = model(i)
                loss = criterion(prediction, o)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            progress.update_entry("loss", total_loss / len(dataset))
            progress.update(e + 1)
            cost.append(total_loss)

    except KeyboardInterrupt:
        pass

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 8))

    ax1.plot(np.arange(1, len(cost) + 1), cost)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")

    prediction = []
    expected = []
    
    with torch.no_grad():
        for i, o in dataset:
            output = model(i)
            prediction.append(int(torch.argmax(output)))
            expected.append(int(torch.argmax(o)))

    matrix = confusion_matrix(prediction, expected)
    sb.heatmap(matrix, annot = True, fmt = 'd', cmap = "Blues", ax = ax2)

    plt.tight_layout()
    plt.show()