import torch
import matplotlib.pyplot as plt
from torch import optim, nn
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sb

from consts import *


def train(model, dataloader, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
    progress.log("training", epochs)
    progress.add_entry("loss")

    cost = []

    try:

        for e in range(epochs):
            total_loss = 0
            for i, o in dataloader:
                optimizer.zero_grad()
                prediction = model(i)
                loss = criterion(prediction, o)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            total_loss /= len(dataloader)
            progress.update_entry("loss", total_loss)
            progress.update(e + 1)
            cost.append(total_loss)

    except KeyboardInterrupt:
        pass

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 8))

    ax1.plot(cost)
    ax1.set_xlabel("epoch")
    ax1.set_xticks(range(1, len(cost) + 1))
    ax1.set_ylabel("loss")

    prediction = []
    expected = []

    model.eval()
    with torch.no_grad():
        for i, o in dataloader:
            output = model(i)
            prediction.extend(output.argmax(dim = 1).cpu().tolist())
            expected.extend(o.argmax(dim = 1).cpu().tolist())

    matrix = confusion_matrix(expected, prediction)
    sb.heatmap(matrix, annot = True, fmt = 'd', cmap = "Blues", ax = ax2)
    ax2.set_xlabel("predicted")
    ax2.set_ylabel("actual")

    plt.tight_layout()
    plt.show()

    ans = input("save model? -> ")
    if ans != "":
        torch.save(model.state_dict(), f"model/{ans}.pth")