import torch
import matplotlib.pyplot as plt
from torch import optim, nn
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sb

from consts import *


def train(model, device, dataloader, epochs):
    weights = torch.tensor(CLASS_WEIGHT).float().to(device)
    criterion = nn.CrossEntropyLoss(weight = weights)
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
    progress.log("training", epochs)
    progress.add_entry("loss")
    progress.add_entry("accuracy")

    cost = []

    try:

        for e in range(epochs):
            total_loss = 0
            total_correct = 0
            for i, o in dataloader:
                optimizer.zero_grad()
                prediction = model(i)
                loss = criterion(prediction, o)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                pred_classes = prediction.argmax(dim = 1).cpu()
                expected_classes = o.argmax(dim = 1).cpu()
                total_correct += ((pred_classes ^ expected_classes) == 0).sum()

            total_loss /= len(dataloader)
            progress.update_entry("loss", total_loss)
            progress.update_entry("accuracy", f"{total_correct / len(dataloader.dataset) * 100:.02f}%")
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
            output_classes = output.argmax(dim = 1).cpu().tolist()
            expected_classes = o.argmax(dim = 1).cpu().tolist()
            prediction.extend(output_classes)
            expected.extend(expected_classes)

    matrix = confusion_matrix(expected, prediction)
    sb.heatmap(matrix, annot = True, fmt = 'd', cmap = "Blues", ax = ax2)
    ax2.set_xlabel("predicted")
    ax2.set_ylabel("actual")

    plt.tight_layout()
    plt.show()

    ans = input("save model? -> ")
    if ans != "":
        torch.save(model.state_dict(), f"src/model/{ans}.pth")