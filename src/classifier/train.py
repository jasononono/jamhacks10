import torch
import matplotlib.pyplot as plt
from torch import optim, nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sb

from consts import *


def train(model, device, dataset, epochs):
    dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True)
    weights = torch.tensor(CLASS_WEIGHT).float().to(device)
    criterion = nn.CrossEntropyLoss(weight = weights)
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE, weight_decay = 0.0001)
    progress.log("training", epochs)
    progress.add_entry("loss")
    progress.add_entry("accuracy")

    cost = []
    accuracy = []

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
            progress.update_entry("loss", format(total_loss, ".04f"))
            progress.update_entry("accuracy", f"{total_correct / len(dataloader.dataset) * 100:.02f}%")
            progress.update(e + 1)
            cost.append(total_loss)
            accuracy.append(total_correct / len(dataloader.dataset) * 100)

    except KeyboardInterrupt:
        pass

    return cost, accuracy

def test(model, device, dataset, ax):
    dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True)
    weights = torch.tensor(CLASS_WEIGHT).float().to(device)
    criterion = nn.CrossEntropyLoss(weight = weights)
    prediction = []
    expected = []

    model.eval()
    with torch.no_grad():
        total_loss = 0
        total_correct = 0
        for i, o in dataloader:
            output = model(i)
            output_classes = output.argmax(dim = 1).cpu()
            expected_classes = o.argmax(dim = 1).cpu()
            loss = criterion(output, o)
            total_loss += loss.item()
            prediction.extend(output_classes.tolist())
            expected.extend(expected_classes.tolist())

            total_correct += ((output_classes ^ expected_classes) == 0).sum()

    matrix = confusion_matrix(expected, prediction)
    sb.heatmap(matrix, annot = True, fmt = 'd', cmap = "Blues", ax = ax)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")

    print(f"loss = {total_loss:.04f}")
    print(f"accuracy = {total_correct / len(dataloader.dataset) * 100:.02f}%")