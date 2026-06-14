import torch
import matplotlib.pyplot as plt
from torch import optim, nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sb

from classifier.consts import *


def train(model, device, dataset, epochs):
    dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True)
    criterion = nn.BCELoss()
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
                loss = criterion(prediction.squeeze(1), o)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                for k in range(len(o)):
                    if round(o[k].item()) == round(prediction[k].item()):
                        total_correct += 1

            total_loss /= len(dataloader)
            progress.update_entry("loss", format(total_loss, ".04f"))
            progress.update_entry("accuracy", f"{total_correct / len(dataloader.dataset) * 100:.02f}%")
            progress.update(e + 1)
            cost.append(total_loss)
            accuracy.append(total_correct / len(dataloader.dataset) * 100)

    except KeyboardInterrupt:
        pass

    return cost, accuracy