import torch

from network import CNN

FILE = "src/model/py3-1.pth"

model = CNN()
weights = torch.load(FILE, weights_only = True)
model.load_state_dict(weights)
model.eval()

model.to("cpu")

torch.save(model.state_dict(), FILE)