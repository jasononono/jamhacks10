import cv2, time
from PIL import Image
import numpy as np

import torch
import torch.nn.functional as f

from network import CNN
from consts import *


camera = cv2.VideoCapture(0)
camera_width, camera_height = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
resolution = min(camera_width, camera_height)

model = CNN()
weights = torch.load("src/model/py3-1.pth", weights_only = True)
model.load_state_dict(weights)
model.eval()


def on_button_press():
    success, frame = camera.read()
    if not success:
        return
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = torch.tensor(crop(frame).transpose(2, 0, 1), dtype = torch.float32) / 255

    prediction = f.softmax(model(image.unsqueeze(0)))
    c = torch.argmax(prediction, axis = 1)
    print(LABELS[c])
    return c in (0, 1)

def crop(frame):
    img = Image.fromarray(frame, "RGB")
    x = (camera_width - resolution) // 2
    y = (camera_height - resolution) // 2
    cropped = img.crop((x, y, x + resolution, y + resolution))
    img.show()
    return np.array(cropped.resize((128, 128)))
        

time.sleep(1)
print(on_button_press())