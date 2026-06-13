import cv2, time
from PIL import Image
import numpy as np

import torch
import torch.nn.functional as f

from classifier.network import CNN
from classifier.consts import *


camera = cv2.VideoCapture(0)
camera_width, camera_height = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
resolution = min(camera_width, camera_height)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

model = CNN().to(device)
weights = torch.load("src/model/py3-2.pth", weights_only = True)
model.load_state_dict(weights)
model.eval()

cascade = cv2.CascadeClassifier("src/external/haarcascade_frontalface_default.xml")


def on_button_press():
    success, frame = camera.read()
    if not success:
        return

    if recyclable(frame):
        deposit()
    else:
        center = camera_width // 2
        tick = 0

        while tick < PATIENCE:
            face = cascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if len(face) == 0:
                tick += 1
                continue
            target = (face[0][0] + face[0][2]) // 2
            if abs(target - center) < THRESHOLD:
                break
            elif target > center:
                stepper_right()
            else:
                stepper_left()

        yeet()

    return frame


def recyclable(frame):
    image = torch.tensor(crop(frame).transpose(2, 0, 1), dtype = torch.float32).to(device) / 255
    x_min = image.min()
    x_max = image.max()
    image = (image - x_min) / (x_max - x_min)

    prediction = f.softmax(model(image.unsqueeze(0)), dim = 1)
    c = torch.argmax(prediction, axis = 1)
    print(LABELS[c])
    return c in (0, 1)

def crop(frame):
    img = Image.fromarray(frame, "RGB")
    x = (camera_width - resolution) // 2
    y = (camera_height - resolution) // 2
    cropped = img.crop((x, y, x + resolution, y + resolution))
    return np.array(cropped.resize((128, 128)))


# EXTERNS
def yeet():
    # print("yeet")
    pass

def deposit():
    # print("deposit")
    pass

def stepper_right():
    # print("stepper_right")
    pass

def stepper_left():
    # print("stepper_left")
    pass


# TESTING

# print()
# while True:
#     frame = on_button_press()
#     cv2.imshow("picky yeeter", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break


frame = on_button_press()
camera.release()
cv2.destroyAllWindows()