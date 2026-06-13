import cv2, time
from PIL import Image
import numpy as np

import torch
import torch.nn.functional as f
from torchvision.transforms import v2

from classifier.network import CNN
from classifier.consts import *

import Jetson.GPIO as GPIO
import time
import sys

STEP_PIN = 12  # Connects to STEP on TMC2208
DIR_PIN = 16   # Connects to DIR on TMC2208

STEP2_PIN = 11
DIR2_PIN = 15

INPUT_PIN = 13


def setup_gpio():
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(STEP_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(DIR_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(STEP2_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(DIR2_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(INPUT_PIN, GPIO.IN)

def move_motor(steps, direction, speed_delay=0.001):
    # Set the direction pin
    GPIO.output(DIR_PIN, direction)

    # Pulse generation loop
    for _ in range(steps):
        GPIO.output(STEP_PIN, GPIO.HIGH)
        time.sleep(speed_delay)
        GPIO.output(STEP_PIN, GPIO.LOW)
        time.sleep(speed_delay)


def move_right(steps, speed_delay=0.001):
    move_motor(steps, direction=GPIO.HIGH, speed_delay=speed_delay)


def move_left(steps, speed_delay=0.001):
    move_motor(steps, direction=GPIO.LOW, speed_delay=speed_delay)


def charge_launch():
    direction = GPIO.LOW
    speed_delay = 0.001

    # Set direction for both steppers
    GPIO.output(DIR2_PIN, direction)

    # Pulse both steppers simultaneously
    for _ in range(283):
        GPIO.output(STEP2_PIN, GPIO.HIGH)
        time.sleep(speed_delay)
        GPIO.output(STEP2_PIN, GPIO.LOW)
        time.sleep(speed_delay)

    time.sleep(1)
    for _ in range(283):
        GPIO.output(STEP2_PIN, GPIO.HIGH)
        time.sleep(speed_delay)
        GPIO.output(STEP2_PIN, GPIO.LOW)
        time.sleep(speed_delay)
    time.sleep(1)
    for _ in range(117):
        GPIO.output(STEP2_PIN, GPIO.HIGH)
        time.sleep(speed_delay)
        GPIO.output(STEP2_PIN, GPIO.LOW)
        time.sleep(speed_delay)

cam_int = cv2.VideoCapture(0)
cam_int_width, cam_int_height = int(cam_int.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cam_int.get(cv2.CAP_PROP_FRAME_HEIGHT))
cam_int_resolution = min(cam_int_width, cam_int_height)
cam_ext = cv2.VideoCapture(1)
cam_ext_width, cam_ext_height = int(cam_ext.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cam_ext.get(cv2.CAP_PROP_FRAME_HEIGHT))
cam_ext_resolution = min(cam_ext_width, cam_ext_height)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

model = CNN().to(device)
weights = torch.load("src/model/py4-0.pth", weights_only = True)
model.load_state_dict(weights)
model.eval()

cascade = cv2.CascadeClassifier("src/external/haarcascade_frontalface_default.xml")

transpose = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale = True),
    v2.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
])


def on_button_press():
    success, frame_int = cam_int.read()
    if not success:
        return
    success, frame_ext = cam_ext.read()
    if not success:
        return

    if recyclable(frame_int):
        deposit()
    else:
        center = cam_ext_width // 2
        tick = 0

        while tick < PATIENCE:
            face = cascade.detectMultiScale(cv2.cvtColor(frame_ext, cv2.COLOR_BGR2RGB))
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

    return frame_int


def recyclable(frame):
    image = torch.tensor(crop(frame).transpose(2, 0, 1)).to(device)

    with torch.no_grad():
        prediction = f.softmax(model(transpose(image).unsqueeze(0)), dim = 1)
    c = torch.argmax(prediction, axis = 1)
    print(LABELS[c])
    return c in (0, 1)

def crop(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), "RGB")
    x = (cam_int_width - cam_int_resolution) // 2
    y = (cam_int_height - cam_int_resolution) // 2
    cropped = img.crop((x, y, x + cam_int_resolution, y + cam_int_resolution))
    return np.array(cropped.resize((128, 128)))


# EXTERNS
def yeet():
    setup_gpio()

    charge_launch()
    pass

def deposit():
    move_right(steps=200, speed_delay=0.0005)
    time.sleep(1)
    charge_launch()
    pass

def stepper_right():
    move_right(steps=50, speed_delay=0.0005)
    pass

def stepper_left():
    move_left(steps=50, speed_delay=0.0005)
    pass


# TESTING


# print()
# while True:
#     success, frame = cam_int.read()
#     if not success: continue
#     recyclable(frame)
#     cv2.imshow("picky yeeter", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break


frame = on_button_press()


cam_int.release()
cv2.destroyAllWindows()