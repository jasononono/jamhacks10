import cv2
import time
from PIL import Image
import numpy as np
import sys

import torch
from torchvision.transforms import v2

from classifier.network import CNN

# Jetson.GPIO is hardware specific; keep it as-is on the Jetson device
import Jetson.GPIO as GPIO

# GLOBALS #

THRESHOLD = 20  # aim at face margin of error
PATIENCE = 80   # amount of ticks to wait before forced yeet

STEP_PIN = 12  # stepper 1 STEP
DIR_PIN = 16   # stepper 1 DIR

STEP2_PIN = 11  # stepper 2 STEP
DIR2_PIN = 15   # stepper 2 DIR

# Stepper1 rotation limit (units: steps). Calibrate these for your hardware.
STEPPER1_MIN = 0
STEPPER1_MAX = 2000  # example limit; change to your calibrated max
STEPPER1_POS = 0     # current step position (0..MAX)

# GPIO init
GPIO.setmode(GPIO.BOARD)
GPIO.setup(STEP_PIN, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(DIR_PIN, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(STEP2_PIN, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(DIR2_PIN, GPIO.OUT, initial=GPIO.LOW)


def move_motor(steps, direction, motor=1, speed_delay=0.001):
    """
    Move the selected motor a number of steps in the given direction.
    motor=1 -> uses STEP_PIN/DIR_PIN and is subject to rotation limits.
    motor=2 -> uses STEP2_PIN/DIR2_PIN (no limit applied).
    direction -> GPIO.HIGH or GPIO.LOW
    """
    global STEPPER1_POS

    if motor == 1:
        step_pin = STEP_PIN
        dir_pin = DIR_PIN

        # compute signed step delta (positive for GPIO.HIGH if we treat HIGH as +)
        delta = int(steps) if direction == GPIO.HIGH else -int(steps)

        # clamp to limits
        new_pos = STEPPER1_POS + delta
        if new_pos > STEPPER1_MAX:
            allowed_delta = STEPPER1_MAX - STEPPER1_POS
        elif new_pos < STEPPER1_MIN:
            allowed_delta = STEPPER1_MIN - STEPPER1_POS
        else:
            allowed_delta = delta

        steps_to_move = abs(int(allowed_delta))
        actual_direction = GPIO.HIGH if allowed_delta >= 0 else GPIO.LOW

        if steps_to_move == 0:
            # nothing to do (would exceed limits)
            print(f"Stepper1 limit reached (pos={STEPPER1_POS}). Requested {delta} steps; allowed 0.")
            return

        GPIO.output(dir_pin, actual_direction)
        for _ in range(steps_to_move):
            GPIO.output(step_pin, GPIO.HIGH)
            time.sleep(speed_delay)
            GPIO.output(step_pin, GPIO.LOW)
            time.sleep(speed_delay)

        # update position
        STEPPER1_POS += allowed_delta
        print(f"Stepper1 moved {allowed_delta} steps -> pos={STEPPER1_POS}")

    elif motor == 2:
        step_pin = STEP2_PIN
        dir_pin = DIR2_PIN
        GPIO.output(dir_pin, direction)
        for _ in range(int(steps)):
            GPIO.output(step_pin, GPIO.HIGH)
            time.sleep(speed_delay)
            GPIO.output(step_pin, GPIO.LOW)
            time.sleep(speed_delay)
    else:
        raise ValueError("Unknown motor id. Use 1 or 2.")


def move_right(steps, speed_delay=0.001, motor=1):
    # choose GPIO.HIGH as "right" by convention
    move_motor(steps, direction=GPIO.HIGH, motor=motor, speed_delay=speed_delay)


def move_left(steps, speed_delay=0.001, motor=1):
    move_motor(steps, direction=GPIO.LOW, motor=motor, speed_delay=speed_delay)


def charge_launch():
    # use motor 2 to launch
    steps = 400
    direction = GPIO.LOW
    speed_delay = 0.001

    # Set direction for motor 2
    GPIO.output(DIR2_PIN, direction)
    for _ in range(int(steps)):
        GPIO.output(STEP2_PIN, GPIO.HIGH)
        time.sleep(speed_delay)
        GPIO.output(STEP2_PIN, GPIO.LOW)
        time.sleep(speed_delay)


# CAM / CNN INIT

def load_model(device):
    model = CNN().to(device)
    # load a state dict file saved with torch.save(model.state_dict())
    weights = torch.load("src/model/py5-5.pth")
    model.load_state_dict(weights)
    model.eval()
    return model


transpose = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])


def crop(frame, cam_int_width, cam_int_height, cam_int_resolution):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), "RGB")
    x = (cam_int_width - cam_int_resolution) // 2
    y = (cam_int_height - cam_int_resolution) // 2
    cropped = img.crop((x, y, x + cam_int_resolution, y + cam_int_resolution))
    return np.array(cropped.resize((128, 128)))


def recyclable(frame, model, device, cam_int_width, cam_int_height, cam_int_resolution):
    image_np = crop(frame, cam_int_width, cam_int_height, cam_int_resolution)
    image = torch.tensor(image_np.transpose(2, 0, 1)).to(device)
    with torch.no_grad():
        # apply the same transforms pipeline used at training (transpose)
        inp = transpose(image).unsqueeze(0)
        prediction = round(model(inp).item())
    print("RECYCLE" if prediction else "DO NOT RECYCLE")
    return bool(prediction)


def yeet():
    print("yeet")
    charge_launch()


def deposit():
    print("deposit")
    # move stepper1 to deposit position (use motor=1 so limits apply)
    move_right(steps=200, speed_delay=0.0005, motor=1)
    time.sleep(1)
    charge_launch()
    move_left(steps=200, speed_delay=0.0005, motor=1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    model = load_model(device)

    cascade = cv2.CascadeClassifier("src/external/haarcascade_frontalface_default.xml")

    cam_int = cv2.VideoCapture(2)
    cam_ext = cv2.VideoCapture(0)

    try:
        if not cam_int.isOpened():
            print("cam_int failed to open")
            return
        if not cam_ext.isOpened():
            print("cam_ext failed to open")
            return

        cam_int_width = int(cam_int.get(cv2.CAP_PROP_FRAME_WIDTH))
        cam_int_height = int(cam_int.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cam_int_resolution = min(cam_int_width, cam_int_height)

        cam_ext_width = int(cam_ext.get(cv2.CAP_PROP_FRAME_WIDTH))
        cam_ext_height = int(cam_ext.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cam_ext_resolution = min(cam_ext_width, cam_ext_height)

        # read internal camera frame for classification
        success, frame_int = cam_int.read()
        if not success:
            print("cam_int failed to read")
            return

        if recyclable(frame_int, model, device, cam_int_width, cam_int_height, cam_int_resolution):
            cam_int.release()
            cam_ext.release()
            deposit()
            return

        # otherwise attempt to center using external camera
        center = cam_ext_width // 2
        tick = 0

        while tick < PATIENCE:
            success, frame_ext = cam_ext.read()
            if not success:
                tick += 1
                continue

            # cascade works best on gray images
            gray = cv2.cvtColor(frame_ext, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray)
            print("tick:", tick)
            cv2.imshow("camera external", frame_ext)

            if len(faces) == 0:
                tick += 1
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            x, y, w, h = faces[0]
            target = x + w // 2
            if abs(target - center) < THRESHOLD:
                break
            elif target > center:
                # move stepper1 right (motor=1)
                move_right(5, motor=1)
            else:
                move_left(5, motor=1)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cam_int.release()
        cam_ext.release()
        cv2.destroyAllWindows()
        yeet()

    finally:
        # always release and cleanup GPIO
        if cam_int.isOpened():
            cam_int.release()
        if cam_ext.isOpened():
            cam_ext.release()
        cv2.destroyAllWindows()
        GPIO.cleanup()


if __name__ == "__main__":
    main()