import time

import board
import digitalio


# Motor 1 pins
STEP_PIN = board.D12
DIR_PIN = board.D16

# Motor 2 pins
STEP2_PIN = board.D11
DIR2_PIN = board.D15


def setup_pin(pin, direction=digitalio.Direction.OUTPUT, value=False):
    dio = digitalio.DigitalInOut(pin)
    dio.direction = direction
    dio.value = value
    return dio


# Configure outputs
step1 = setup_pin(STEP_PIN, value=False)
dir1 = setup_pin(DIR_PIN, value=False)

step2 = setup_pin(STEP2_PIN, value=False)
dir2 = setup_pin(DIR2_PIN, value=False)


def step_motor(step_pin, dir_pin, steps, direction, speed_delay=0.001):
    """
    Send single STEP pulses to one motor driver.
    direction: False/True (pick which one maps to right/left on your hardware)
    """
    dir_pin.value = direction

    for _ in range(int(steps)):
        # single-step pulse
        step_pin.value = True
        time.sleep(speed_delay)
        step_pin.value = False
        time.sleep(speed_delay)


def move_right(steps, speed_delay=0.001):
    # Adjust True/False if your motor turns the opposite direction
    step_motor(step1, dir1, steps, direction=True, speed_delay=speed_delay)


def move_left(steps, speed_delay=0.001):
    # Opposite of move_right
    step_motor(step1, dir1, steps, direction=False, speed_delay=speed_delay)


def charge_launch(steps=400, speed_delay=0.001):
    """
    Move both stepper motors together, in single-step mode.
    """
    # Set both directions the same
    dir1.value = False
    dir2.value = False

    for _ in range(int(steps)):
        # Single step pulse for both motors at the same time
        step1.value = True
        step2.value = True
        time.sleep(speed_delay)

        step1.value = False
        step2.value = False
        time.sleep(speed_delay)


if __name__ == "__main__":
    try:
        move_right(steps=400, speed_delay=0.0005)
        time.sleep(1)
        move_left(steps=400, speed_delay=0.0005)
        time.sleep(1)
        charge_launch()
    finally:
        # Release GPIO
        step1.deinit()
        dir1.deinit()
        step2.deinit()
        dir2.deinit()