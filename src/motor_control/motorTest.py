import Jetson.GPIO as GPIO
import time
import sys

STEP_PIN = 12  # Connects to STEP on TMC2208
DIR_PIN = 16   # Connects to DIR on TMC2208

def setup_gpio():

    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(STEP_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(DIR_PIN, GPIO.OUT, initial=GPIO.LOW)

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
    move_motor(149.94, direction=GPIO.LOW, speed_delay=0.001)
if __name__ == "__main__":
    try:
        setup_gpio()

        move_right(steps=200, speed_delay=0.002)
        time.sleep(1)
        move_left(steps=200, speed_delay=0.002)

    finally:
        GPIO.cleanup()