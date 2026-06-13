import Jetson.GPIO as GPIO
import time
import sys

STEP_PIN = 12  # Connects to STEP on TMC2208
DIR_PIN = 16  # Connects to DIR on TMC2208


def setup_gpio():

    GPIO.setmode(GPIO.BOARD)

    # Configure pins as outputs and set them to LOW initially
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


if __name__ == "__main__":
    try:
        setup_gpio()


        move_motor(steps=200, direction=GPIO.HIGH, speed_delay=0.002)

        # Pause for 1 second
        time.sleep(1)

        move_motor(steps=200, direction=GPIO.LOW, speed_delay=0.002)



    finally:

        GPIO.cleanup()
