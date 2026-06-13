import Jetson.GPIO as GPIO
import time
import sys

STEP_PIN = 12  # Connects to STEP on TMC2208
DIR_PIN = 16   # Connects to DIR on TMC2208

STEP2_PIN = 11
DIR2_PIN = 15
def setup_gpio():

    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(STEP_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(DIR_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(STEP2_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(DIR2_PIN, GPIO.OUT, initial=GPIO.LOW)

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
    steps = 200
    direction = GPIO.LOW
    speed_delay = 0.001
    
    # Set direction for both steppers
    GPIO.output(DIR2_PIN, direction)
    
    # Pulse both steppers simultaneously
    for _ in range(int(steps)):
        GPIO.output(STEP2_PIN, GPIO.HIGH)
        time.sleep(speed_delay)
        GPIO.output(STEP2_PIN, GPIO.LOW)
        time.sleep(speed_delay)
if __name__ == "__main__":
    try:
        setup_gpio()

        move_right(steps=200, speed_delay=0.002)
        time.sleep(1)
        move_left(steps=200, speed_delay=0.002)
        time.sleep(1)
        charge_launch()

    finally:
        GPIO.cleanup()