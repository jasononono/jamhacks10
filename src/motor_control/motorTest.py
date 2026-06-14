import time

from adafruit_motor import stepper
from adafruit_motorkit import MotorKit


MOTOR_KIT_ADDRESS = 0x60


def setup_motor_kit():
    return MotorKit(i2c=board.I2C(), address=MOTOR_KIT_ADDRESS)


def step_motor(motor, steps, direction, speed_delay=0.001):
    for _ in range(int(steps)):
        motor.onestep(direction=direction, style=stepper.SINGLE)
        time.sleep(speed_delay)


def move_right(kit, steps, speed_delay=0.001):
    step_motor(kit.stepper1, steps, stepper.FORWARD, speed_delay=speed_delay)


def move_left(kit, steps, speed_delay=0.001):
    step_motor(kit.stepper1, steps, stepper.BACKWARD, speed_delay=speed_delay)


def charge_launch(kit, steps=400, speed_delay=0.001):
    for _ in range(int(steps)):
        kit.stepper1.onestep(direction=stepper.FORWARD, style=stepper.SINGLE)
        kit.stepper2.onestep(direction=stepper.FORWARD, style=stepper.SINGLE)
        time.sleep(speed_delay)


if __name__ == "__main__":
    kit = setup_motor_kit()

    try:
        move_right(kit, steps=400, speed_delay=0.0005)
        time.sleep(1)
        move_left(kit, steps=400, speed_delay=0.0005)
        time.sleep(1)
        charge_launch(kit)
    finally:
        kit.stepper1.release()
        kit.stepper2.release()
