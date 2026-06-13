from util import Progress, Timer


CLASSES = 6
LABELS = ["BOTTLE", "CAN", "JUICE_BOX", "MILK_CARTON", "STYROFOAM", "UTENSIL"]
IMG_EXTENSIONS = [".png", ".jpg", ".jpeg"]

SEED = 42
LEARNING_RATE = 0.005

timer = Timer()
progress = Progress(timer)
timer.reset()