from classifier.util import Progress, Timer


CLASSES = 6
LABELS = ["BOTTLE", "CAN", "JUICE_BOX", "MILK_CARTON", "STYROFOAM", "UTENSIL"]
IMG_EXTENSIONS = [".png", ".jpg", ".jpeg"]
CLASS_WEIGHT = [3.655, 4.161, 3.629, 3.221, 0.652, 0.669] # w = (total # of images) / (# of classes * # of images in class)

DATA_SPLIT = 0.8
SEED = 67
LEARNING_RATE = 0.0004
BATCH_SIZE = 32
EPOCHS = 500

THRESHOLD = 20 # aim at face margin of error
PATIENCE = 80 # amount of ticks to wait before forced yeet

timer = Timer()
progress = Progress(timer)
timer.reset()