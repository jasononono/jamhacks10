from classifier.util import Progress, Timer


DATA_SPLIT = 0.8
SEED = 67
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 500

timer = Timer()
progress = Progress(timer)
timer.reset()