from PIL import Image, ImageOps
import numpy as np
import pathlib

from consts import *


def load_set(directory):
    for f in directory.iterdir():
        if f.is_file() and f.suffix in IMG_EXTENSIONS:
            img = Image.open(f)
            padded = ImageOps.pad(img, (128, 128), color = "white")
            yield np.array(padded.convert("RGB"))

def load(directory):
    inputs = []
    outputs = []

    for i, c in enumerate(LABELS):
        path = pathlib.Path(directory) / c
        count = 0
        for a in load_set(path):
            inputs.append(a)
            outputs.append(i)
            count += 1
        
        print(f"{c}: {count} images")

    print(f"total: {len(outputs)} images")
    return np.array(inputs), np.array(outputs)


inputs, outputs = load("dataset")
np.save("data/inputs.npy", inputs)
np.save("data/outputs.npy", outputs)