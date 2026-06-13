from PIL import Image, ImageOps
import numpy as np
from util import json_load
import pathlib


img_extensions = [".png", ".jpg", ".jpeg"]
split = 0.8


def load_set(directory):
    for f in directory.iterdir():
        if f.is_file() and f.suffix in img_extensions:
            img = Image.open(f)
            padded = ImageOps.pad(img, (256, 256), color = "white")
            yield np.array(padded.convert("RGB"))

def load(directory):
    classes = json_load("data.json")["classes"]
    inputs = []
    outputs = []

    for i, c in enumerate(classes):
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