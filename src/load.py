from PIL import Image, ImageOps
import numpy as np
import pathlib

from classifier.consts import *
            

def load_set(directory):
    for f in directory.iterdir():
        if f.is_file() and f.suffix in IMG_EXTENSIONS:
            img = Image.open(f)
            width, height = img.size
            if width < height:
                new_width = 128
                new_height = int(height * (128 / width))
            else:
                new_height = 128
                new_width = int(width * (128 / height))
            resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            x = (new_width - 128) // 2
            y = (new_height - 128) // 2
            cropped = resized.crop((x, y, x + 128, y + 128))
            yield np.array(cropped.convert("RGB"))

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