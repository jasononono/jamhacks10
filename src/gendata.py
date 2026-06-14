import cv2
import numpy as np
from PIL import Image

cam_ext = cv2.VideoCapture(0)
cam_ext_width, cam_ext_height = int(cam_ext.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cam_ext.get(cv2.CAP_PROP_FRAME_HEIGHT))
cam_ext_resolution = min(cam_ext_width, cam_ext_height)

recyclable = False
inputs = []
outputs = []

while True:
    success, frame_int = cam_ext.read()
    if not success:
        print("cam_ext failed to read")
        break

    cv2.imshow("cam exterior", frame_int)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('1'):
        recyclable = False
        print("recyclable: FALSE")
    elif key == ord('2'):
        recyclable = True
        print("recyclable: TRUE")
    elif key == ord(' '):
        arr = np.array(cv2.cvtColor(frame_int, cv2.COLOR_BGR2RGB))
        x = (cam_ext_width - cam_ext_resolution) // 2
        y = (cam_ext_height - cam_ext_resolution) // 2
        cropped = arr[y:y + cam_ext_resolution, x:x + cam_ext_resolution]
        img = Image.fromarray(cropped, "RGB")
        inputs.append(np.array(img.resize((128, 128))))
        outputs.append(int(recyclable))
    

cam_ext.release()

np.save("data/inputs.npy", np.array(inputs))
np.save("data/outputs.npy", np.array(outputs))
print(np.array(inputs).shape)
print(np.array(outputs).shape)