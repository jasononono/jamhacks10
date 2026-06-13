import cv2


camera = cv2.VideoCapture(0)
camera_width, camera_height = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
resolution = min(camera_width, camera_height)


success, frame = camera.read()
if success:
    print(type(frame))


# def recyclable()