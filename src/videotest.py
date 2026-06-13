import cv2

cam_int = cv2.VideoCapture(0)
cam_ext = cv2.VideoCapture(2)

while True:
    success, frame_int = cam_int.read()
    if not success:
        print("cam_int failed to read")
        break
    success, frame_ext = cam_ext.read()
    if not success:
        print("cam_ext failed to read")
        break

    cv2.imshow("cam interior", frame_int)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam_int.release()
cam_ext.release()