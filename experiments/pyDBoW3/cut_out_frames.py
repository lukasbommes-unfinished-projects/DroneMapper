import cv2
import numpy as np


video = "../../phantom3-village-original/flight_truncated.MOV"
cut_idxs = [300, 2025, 1750, 3000, 3125, 4000]

cap = cv2.VideoCapture(video)

frame_idx = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    if frame_idx in cut_idxs:
        cv2.imwrite("frame_{}.png".format(frame_idx), frame)
        print("Cut out frame")

    frame_idx += 1
    print(frame_idx)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
