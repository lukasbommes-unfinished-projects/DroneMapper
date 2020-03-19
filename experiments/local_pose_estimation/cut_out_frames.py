import os
import cv2
import numpy as np


video = "../../phantom3-village-original/flight_truncated.MOV"

cap = cv2.VideoCapture(video)

w = 1920
h = 1080
fx = 1184.51770
fy = 1183.63810
cx = 978.30778
cy = 533.85598
camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
dist_coeffs = np.array([-0.01581, 0.01052, -0.00075, 0.00245, 0.00000])

# precompute undistortion maps
mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, camera_matrix, (w, h), cv2.CV_32FC1)

frame_idx = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    frame = cv2.remap(frame, mapx, mapy, cv2.INTER_CUBIC)  # undistort

    cv2.imwrite(os.path.join("frames", "frame_{:04d}.png".format(frame_idx)), frame)

    frame_idx += 1
    print(frame_idx)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
