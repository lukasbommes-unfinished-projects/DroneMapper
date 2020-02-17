from random import shuffle

import numpy as np
import cv2

from ssc import ssc

# camera parameters
w = 1920
h = 1080
fx = 1184.51770
fy = 1183.63810
cx = 978.30778
cy = 533.85598
camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
dist_coeffs = np.array([-0.01581, 0.01052, -0.00075, 0.00245, 0.00000])

# read video
video_file = "phantom3-village-original/flight.MOV"
cap = cv2.VideoCapture(video_file)

# precompute undistortion maps
new_camera_matrix = camera_matrix
#new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), alpha=0)
mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), cv2.CV_32FC1)

cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("frame", 1600, 900)

orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

keyframes = []
featurepoints = []

# while(True):
#     # read and undistort frames
#     ret, frame = cap.read()
#     if not ret:
#         print("Could not read frame. Stopping.")
#         break
#
#     frame = cv2.remap(frame, mapx, mapy, cv2.INTER_CUBIC)
#
#     # detect keypoints and compute descriptors for matching
#     kp, des = orb.detectAndCompute(frame, None)
#     featurepoints.append((kp, des))
#
#     # generate matches with last key frame
#     #matches = bf.match(des, des2)
#
#     frame = cv2.drawKeypoints(frame, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#
#     cv2.imshow("frame", frame)
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         break

cv2.namedWindow("frame1", cv2.WINDOW_NORMAL)
cv2.resizeWindow("frame1", 1600, 900)
cv2.namedWindow("frame2", cv2.WINDOW_NORMAL)
cv2.resizeWindow("frame2", 1600, 900)

# ret1, frame1 = cap.read()
# [cap.read() for i in range(600)]
# ret2, frame2 = cap.read()
# frame1 = cv2.remap(frame1, mapx, mapy, cv2.INTER_CUBIC)
# frame2 = cv2.remap(frame2, mapx, mapy, cv2.INTER_CUBIC)

# kp1, des1 = orb.detectAndCompute(frame1, None)
# kp2, des2 = orb.detectAndCompute(frame2, None)
#
# frame1 = cv2.drawKeypoints(frame1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# frame2 = cv2.drawKeypoints(frame2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#
# matches = bf.match(des1, des2)
# matches = sorted(matches, key = lambda x:x.distance)
# print("Found {} matches".format(len(matches)))
#
# frame = cv2.drawMatches(frame1, kp1, frame2, kp2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)




fast = cv2.FastFeatureDetector_create(threshold=12)
num_ret_points = 3000
tolerance = 0.1

ret1, frame1 = cap.read()
[cap.read() for i in range(50)]
ret2, frame2 = cap.read()
frame1 = cv2.remap(frame1, mapx, mapy, cv2.INTER_CUBIC)
frame2 = cv2.remap(frame2, mapx, mapy, cv2.INTER_CUBIC)

kp1 = fast.detect(frame1, None)
kp2 = fast.detect(frame2, None)
kp1 = sorted(kp1, key = lambda x:x.response)
kp2 = sorted(kp2, key = lambda x:x.response)
print("Found {} keypoint in frame 1".format(len(kp1)))
print("Found {} keypoint in frame 2".format(len(kp2)))
#shuffle(kp1)  # simulating sorting by score with random shuffle
#shuffle(kp2)

kp_id = 1
print(kp1[kp_id].angle)
print(kp1[kp_id].class_id)
print(kp1[kp_id].octave)
print(kp1[kp_id].pt)
print(kp1[kp_id].response)
print(kp1[kp_id].size)
kp_converted = cv2.KeyPoint_convert(kp1)
print(kp_converted)

kp1 = ssc(kp1, num_ret_points, tolerance, frame1.shape[1], frame1.shape[0])
kp2 = ssc(kp2, num_ret_points, tolerance, frame2.shape[1], frame2.shape[0])
kp1, des1 = orb.compute(frame1, kp1)
kp2, des2 = orb.compute(frame2, kp2)
frame1 = cv2.drawKeypoints(frame1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
frame2 = cv2.drawKeypoints(frame2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

matches = bf.match(des1, des2)
matches = sorted(matches, key = lambda x:x.distance)
print("Found {} matches".format(len(matches)))

frame = cv2.drawMatches(frame1, kp1, frame2, kp2, matches[:100], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


while(True):

    # ret1, frame1 = cap.read()
    # frame1 = cv2.remap(frame1, mapx, mapy, cv2.INTER_CUBIC)
    # keypoints = fast.detect(frame1, None)
    # #frame1 = cv2.drawKeypoints(frame1, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # shuffle(keypoints)  # simulating sorting by score with random shuffle
    #
    # selected_keypoints = ssc(keypoints, num_ret_points, tolerance, frame1.shape[1], frame1.shape[0])
    # #selected_keypoints, des = orb.compute(frame1, selected_keypoints)
    # frame1 = cv2.drawKeypoints(frame1, selected_keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow("frame1", frame1)
    cv2.imshow("frame2", frame2)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
