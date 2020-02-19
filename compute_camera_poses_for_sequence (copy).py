import numpy as np
import cv2

from ssc import ssc

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pytransform3d.rotations import *

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

orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
fast = cv2.FastFeatureDetector_create(threshold=12)
num_ret_points = 3000
tolerance = 0.1
num_matches = 50


def extract_kp_des(current_frame, fast, orb, draw=True):
    kp = fast.detect(current_frame, None)
    kp = sorted(kp, key = lambda x:x.response, reverse=True)
    kp = ssc(kp, num_ret_points, tolerance, current_frame.shape[1], current_frame.shape[0])
    kp, des = orb.compute(current_frame, kp)
    if draw:
        current_frame = cv2.drawKeypoints(current_frame, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return kp, des, current_frame


def match(last_keyframe, current_frame, bf, last_des, des, last_kp, kp, draw=True):
    matches = bf.match(last_des, des)
    matches = sorted(matches, key = lambda x:x.distance)
    print("Found {} matches of current frame with last frame".format(len(matches)))
    # get matched keypoints
    last_pts = np.array([last_kp[m.queryIdx].pt for m in matches[:num_matches]]).reshape(1, -1, 2)
    current_pts = np.array([kp[m.trainIdx].pt for m in matches[:num_matches]]).reshape(1, -1, 2)
    if draw:
        match_frame = cv2.drawMatches(last_keyframe, last_kp, current_frame, kp, matches[:num_matches], None, matchColor=(0, 0, 0), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return matches, last_pts, current_pts, match_frame


cv2.namedWindow("last_frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("last_frame", 1600, 900)
cv2.namedWindow("current_frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("current_frame", 1600, 900)
cv2.namedWindow("match_frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("match_frame", 1600, 900)

step_wise = True

# 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# #ax.scatter(last_pts.reshape(-1, 2)[:, 0], last_pts.reshape(-1, 2)[:, 1], np.zeros(last_pts.shape[1]))
# #ax.scatter(current_pts.reshape(-1, 2)[:, 0], current_pts.reshape(-1, 2)[:, 1], np.zeros(current_pts.shape[1]))
# ax.scatter(triangulatedPoints[:, 0], triangulatedPoints[:, 1], triangulatedPoints[:, 2])
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.set_zlabel("z")
# plt.legend(["frame1", "frame2", "triangulated"])
# plt.show()

# discard intial part of video
[cap.read() for i in range(1000)]

last_kf = {
    "frame": None,
    "key_points": None,
    "descriptors": None,
}
match_frame = None

Rs = []
ts = []
map_points = []

frame_idx = 0

fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("z")

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.set_xlabel("x")
ax2.set_ylabel("y")


while(True):

    print("frame", frame_idx)

    retc, current_frame = cap.read()
    current_frame = cv2.remap(current_frame, mapx, mapy, cv2.INTER_CUBIC)

    kp, des, current_frame = extract_kp_des(current_frame, fast, orb, draw=True)
    print("Found {} keypoint in current frame".format(len(kp)))

    if frame_idx == 0:
        last_kf["frame"] = current_frame
        last_kf["key_points"] = kp
        last_kf["descriptors"] = des
        frame_idx += 1
        continue

    matches, last_pts, current_pts, match_frame = match(last_kf["frame"], current_frame, bf, last_kf["descriptors"], des, last_kf["key_points"], kp, draw=True)

    # recover camera pose from point correspondences
    essential_mat, _ = cv2.findEssentialMat(last_pts, current_pts, camera_matrix, method=cv2.RANSAC)
    retval, R, t, mask = cv2.recoverPose(essential_mat, last_pts, current_pts, camera_matrix)
    Rs.append(R)
    ts.append(t)
    map_points.append(current_pts)

    print("R", R)
    print("t", t)
    print(retval)

    plot_basis(ax1, R, t.reshape(3,))

    ax2.scatter(last_pts.reshape(-1, 2)[:, 0], last_pts.reshape(-1, 2)[:, 1])
    ax2.scatter(current_pts.reshape(-1, 2)[:, 0], current_pts.reshape(-1, 2)[:, 1])

    cv2.imshow("current_frame", current_frame)
    if last_kf["frame"] is not None:
        cv2.imshow("last_frame", last_kf["frame"])
    if match_frame is not None:
        cv2.imshow("match_frame", match_frame)

    # handle key presses
    # 'q' - Quit the running program
    # 's' - enter stepwise mode
    # 'a' - exit stepwise mode
    key = cv2.waitKey(1)
    if not step_wise and key == ord('s'):
        step_wise = True
    if key == ord('q'):
        break
    if step_wise:
        while True:
            key = cv2.waitKey(1)
            if key == ord('s'):
                break
            elif key == ord('a'):
                step_wise = False
                break

    # TODO: define a robust thresholding mechanism to decide when to insert a new keyframe
    # first compute threshold

    # update last infos for next iteration (ONLY DO THIS WHEN THRESHOLD IS EXCEEDED)
    #last_kf["frame"] = current_frame
    #last_kf["key_points"] = kp
    #last_kf["descriptors"] = des

    frame_idx += 1

    if frame_idx == 100:
        break

cap.release()
cv2.destroyAllWindows()

plt.show()

import pickle
pickle.dump(Rs, open("Rs.pkl", "wb"))
pickle.dump(ts, open("ts.pkl", "wb"))
pickle.dump(map_points, open("map_points.pkl", "wb"))
