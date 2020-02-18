import numpy as np
import cv2

from ssc import ssc

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
fast = cv2.FastFeatureDetector_create(threshold=12)
num_ret_points = 3000
tolerance = 0.1
num_matches = 50

cv2.namedWindow("last_frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("last_frame", 1600, 900)
cv2.namedWindow("current_frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("current_frame", 1600, 900)
cv2.namedWindow("match_frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("match_frame", 1600, 900)


# 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# #ax.scatter(src_pts.reshape(-1, 2)[:, 0], src_pts.reshape(-1, 2)[:, 1], np.zeros(src_pts.shape[1]))
# #ax.scatter(dst_pts.reshape(-1, 2)[:, 0], dst_pts.reshape(-1, 2)[:, 1], np.zeros(dst_pts.shape[1]))
# ax.scatter(triangulatedPoints[:, 0], triangulatedPoints[:, 1], triangulatedPoints[:, 2])
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.set_zlabel("z")
# plt.legend(["frame1", "frame2", "triangulated"])
# plt.show()

# discard intial part of video
[cap.read() for i in range(1000)]

last = {
    "frame": None,
    "kp": None,
    "des": None
}

loaded_intial_frame = False

Rs = []
ts = []

frame_idx = 0

while(True):

    print("frame", frame_idx)

    retc, current_frame = cap.read()
    current_frame = cv2.remap(current_frame, mapx, mapy, cv2.INTER_CUBIC)

    kp = fast.detect(current_frame, None)
    kp = sorted(kp, key = lambda x:x.response, reverse=True)
    print("Found {} keypoint in current frame".format(len(kp)))

    kp = ssc(kp, num_ret_points, tolerance, current_frame.shape[1], current_frame.shape[0])
    kp, des = orb.compute(current_frame, kp)

    if loaded_intial_frame:

        last["frame"] = cv2.drawKeypoints(last["frame"], last["kp"], None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        current_frame = cv2.drawKeypoints(current_frame, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        matches = bf.match(last["des"], des)
        matches = sorted(matches, key = lambda x:x.distance)
        print("Found {} matches of current frame with last frame".format(len(matches)))

        match_frame = cv2.drawMatches(last["frame"], last["kp"], current_frame, kp, matches[:num_matches], None, matchColor=(0, 0, 0), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # get matched keypoints
        src_pts = np.array([last["kp"][m.queryIdx].pt for m in matches[:num_matches]]).reshape(1, -1, 2)
        dst_pts = np.array([kp[m.trainIdx].pt for m in matches[:num_matches]]).reshape(1, -1, 2)

        # recover camera pose from point correspondences
        essential_mat, _ = cv2.findEssentialMat(src_pts, dst_pts, camera_matrix, method=cv2.RANSAC)
        retval, R, t, mask = cv2.recoverPose(essential_mat, src_pts, dst_pts, camera_matrix)
        Rs.append(R)
        ts.append(t)

        cv2.imshow("last_frame", last["frame"])
        cv2.imshow("current_frame", current_frame)
        cv2.imshow("match_frame", match_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # update last infos for next iteration
    last["frame"] = current_frame
    last["kp"] = kp
    last["des"] = des

    loaded_intial_frame = True
    frame_idx += 1

    if frame_idx == 10:
        break

cap.release()
#cv2.destroyAllWindows()

print(ts)
print(Rs)
