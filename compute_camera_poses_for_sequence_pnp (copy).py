import numpy as np
import cv2
import pickle

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
num_matches = 3000

lk_params = dict( winSize  = (21, 21),
                  maxLevel = 3,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


def extract_kp_des(current_frame, fast, orb):
    kp = fast.detect(current_frame, None)
    kp = sorted(kp, key = lambda x:x.response, reverse=True)
    kp = ssc(kp, num_ret_points, tolerance, current_frame.shape[1], current_frame.shape[0])
    kp, des = orb.compute(current_frame, kp)
    # good features to track lead to cleaner tracks, but much more noisy pose estimates
    #kp = cv2.goodFeaturesToTrack(current_frame, **feature_params)
    #kp = cv2.KeyPoint_convert(kp)
    #kp, des = orb.compute(current_frame, kp)
    return kp, des


def match(bf, last_keyframe, current_frame, last_des, des, last_kp, kp, num_matches, draw=True):
    matches = bf.match(last_des, des)
    matches = sorted(matches, key = lambda x:x.distance)
    print("Found {} matches of current frame with last frame".format(len(matches)))
    last_pts = np.array([last_kp[m.queryIdx].pt for m in matches[:num_matches]]).reshape(1, -1, 2)
    current_pts = np.array([kp[m.trainIdx].pt for m in matches[:num_matches]]).reshape(1, -1, 2)
    last_des = [last_des[m.queryIdx] for m in matches[:num_matches]]
    current_des = [des[m.trainIdx] for m in matches[:num_matches]]
    match_frame = np.zeros_like(current_frame)
    if draw:
        match_frame = cv2.drawMatches(last_keyframe, last_kp, current_frame, kp, matches[:num_matches], None, matchColor=(0, 0, 0), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return matches, last_pts, current_pts, last_des, current_des, match_frame


def to_twist(R, T):
    """Convert a 3x3 rotation matrix and translation vector (shape (3,))
    into a 6D twist coordinate (shape (6,))."""
    r, _ = cv2.Rodrigues(R)
    twist = np.zeros((6,))
    twist[:3] = r.reshape(3,)
    twist[3:] = t
    return twist

def from_twist(twist):
    """Convert a 6D twist coordinate (shape (6,)) into a 3x3 rotation matrix
    and translation vector (shape (3,))."""
    r = twist[:3]
    t = twist[3:]
    R, _ = cv2.Rodrigues(r)
    return R, t


cv2.namedWindow("last_frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("last_frame", 1600, 900)
cv2.namedWindow("current_frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("current_frame", 1600, 900)
cv2.namedWindow("match_frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("match_frame", 1600, 900)

step_wise = True

last_kf = {
    "frame": None,
    "key_points": None,
    "descriptors": None,
}
match_frame = None

Rs = []
ts = []

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

previous_frame = None
previous_kp = None
detect_interval = 10


def initialize(fast, orb, camera_matrix, kf_idx=[1000, 1050]):
    keyframes = []
    map_points = np.empty((0, 3), dtype=np.float64)
    map_points_mask = np.empty((0,), dtype=np.bool)
    ret = False
    # get two frames with indices as specified in kf_idx
    kf_idx[1] = kf_idx[1] - kf_idx[0]
    for i in range(2):
        [cap.read() for i in range(kf_idx[i])]
        retc, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.remap(frame, mapx, mapy, cv2.INTER_CUBIC)
        kp, des = extract_kp_des(frame, fast, orb)
        keyframes.append({"frame": frame, "kp": kp, "des": des})

    matches, last_pts, current_pts, last_des, current_des, match_frame = match(bf,
        keyframes[0]["frame"], keyframes[1]["frame"], keyframes[0]["des"],
        keyframes[1]["des"], keyframes[0]["kp"], keyframes[1]["kp"], num_matches, draw=False)

    # update keypoints and descriptors to only those belonging to matches
    keyframes[0]["kp"] = last_pts
    keyframes[0]["des"] = last_des
    keyframes[1]["kp"] = current_pts
    keyframes[1]["des"] = current_des

    # compute relative camera pose for second frame
    essential_mat, _ = cv2.findEssentialMat(last_pts, current_pts, camera_matrix, method=cv2.RANSAC)
    retval, R, t, mask = cv2.recoverPose(essential_mat, last_pts, current_pts, camera_matrix)
    mask = mask.astype(np.bool).reshape(-1,)
    map_points_mask = np.hstack((map_points_mask, mask))
    if retval >= 0.25*current_pts.shape[1]:
        ret = True
        print("init R", R)
        print("init t", t)

        # insert pose (in twist coordinates) of KF0 and KF1 into keyframes dict
        # poses are w.r.t. KF0 which is the base coordinate system of the entire map
        keyframes[0]["pose"] = np.zeros(6,)
        keyframes[1]["pose"] = to_twist(R, t)

        # triangulate initial 3D point cloud
        proj_matrix1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        proj_matrix2 = np.zeros((3, 4))
        proj_matrix2[:, :3] = R
        proj_matrix2[:, -1] = t.reshape(3,)
        proj_matrix1 = np.matmul(camera_matrix, proj_matrix1)
        proj_matrix2 = np.matmul(camera_matrix, proj_matrix2)
        pts_3d = cv2.triangulatePoints(proj_matrix1, proj_matrix2, last_pts.T, current_pts.T).reshape(-1, 4)
        pts_3d = cv2.convertPointsFromHomogeneous(pts_3d).reshape(-1, 3)

        # add triangulated points to map points
        map_points = np.vstack((map_points, pts_3d))

        # data = {
        #     "last_pts": last_pts,
        #     "current_pts": current_pts,
        #     "pose_R": R,
        #     "pose_t": t,
        #     "map_points": map_points
        # }
        # pickle.dump(data, open("data.pkl", "wb"))

    return ret, keyframes, map_points, map_points_mask


ret, keyframes, map_points, map_points_mask = initialize(fast, orb, camera_matrix)

if not ret:
    raise RuntimeError(("Initialization failed. Could not recover pose. Make "
        "sure enough parallax is present between the two keyframes."))

while(True):

    # TODO: whenever a new frame arrives
    # 1) preprocess frame (warp)
    # 2) extract kps, track kps
    # 3) match kps with kps of previous KF
    # 4) solve PnP for matched keypoints to recover camera pose of current frame
    # 5) compute the relative change of pose w.r.t pose in last KF
    # 6) if pose change exceeds threshold insert a new keyframe

    # TODO: insert a new key frame
    # 1) triangulate a new set of 3D points between the last KF and this KF
    # 2) merge the 3D point cloud with existing map points (3D points), remove duplicates, etc.
    # 3) check which other keyframes share the same keypoints (pyDBoW3)
    # 4) perform BA with local group (new keyframe + keyframes from 1) to adjust pose and 3D point estimates of the local group

    print("frame", frame_idx)

    retc, current_frame = cap.read()
    current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    current_frame = cv2.remap(current_frame, mapx, mapy, cv2.INTER_CUBIC)

    # if frame_idx == 0:
    #     kp, des = extract_kp_des(current_frame, fast, orb)
    #     print("Found {} keypoint in current frame".format(len(kp)))
    #     previous_frame = current_frame
    #     previous_kp = kp
    #     last_kf["frame"] = current_frame
    #     last_kf["key_points"] = kp
    #     last_kf["descriptors"] = des
    #     frame_idx += 1
    #     continue

    if frame_idx % detect_interval == 0:
        kp, des = extract_kp_des(current_frame, fast, orb)
        print("Found {} keypoint in current frame".format(len(kp)))
        vis_current_frame = cv2.drawKeypoints(np.copy(current_frame), kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    else:
        print("performing tracking")
        p0 = np.float32(cv2.KeyPoint_convert(previous_kp)).reshape(-1, 1, 2)
        p1, _st, _err = cv2.calcOpticalFlowPyrLK(previous_frame, current_frame, p0, None, **lk_params)
        p0r, _st, _err = cv2.calcOpticalFlowPyrLK(current_frame, previous_frame, p1, None, **lk_params)  # back-tracking
        d = abs(p0-p0r).reshape(-1, 2).max(-1)
        good = d < 1
        kp = cv2.KeyPoint_convert(p1[good])
        kp, des = orb.compute(current_frame, kp)
        print(len(kp))
        vis_current_frame = cv2.drawKeypoints(np.copy(current_frame), kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    matches, last_pts, current_pts, last_des, current_des, match_frame = match(bf,
        keyframes[-1]["frame"], current_frame, keyframes[-1]["des"], des,
        keyframes[-1]["kp"], kp, num_matches, draw=True)

    print(len(matches))
    print(last_pts.shape)
    print(current_pts.shape)

    # recover pose by solving PnP


    # # recover camera pose from point correspondences
    # essential_mat, _ = cv2.findEssentialMat(last_pts, current_pts, camera_matrix, method=cv2.RANSAC)
    # retval, R, t, mask = cv2.recoverPose(essential_mat, last_pts, current_pts, camera_matrix)
    # if retval >= 0.25*current_pts.shape[1]: # more than 50 % of the points should be valid for the pose to be recovered properly
    #     Rs.append(R)
    #     ts.append(t)
    #     map_points.append(current_pts)#[mask.astype(np.bool)])
    #
    # print("R", R)
    # print("t", t)
    # print(retval)

    #plot_basis(ax1, R, t.reshape(3,))

    #ax2.scatter(last_pts.reshape(-1, 2)[:, 0], last_pts.reshape(-1, 2)[:, 1])
    #ax2.scatter(current_pts.reshape(-1, 2)[:, 0], current_pts.reshape(-1, 2)[:, 1])

    cv2.imshow("current_frame", vis_current_frame)
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
    if frame_idx % 20 == 19:
        last_kf["frame"] = current_frame
        last_kf["key_points"] = kp
        last_kf["descriptors"] = des

    previous_frame = current_frame
    previous_kp = kp
    frame_idx += 1

    #if frame_idx == 100:
    #    break

cap.release()
cv2.destroyAllWindows()

plt.show()

pickle.dump(Rs, open("Rs.pkl", "wb"))
pickle.dump(ts, open("ts.pkl", "wb"))
pickle.dump(map_points, open("map_points.pkl", "wb"))




# while(True):
#
#     # TODO: whenever a new frame arrives
#     # 1) preprocess frame (warp)
#     # 2) extract kps, track kps
#     # 3) match kps with kps of previous KF
#     # 4) solve PnP for matched keypoints to recover camera pose of current frame
#     # 5) compute the relative change of pose w.r.t pose in last KF
#     # 6) if pose change exceeds threshold insert a new keyframe
#
#     # TODO: insert a new key frame
#     # 1) triangulate a new set of 3D points between the last KF and this KF
#     # 2) merge the 3D point cloud with existing map points (3D points), remove duplicates, etc.
#     # 3) check which other keyframes share the same keypoints (pyDBoW3)
#     # 4) perform BA with local group (new keyframe + keyframes from 1) to adjust pose and 3D point estimates of the local group
#
#     print("frame", frame_idx)
#
#     retc, current_frame = cap.read()
#     current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
#     current_frame = cv2.remap(current_frame, mapx, mapy, cv2.INTER_CUBIC)
#
#     if frame_idx == 0:
#         kp, des = extract_kp_des(current_frame, fast, orb)
#         print("Found {} keypoint in current frame".format(len(kp)))
#         previous_frame = current_frame
#         previous_kp = kp
#         last_kf["frame"] = current_frame
#         last_kf["key_points"] = kp
#         last_kf["descriptors"] = des
#         frame_idx += 1
#         continue
#
#     if frame_idx % detect_interval == 0:
#         kp, des = extract_kp_des(current_frame, fast, orb)
#         print("Found {} keypoint in current frame".format(len(kp)))
#         vis_current_frame = cv2.drawKeypoints(np.copy(current_frame), kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#     else:
#         print("performing tracking")
#         p0 = np.float32(cv2.KeyPoint_convert(previous_kp)).reshape(-1, 1, 2)
#         p1, _st, _err = cv2.calcOpticalFlowPyrLK(previous_frame, current_frame, p0, None, **lk_params)
#         p0r, _st, _err = cv2.calcOpticalFlowPyrLK(current_frame, previous_frame, p1, None, **lk_params)  # back-tracking
#         d = abs(p0-p0r).reshape(-1, 2).max(-1)
#         good = d < 1
#         kp = cv2.KeyPoint_convert(p1[good])
#         kp, des = orb.compute(current_frame, kp)
#         print(len(kp))
#         vis_current_frame = cv2.drawKeypoints(np.copy(current_frame), kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#
#     matches, last_pts, current_pts, last_des, current_des, match_frame = match(bf,
#         last_kf["frame"], current_frame, last_kf["descriptors"], des, last_kf["key_points"], kp, num_matches, draw=True)
#
#     # recover camera pose from point correspondences
#     essential_mat, _ = cv2.findEssentialMat(last_pts, current_pts, camera_matrix, method=cv2.RANSAC)
#     retval, R, t, mask = cv2.recoverPose(essential_mat, last_pts, current_pts, camera_matrix)
#     if retval >= 0.25*current_pts.shape[1]: # more than 50 % of the points should be valid for the pose to be recovered properly
#         Rs.append(R)
#         ts.append(t)
#         map_points.append(current_pts)#[mask.astype(np.bool)])
#
#     print("R", R)
#     print("t", t)
#     print(retval)
#
#     plot_basis(ax1, R, t.reshape(3,))
#
#     #ax2.scatter(last_pts.reshape(-1, 2)[:, 0], last_pts.reshape(-1, 2)[:, 1])
#     #ax2.scatter(current_pts.reshape(-1, 2)[:, 0], current_pts.reshape(-1, 2)[:, 1])
#
#     cv2.imshow("current_frame", vis_current_frame)
#     if last_kf["frame"] is not None:
#         cv2.imshow("last_frame", last_kf["frame"])
#     if match_frame is not None:
#         cv2.imshow("match_frame", match_frame)
#
#     # handle key presses
#     # 'q' - Quit the running program
#     # 's' - enter stepwise mode
#     # 'a' - exit stepwise mode
#     key = cv2.waitKey(1)
#     if not step_wise and key == ord('s'):
#         step_wise = True
#     if key == ord('q'):
#         break
#     if step_wise:
#         while True:
#             key = cv2.waitKey(1)
#             if key == ord('s'):
#                 break
#             elif key == ord('a'):
#                 step_wise = False
#                 break
#
#     # TODO: define a robust thresholding mechanism to decide when to insert a new keyframe
#     # first compute threshold
#
#     # update last infos for next iteration (ONLY DO THIS WHEN THRESHOLD IS EXCEEDED)
#     if frame_idx % 20 == 19:
#         last_kf["frame"] = current_frame
#         last_kf["key_points"] = kp
#         last_kf["descriptors"] = des
#
#     previous_frame = current_frame
#     previous_kp = kp
#     frame_idx += 1
#
#     #if frame_idx == 100:
#     #    break
#
# cap.release()
# cv2.destroyAllWindows()
#
# plt.show()
#
# pickle.dump(Rs, open("Rs.pkl", "wb"))
# pickle.dump(ts, open("ts.pkl", "wb"))
# pickle.dump(map_points, open("map_points.pkl", "wb"))
