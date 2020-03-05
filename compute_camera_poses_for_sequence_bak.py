# import sys
# sys.path.append('/home/lukas/Pangolin/build/src')
#
# import pypangolin as pango
# from OpenGL.GL import *
# from pytransform3d.rotations import axis_angle_from_matrix

import numpy as np
import cv2
import pickle

from ssc import ssc

#from matplotlib import pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#from pytransform3d.rotations import *

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
    #last_des = [last_des[m.queryIdx] for m in matches[:num_matches]]
    #current_des = [des[m.trainIdx] for m in matches[:num_matches]]
    match_frame = np.zeros_like(current_frame)
    if draw:
        match_frame = cv2.drawMatches(last_keyframe, last_kp, current_frame, kp, matches[:num_matches], None, matchColor=(0, 0, 0), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return matches, last_pts, current_pts, match_frame


def to_twist(R, t):
    """Convert a 3x3 rotation matrix and translation vector (shape (3,))
    into a 6D twist coordinate (shape (6,))."""
    r, _ = cv2.Rodrigues(R)
    twist = np.zeros((6,))
    twist[:3] = r.reshape(3,)
    twist[3:] = t.reshape(3,)
    return twist

def from_twist(twist):
    """Convert a 6D twist coordinate (shape (6,)) into a 3x3 rotation matrix
    and translation vector (shape (3,))."""
    r = twist[:3].reshape(3, 3)
    t = twist[3:].reshape(3, 1)
    R, _ = cv2.Rodrigues(r)
    return R, t

cv2.namedWindow("last_keyframe", cv2.WINDOW_NORMAL)
cv2.resizeWindow("last_keyframe", 1600, 900)
cv2.namedWindow("current_frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("current_frame", 1600, 900)
#cv2.namedWindow("match_frame", cv2.WINDOW_NORMAL)
#cv2.resizeWindow("match_frame", 1600, 900)

step_wise = True
match_frame = None

Rs = []
ts = []

frame_idx = 0


# for testing remove the first part of the video where drone ascends
[cap.read() for _ in range(1000)]

# TODO: It is a good idea to normalize the frame sbefore performing any operation
#  this helps to account for changes in lighting, exposure, etc.


# TODO: Since a planar scene is observed, using the essential matrix is not optimal
# higher accuracy can be achieved by estimating a homography (cv2.findHomography)
# and decomposing the homography into possible rotations and translations (see experiments/Untitled9.ipynb)
def initialize(fast, orb, camera_matrix, min_parallax=60.0):
    """Initialize two keyframes, the camera poses and a 3D point cloud.

    Args:
        min_parallax (`float`): Threshold for the median distance of all
            keypoint matches between the first keyframe (firs frame) and the
            second key frame. Is used to determine which frame is the second
            keyframe. This is needed to ensure enough parallax to recover
            the camera poses and 3D points.
    """
    keyframes = []
    map_points = []

    # get first key frame
    retc, frame = cap.read()
    if not retc:
        raise RuntimeError("Could not read the first camera frame.")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.remap(frame, mapx, mapy, cv2.INTER_CUBIC)
    kp, des = extract_kp_des(frame, fast, orb)
    keyframes.append({"frame": frame, "kp": kp, "des": des})

    frame_idx_init = 0

    while True:
        retc, frame = cap.read()
        if not retc:
            raise RuntimeError("Could not read next camera frame.")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.remap(frame, mapx, mapy, cv2.INTER_CUBIC)  # undistort frame

        # extract keypoints and match with first key frame
        kp, des = extract_kp_des(frame, fast, orb)
        matches, last_pts, current_pts, match_frame = match(bf,
            keyframes[0]["frame"], frame, keyframes[0]["des"],
            des, keyframes[0]["kp"], kp, num_matches, draw=False)

        # determine median distance between all matched feature points
        median_dist = np.median(np.linalg.norm(last_pts.reshape(-1, 2)-current_pts.reshape(-1, 2), axis=1))
        print(median_dist)

        # if distance exceeds threshold choose frame as second keyframe
        if median_dist >= min_parallax:
            keyframes.append({"frame": frame, "kp": kp, "des": des})
            break

        frame_idx_init += 1

    keyframes[0]["kp"] = cv2.KeyPoint_convert(last_pts)
    #keyframes[0]["des"] = last_des
    keyframes[1]["kp"] = cv2.KeyPoint_convert(current_pts)
    #keyframes[1]["des"] = current_des

    # compute relative camera pose for second frame
    essential_mat, _ = cv2.findEssentialMat(last_pts, current_pts, camera_matrix, method=cv2.RANSAC)
    retval, R, t, mask = cv2.recoverPose(essential_mat, last_pts, current_pts, camera_matrix)
    mask = mask.astype(np.bool).reshape(-1,)
    if retval >= 0.25*current_pts.shape[1]:
        print("init R", R)
        print("init t", t)

        # relative camera pose
        R1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).astype(np.float64)
        t1 = np.array([[0], [0], [0]]).astype(np.float64)
        R2 = R.T
        t2 = -np.matmul(R.T, t.reshape(3,)).reshape(3,1)

        # insert pose (in twist coordinates) of KF0 and KF1 into keyframes dict
        # poses are w.r.t. KF0 which is the base coordinate system of the entire map
        keyframes[0]["pose"] = to_twist(R1, t1)
        keyframes[1]["pose"] = to_twist(R2, t2)

        # create projection matrices needed for triangulation of initial 3D point cloud
        proj_matrix1 = np.hstack([R1.T, -R1.T.dot(t1)])
        proj_matrix2 = np.hstack([R2.T, -R2.T.dot(t2)])
        proj_matrix1 = camera_matrix.dot(proj_matrix1)
        proj_matrix2 = camera_matrix.dot(proj_matrix2)

        # triangulate initial 3D point cloud
        pts_3d = cv2.triangulatePoints(proj_matrix1, proj_matrix2, last_pts.reshape(-1, 2).T, current_pts.reshape(-1, 2).T).T
        pts_3d = cv2.convertPointsFromHomogeneous(pts_3d).reshape(-1, 3)

        # add triangulated points to map points
        map_points.append({"pts_3d": pts_3d, "mask": mask})  # map_points[0] stores 3D points w.r.t. KF0, mask demarks good points in the set

        #pickle.dump(cv2.KeyPoint_convert(keyframes[0]["kp"]), open("img_points_kf0.pkl", "wb"))
        #pickle.dump(cv2.KeyPoint_convert(keyframes[1]["kp"]), open("img_points_kf1.pkl", "wb"))

        print("Initialization successful. Choose frames 0 and {} as key frames".format(frame_idx_init))

    else:
        raise RuntimeError("Could not recover intial camera pose based on selected keyframes. Try choosing a different pair of initial keyframes.")

    return keyframes, map_points


keyframes, map_points = initialize(fast, orb, camera_matrix)


previous_frame = None
previous_kp = None

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

    if frame_idx == 0:
        kp, des = extract_kp_des(current_frame, fast, orb)
        print("Found {} keypoints in current frame".format(len(kp)))
        previous_frame = current_frame
        previous_kp = keyframes[-1]["kp"]
        vis_current_frame = cv2.drawKeypoints(np.copy(current_frame), previous_kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        frame_idx += 1
        continue

    #if frame_idx % detect_interval == 0:
    #    kp, des = extract_kp_des(current_frame, fast, orb)
    #    print("Found {} keypoint in current frame".format(len(kp)))
    #    vis_current_frame = cv2.drawKeypoints(np.copy(current_frame), kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #else:

    # track matched kps of last key frame
    print("performing tracking")
    p0 = np.float32(cv2.KeyPoint_convert(previous_kp)).reshape(-1, 1, 2)
    p1, _st, _err = cv2.calcOpticalFlowPyrLK(previous_frame, current_frame, p0, None, **lk_params)
    p0r, _st, _err = cv2.calcOpticalFlowPyrLK(current_frame, previous_frame, p1, None, **lk_params)  # back-tracking
    d = abs(p0-p0r).reshape(-1, 2).max(-1)
    good = d < 1
    kp = cv2.KeyPoint_convert(p1)
    #kp, des = orb.compute(current_frame, kp)
    vis_current_frame = cv2.drawKeypoints(np.copy(current_frame), kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


    #print("len(kp)", len(kp))


    #print("map_points[-1]['pts_3d']", map_points[-1]["pts_3d"])

    #matches, last_pts, current_pts, last_des, current_des, match_frame = match(bf,
    #    keyframes[-1]["frame"], current_frame, keyframes[-1]["des"], des,
    #    keyframes[-1]["kp"], kp, num_matches, draw=True)

    # recover pose by solving PnP
    mask = map_points[-1]["mask"]
    img_points = cv2.KeyPoint_convert(kp)#[mask&good, :]  # 2D points in current frame
    pts_3d = map_points[-1]["pts_3d"]#[mask&good, :]  # corresponding 3D points in previous key frame
    print("current_pts before PnP", img_points, "len", img_points.shape)
    retval, rvec, tvec, inliers = cv2.solvePnPRansac(pts_3d.reshape(-1, 1, 3), img_points.reshape(-1, 1, 2), camera_matrix, None, reprojectionError=8, iterationsCount=100)
    if not retval:
        raise RuntimeError("Could not compute the camera pose for the new frame with solvePnP.")
    #print(retval)
    #print(inliers)
    R_rel = cv2.Rodrigues(rvec)[0].T
    t_rel = -np.matmul(cv2.Rodrigues(rvec)[0].T, tvec)
    print(R_rel, t_rel)

    Rs.append(R_rel)
    ts.append(t_rel)

    #pickle.dump(img_points, open("img_points_f0.pkl", "wb"))
    #pickle.dump(pts_3d, open("pts_3d.pkl", "wb"))

    cv2.imshow("current_frame", vis_current_frame)
    #if last_kf["frame"] is not None:
    #    cv2.imshow("last_frame", last_kf["frame"])
    #if match_frame is not None:
    #    cv2.imshow("match_frame", match_frame)
    cv2.imshow("last_keyframe", keyframes[-1]["frame"])

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

    # TODO: extend below: KF insertion decision based on travelled GPS distance and number of frames processed

    # decide when to insert a new keyframe based on a robust thresholding mechanism
    # the weights are chosen to make the difference more sensitive to changes in rotation and z-coordinate
    pose_distance_weights = np.array([[10, 0, 0, 0, 0, 0],  # r1
                                      [0, 10, 0, 0, 0, 0],  # r2
                                      [0, 0, 10, 0, 0, 0],  # r3
                                      [0, 0, 0, 1, 0, 0],   # t1 = x
                                      [0, 0, 0, 0, 1, 0],   # t2 = y
                                      [0, 0, 0, 0, 0, 10]]) # t3 = z
    pose_distance_threshold = 15
    curent_pose = to_twist(R_rel, t_rel).reshape(6, 1)
    current_dist = np.matmul(np.matmul(curent_pose.T, pose_distance_weights), curent_pose)
    print(current_dist)
    if current_dist >= pose_distance_threshold:
        print("########## insert new KF ###########")

        pickle.dump(current_frame, open("new_keyframe.pkl", "wb"))
        pickle.dump(cv2.KeyPoint_convert(kp), open("new_kp.pkl", "wb"))
        pickle.dump(curent_pose, open("new_pose.pkl", "wb"))

        keyframes[-1]["kp"] = cv2.KeyPoint_convert(keyframes[-1]["kp"])
        pickle.dump(keyframes[-1], open("last_keyframe.pkl", "wb"))

        print("Currently, {} KFs are stored".format(len(keyframes)))

        break # for testing only

        # extract new keypoints and compute ORB descriptors

    # update last infos for next iteration (ONLY DO THIS WHEN THRESHOLD IS EXCEEDED)
    #if frame_idx % 20 == 19:
    #    last_kf["frame"] = current_frame
    #    last_kf["key_points"] = kp
    #    last_kf["descriptors"] = des

    previous_frame = current_frame
    previous_kp = kp
    frame_idx += 1

    #if frame_idx == 100:
    #    break

cap.release()
cv2.destroyAllWindows()

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
