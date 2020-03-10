import pickle
import numpy as np
import cv2


map_points = pickle.load(open("map_points.pkl", "rb"))
kf_visible_map_points = pickle.load(open("kf_visible_map_points.pkl", "rb"))
kf_poses = pickle.load(open("kf_poses.pkl", "rb"))
