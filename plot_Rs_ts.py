import pickle
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pytransform3d.rotations import *

Rs = pickle.load(open("Rs.pkl", "rb"))
ts = pickle.load(open("ts.pkl", "rb"))
map_points = pickle.load(open("map_points.pkl", "rb"))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for R, t in zip(Rs, ts):
    plot_basis(ax, R, t.reshape(3,))

plt.show()
