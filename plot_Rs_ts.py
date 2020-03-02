import sys
sys.path.append('/home/lukas/Pangolin/build/src')

import pypangolin as pango
from OpenGL.GL import *

import numpy as np
import pickle

from pytransform3d.rotations import axis_angle_from_matrix

Rs = pickle.load(open("Rs.pkl", "rb"))
ts = pickle.load(open("ts.pkl", "rb"))
map_points = pickle.load(open("map_points.pkl", "rb"))

m = map_points[0]["mask"]
mp_x = map_points[0]["pts_3d"][m, 0]
mp_y = map_points[0]["pts_3d"][m, 1]
mp_z = map_points[0]["pts_3d"][m, 2]

def main():
    win = pango.CreateWindowAndBind("pySimpleDisplay", 1600, 900)
    glEnable(GL_DEPTH_TEST)

    aspect = 1600/900
    cam_scale = 0.5
    cam_aspect = aspect

    pm = pango.ProjectionMatrix(1600,900,1000,1000,800,450,0.1,1000);
    mv = pango.ModelViewLookAt(0, 0, -1, 0, 0, 0, pango.AxisY)
    s_cam = pango.OpenGlRenderState(pm, mv)

    handler=pango.Handler3D(s_cam)
    d_cam = pango.CreateDisplay().SetBounds(pango.Attach(0),
                                            pango.Attach(1),
                                            pango.Attach(0),
                                            pango.Attach(1),
                                            -aspect).SetHandler(handler)

    while not pango.ShouldQuit():
        glClearColor(0.0, 0.5, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        d_cam.Activate(s_cam)
        glMatrixMode(GL_MODELVIEW)

        # draw map points
        glPointSize(2)
        glBegin(GL_POINTS)
        glColor3f(1.0, 1.0, 1.0)
        for p_x, p_y, p_z in zip(mp_x, mp_y, mp_z):
            glVertex3f(p_x, p_y, p_z)
        glEnd()

        # draw origin coordinate system (red: x, green: y, blue: z)
        glBegin(GL_LINES)
        glColor3f(1.0, 0, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(3, 0, 0)

        glColor3f(0, 1.0, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 3, 0)

        glColor3f(0, 0, 1.0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, 3)
        glEnd()

        # draw every camera pose as view frustrum
        for R, t in zip(Rs, ts):
            glPushMatrix()
            glTranslatef(*t)
            r = axis_angle_from_matrix(R) # returns x, y, z, angle
            r[-1] = r[-1]*180.0/np.pi  # rad -> deg
            glRotatef(r[3], r[0], r[1], r[2])  # angle, x, y, z
            glBegin(GL_LINES)
            glColor3f(1.0, 0, 0)
            glVertex3f(0, 0, 0)
            glVertex3f(cam_scale, 0, 0)

            glColor3f(0, 1.0, 0)
            glVertex3f(0, 0, 0)
            glVertex3f(0, cam_scale, 0)

            glColor3f(0, 0, 1.0)
            glVertex3f(0, 0, 0)
            glVertex3f(0, 0, cam_scale)
            glEnd()

            glBegin(GL_LINE_LOOP)
            glColor3f(1.0, 1.0, 1.0)
            glVertex3f(-1.0*cam_scale, -1.0*cam_scale/cam_aspect, 0.75*cam_scale)
            glVertex3f(1.0*cam_scale, -1.0*cam_scale/cam_aspect, 0.75*cam_scale)
            glVertex3f(1.0*cam_scale, 1.0*cam_scale/cam_aspect, 0.75*cam_scale)
            glVertex3f(-1.0*cam_scale, 1.0*cam_scale/cam_aspect, 0.75*cam_scale)
            glEnd()

            glBegin(GL_LINES)
            glColor3f(1.0, 1.0, 1.0)
            glVertex3f(-1.0*cam_scale, -1.0*cam_scale/cam_aspect, 0.75*cam_scale)
            glVertex3f(0, 0, 0)
            glVertex3f(1.0*cam_scale, -1.0*cam_scale/cam_aspect, 0.75*cam_scale)
            glVertex3f(0, 0, 0)
            glVertex3f(1.0*cam_scale, 1.0*cam_scale/cam_aspect, 0.75*cam_scale)
            glVertex3f(0, 0, 0)
            glVertex3f(-1.0*cam_scale, 1.0*cam_scale/cam_aspect, 0.75*cam_scale)
            glVertex3f(0, 0, 0)
            glEnd()

            glPopMatrix()

        pango.FinishFrame()


if __name__ == "__main__":
    main()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# for R, t in zip(Rs, ts):
#     plot_basis(ax, R, t.reshape(3,))
#
# plt.show()
