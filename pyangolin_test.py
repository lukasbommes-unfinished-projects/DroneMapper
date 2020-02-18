import sys
sys.path.append('/home/lukas/Pangolin/build/src')

import pypangolin as pango
from OpenGL.GL import *

import numpy as np

def main():
    win = pango.CreateWindowAndBind("pySimpleDisplay", 1600, 900)
    glEnable(GL_DEPTH_TEST)

    pm = pango.ProjectionMatrix(1600,900,1500,1500,800,450,0.1,1000);
    mv = pango.ModelViewLookAt(-0, 0.5, -3, 0, 0, 0, pango.AxisY)
    s_cam = pango.OpenGlRenderState(pm, mv)

    handler=pango.Handler3D(s_cam)
    d_cam = pango.CreateDisplay().SetBounds(pango.Attach(0),
                                            pango.Attach(1),
                                            pango.Attach(0),
                                            pango.Attach(1),
                                            -1600.0/900.0).SetHandler(handler)

    while not pango.ShouldQuit():
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        d_cam.Activate(s_cam)
        #pango.glDrawColouredCube()

        #points = np.random.random((10000, 3)) * 3 - 4
        #glPointSize(1)
        #glColor3f(1.0, 0.0, 0.0)
        #pango.DrawPoints(points)

        glPointSize(10)
        glColor3f(1.0, 0.0, 0.0)
        glBegin(GL_POINTS)
        for i in range(10):
            glVertex3f(i, 0.1, 0)
        glEnd()

        pango.FinishFrame()


if __name__ == "__main__":
    main()
