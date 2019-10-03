import numpy as np
import pygame
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *

# density goes from 0 to 20
def Lines(density):
    # nothing to show..
    if(density < 0.01):
        return
    glLineWidth(1.0)
    glBegin(GL_LINES)
    z = 0.0
    # glLineWidth(width);
    #alpha = min(1.0, 10.0/step)
    step = 2.0*np.pi / (density*500.0)
    alpha = min(1.0, 0.7/density)
    glColor4f(1.0, 0.0, 0.0, alpha)

    for angle in np.arange(0.0, np.pi-5e-6, step):
    # for angle in np.concatenate((
    #                 np.arange(0.0, np.pi-5e-6, step),
    #                 np.random.uniform(0.0, np.pi, int(np.pi/step)))):
        x = np.sin(angle)
        y = np.cos(angle)
        glVertex3d(x, y, z)
        x = np.sin(angle + np.pi)
        y = np.cos(angle + np.pi)
        glVertex3d(x, y, z)
    glEnd()
    print('Step: {0}'.format(step))


def show():
    pygame.init()
    display = (800,600)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
    glEnable(GL_BLEND)
    glEnable(GL_MULTISAMPLE)
    # glDisable(GL_LIGHTING)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    # glEnable(GL_LINE_SMOOTH)
    # glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
    # glEnable(GL_POLYGON_SMOOTH)
    # glHint( GL_POLYGON_SMOOTH_HINT, GL_NICEST )
    glShadeModel(GL_FLAT)

    # glutSetOption(GLUT_MULTISAMPLE, 8);
    # glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE | GLUT_MULTISAMPLE);

    # gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
    # glTranslatef(0.0,0.0, -5)
    density = 0.01
    glScalef(2.0, 2.0, 2.0)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        # glRotatef(1, 3, 1, 1)

        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        # Cube()
        Lines(density)
        density += 0.001
        pygame.display.flip()
        pygame.time.wait(30)

if __name__ == "__main__":
    show()



