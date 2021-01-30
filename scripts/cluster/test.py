# created by Florian Schunck on 05.08.2020
# Project: Tainter
# Short description of the feature:
# 
# 
# ------------------------------------------------------------------------------
# Open tasks:
# TODO:
# 
# ------------------------------------------------------------------------------
# Define a nice function of distance from individual pts
import numpy as np
import matplotlib.pyplot as plt


def tellme(s):
    print(s)
    plt.title(s, fontsize=16)
    plt.draw()

##################################################
# Define a triangle by clicking three points
##################################################
plt.clf()
plt.axis([-1., 1., -1., 1.])
plt.setp(plt.gca(), autoscale_on=False)

tellme('You will define a triangle, click to begin')

plt.waitforbuttonpress()

happy = False
while not happy:
    pts = []
    while len(pts) < 3:
        tellme('Select 3 corners with mouse')
        pts = np.asarray(plt.ginput(3, timeout=-1))
        if len(pts) < 3:
            tellme('Too few points, starting over')
            time.sleep(1)  # Wait a second

    ph = plt.fill(pts[:, 0], pts[:, 1], 'r', lw=2)

    tellme('Happy? Key click for yes, mouse click for no')

    happy = plt.waitforbuttonpress()

    # Get rid of fill
    if not happy:
        for p in ph:
            p.remove()

def f(x, y, pts):
    z = np.zeros_like(x)
    for p in pts:
        z = z + 1/(np.sqrt((x - p[0])**2 + (y - p[1])**2))
    return 1/z


X, Y = np.meshgrid(np.linspace(-1, 1, 51), np.linspace(-1, 1, 51))
Z = f(X, Y, pts)

CS = plt.contour(X, Y, Z, 20)

tellme('Use mouse to select contour label locations, middle button to finish')
CL = plt.clabel(CS, manual=True)