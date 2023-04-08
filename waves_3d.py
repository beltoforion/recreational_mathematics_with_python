import numpy as np
import math

from mayavi import mlab
from tvtk.util.ctf import *

hs = ts = 1        # time and space step width
dimx = dimy = 100  # dimensions of the simulation domain 
dimz = 60           

u = np.zeros((3, dimx, dimy, dimz))   # three state arrays of the field variable

# Initial conditions
vel = np.full((dimx, dimy, dimz), 0.3)
vel[0:dimx, 0:dimy, 4:6] = 0.0

for i in range(dimx):
    for j in range(dimy):
        for k in range(4, 6):
            distance1 = np.linalg.norm(np.array([i, j, k]) - np.array([40, 50, 5]))
            distance2 = np.linalg.norm(np.array([i, j, k]) - np.array([60, 50, 5]))
            if distance1 <= 5 or distance2 <= 5:
                vel[i, j, k] = 0.3

tau = ( (vel*ts) / hs )**2
k = ts * vel / hs  

def update(u : any):
    u[2] = u[1]
    u[1] = u[0]
    
    u[0, 1:dimx-1, 1:dimy-1, 1:dimz-1] = tau[1:dimx-1, 1:dimy-1, 1:dimz-1] \
        * (   u[1, 0:dimx-2, 1:dimy-1, 1:dimz-1] +     u[1, 1:dimx-1, 0:dimy-2, 1:dimz-1]
            + u[1, 1:dimx-1, 1:dimy-1, 0:dimz-2] - 6 * u[1, 1:dimx-1, 1:dimy-1, 1:dimz-1]
            + u[1, 2:dimx  , 1:dimy-1, 1:dimz-1] +     u[1, 1:dimx-1, 2:dimy,   1:dimz-1]
            + u[1, 1:dimx-1, 1:dimy-1, 2:dimz] ) + 2 * u[1, 1:dimx-1, 1:dimy-1, 1:dimz-1] - u[2, 1:dimx-1, 1:dimy-1, 1:dimz-1]

    # Absorbing boundary conditions for all 6 sides of the simulation domain
    c = dimx - 1
    u[0, dimx-2:c, 1:dimy-1, 1:dimz-1] = u[1, dimx-3:c-1, 1:dimy-1, 1:dimz-1] + (k[dimx-2:c, 1:dimy-1, 1:dimz-1]-1)/(k[dimx-2:c, 1:dimy-1, 1:dimz-1]+1) * (u[0, dimx-3:c-1, 1:dimy-1, 1:dimz-1] - u[1, dimx-2:c,1:dimy-1, 1:dimz-1])
    c = 0
    u[0, c:1, 1:dimy-1, 1:dimz-1] = u[1, 1:2, 1:dimy-1, 1:dimz-1] + (k[c:1, 1:dimy-1, 1:dimz-1]-1)/(k[c:1, 1:dimy-1, 1:dimz-1]+1) * (u[0, 1:2,1:dimy-1, 1:dimz-1] - u[1, c:1,1:dimy-1, 1:dimz-1])
    r = dimy-1
    u[0, 1:dimx-1, dimy-2:r, 1:dimz-1] = u[1, 1:dimx-1, dimy-3:r-1, 1:dimz-1] + (k[1:dimx-1, dimy-2:r, 1:dimz-1]-1)/(k[1:dimx-1, dimy-2:r, 1:dimz-1]+1) * (u[0, 1:dimx-1, dimy-3:r-1, 1:dimz-1] - u[1, 1:dimx-1, dimy-2:r, 1:dimz-1])
    r = 0
    u[0, 1:dimx-1, r:1, 1:dimz-1] = u[1, 1:dimx-1, 1:2, 1:dimz-1] + (k[1:dimx-1, r:1, 1:dimz-1]-1)/(k[1:dimx-1, r:1, 1:dimz-1]+1) * (u[0, 1:dimx-1, 1:2, 1:dimz-1] - u[1, 1:dimx-1, r:1, 1:dimz-1])
    d = dimz-1
    u[0, 1:dimx-1, 1:dimy-1, dimz-2:d] = u[1, 1:dimx-1, 1:dimy-1, dimz-3:d-1] + (k[1:dimx-1, 1:dimy-1, dimz-2: d]-1)/(k[1:dimx-1, 1:dimy-1, dimz-2:d]+1) * (u[0, 1:dimx-1, 1:dimy-1, dimz-3:d-1] - u[1, 1:dimx-1, 1:dimy-1, dimz-2:d])
    d = 0
    u[0, 1:dimx-1, 1:dimy-1, d:1] = u[1, 1:dimx-1, 1:dimy-1, d+1:2] + (k[1:dimx-1, 1:dimy-1, d:1]-1)/(k[1:dimx-1, 1:dimy-1, d:1]+1) * (u[0, 1:dimx-1, 1:dimy-1, d+1:2] - u[1, 1:dimx-1, 1:dimy-1, d:1])

@mlab.animate(delay=20)
def update_loop():
    fig = mlab.figure(bgcolor=(0, 0, 0))
    src = mlab.pipeline.scalar_field(u[0])
    volume = mlab.pipeline.volume(src)
    tick = 0
    while True:
        tick += 0.1
        mlab.view(azimuth=4*tick, elevation=80, distance=200, focalpoint=(int(dimx/2), int(dimy/2), 20))

        u[0:2, 1:dimx-1, 1:dimy-1, 1:3] += 2 * math.sin(tick*1.5)
        update(u)

        src.mlab_source.scalars = u[0]
        absmax = 1 

        otf = PiecewiseFunction()
        for val, opacity in [(absmax, 1), (absmax * 0.1, 0), (-absmax, 1), (-absmax * 0.1, 0)]:
            otf.add_point(val, opacity)
        volume._volume_property.set_scalar_opacity(otf) 

        ctf = ColorTransferFunction()
        for p in [(absmax, 0, 1, 0), (0, 0, 0, 0), (-absmax, 1, 0, 0)]:
            ctf.add_rgb_point(*p)
        volume._volume_property.set_color(ctf)

        yield

if __name__ == "__main__":
    animate = update_loop()
    mlab.show()