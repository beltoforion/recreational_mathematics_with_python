import numpy as np
import random
import math

from mayavi import mlab
from tvtk.util.ctf import *

hs = ts = 1   # time and space step width
dimx = dimy = 100   
dimz = 200   

def create_arrays():
    global velocity, tau, kappa, gauss_peak, u
    
    u = np.zeros((3, dimx, dimy, dimz))
    velocity = np.zeros((dimx, dimy, dimz))   

    sz = 10
    sigma = 2
    xx, yy, zz = np.meshgrid(range(-sz, sz), range(-sz, sz), range(-sz, sz))
    gauss_peak = np.zeros((sz, sz, sz))
    gauss_peak = 300 / (sigma*2*math.pi) * (math.sqrt(2*math.pi)) * np.exp(- 0.5 * ((xx**2 + yy**2 + zz**2)/(sigma**2)))

def set_initial_conditions(u):
    global velocity, tau, kappa, gauss_peak

    velocity[0:dimx, 0:dimy, 0:dimz] = 0.3      # 0.39 m/s Wave velocity of shallow water waves (lambda 0.1, depth 0.1)
#    velocity[40:dimx-40, 40:dimy-40, 40:dimy-40] = 0.1    # will be set to a constant value of tau

    tau = ( (velocity*ts) / hs )**2
    kappa = ts * velocity / hs  

    # Place a single gaussian peak at the center of the simulation
#    put_gauss_peak(u, int(dimx/2), int(dimy/2), int(dimz/2), 10)
#    put_gauss_peak(u, 20, int(dimy/2), int(dimz/2), 10)
#    put_gauss_peak(u, int(dimx/2), 20, int(dimz/2), 10)
#    put_gauss_peak(u, int(dimx/2), int(dimy/2), 20, 10)
            

def put_gauss_peak(u, x : int, y : int, z: int, height):
    w, h, d = gauss_peak.shape
    w = int(w/2)
    h = int(h/2)
    d = int(d/2)
    u[0:2, x-w:x+w, y-h:y+h, z-d:z+d] += height * gauss_peak

def update(u : any, method : int):
    u[2] = u[1]
    u[1] = u[0]
    
    if method==0: 
        boundary_size = 1
        u[0, 1:dimx-1, 1:dimy-1, 1:dimz-1] = tau[1:dimx-1, 1:dimy-1, 1:dimz-1] \
                                   * (    1  * u[1, 0:dimx-2, 1:dimy-1, 1:dimz-1]  # c-1, r  , z   =>  1
                                        + 1  * u[1, 1:dimx-1, 0:dimy-2, 1:dimz-1]  # c  , r-1, z   =>  1
                                        + 1  * u[1, 1:dimx-1, 1:dimy-1, 0:dimz-2]  # c,   r  , z-1 =>  1
                                        - 6  * u[1, 1:dimx-1, 1:dimy-1, 1:dimz-1]  # c,   r  , z => -6
                                        + 1  * u[1, 2:dimx  , 1:dimy-1, 1:dimz-1]  # c+1, r  , z =>  1
                                        + 1  * u[1, 1:dimx-1, 2:dimy,   1:dimz-1]  # c,   r+1 =>  1
                                        + 1  * u[1, 1:dimx-1, 1:dimy-1, 2:dimz]    # c,   r+1 =>  1
                                     ) \
                                + 2 * u[1, 1:dimx-1, 1:dimy-1, 1:dimz-1] \
                                -     u[2, 1:dimx-1, 1:dimy-1, 1:dimz-1]
    elif method==1: # ok, (4)th Order https://www.ams.org/journals/mcom/1988-51-184/S0025-5718-1988-0935077-0/S0025-5718-1988-0935077-0.pdf; Page 702
        boundary_size = 2
        u[0, 2:dimx-2, 2:dimy-2, 2:dimz-2]  = tau[2:dimx-2, 2:dimy-2, 2:dimz-2]\
                                    * ( -  1 * u[1, 2:dimx-2, 0:dimy-4, 2:dimz-2]  # c    , r-2 => -1
                                        + 16 * u[1, 2:dimx-2, 1:dimy-3, 2:dimz-2]  # c    , r-1 => 16                                       

                                        -  1 * u[1, 0:dimx-4, 2:dimy-2, 2:dimz-2]  # c - 2, r => -1
                                        + 16 * u[1, 1:dimx-3, 2:dimy-2, 2:dimz-2]  # c - 1, r => 16
                                        - 90 * u[1, 2:dimx-2, 2:dimy-2, 2:dimz-2]  # c    , r => -60
                                        + 16 * u[1, 3:dimx-1, 2:dimy-2, 2:dimz-2]  # c+1  , r => 16
                                        -  1 * u[1, 4:dimx,   2:dimy-2, 2:dimz-2]  # c+2  , r => -1

                                        + 16 * u[1, 2:dimx-2, 3:dimy-1, 2:dimz-2]  # c    , r+1 => 16                                       
                                        - 1  * u[1, 2:dimx-2, 4:dimy,   2:dimz-2]  # c    , r+2 => -1 

                                        + 16 * u[1, 2:dimx-2, 2:dimx-2, 1:dimz-3]
                                        - 1  * u[1, 2:dimx-2, 2:dimx-2, 0:dimz-4]
                                        + 16 * u[1, 2:dimx-2, 2:dimx-2, 3:dimz-1]
                                        - 1  * u[1, 2:dimx-2, 2:dimx-2, 4:dimz]

                                        ) / 12 \
                                    + 2*u[1, 2:dimx-2, 2:dimy-2, 2:dimz-2] \
                                    -   u[2, 2:dimx-2, 2:dimy-2, 2:dimz-2]         
    elif method==2: # (6th) https://www.ams.org/journals/mcom/1988-51-184/S0025-5718-1988-0935077-0/S0025-5718-1988-0935077-0.pdf; Page 702
        boundary_size = 3
        u[0, 3:dimx-3, 3:dimy-3, 3:dimz-3]  = tau[3:dimx-3, 3:dimy-3, 3:dimz-3]\
                                    * (     2 * u[1, 3:dimx-3, 0:dimy-6, 3:dimz-3]  # c,   r-3
                                        -  27 * u[1, 3:dimx-3, 1:dimy-5, 3:dimz-3]  # c,   r-2
                                        + 270 * u[1, 3:dimx-3, 2:dimy-4, 3:dimz-3]  # c,   r-1

                                        +   2 * u[1, 0:dimx-6, 3:dimy-3, 3:dimz-3] # c - 3, r
                                        -  27 * u[1, 1:dimx-5, 3:dimy-3, 3:dimz-3] # c - 2, r
                                        + 270 * u[1, 2:dimx-4, 3:dimy-3, 3:dimz-3] # c - 1, r
                                       - 1470 * u[1, 3:dimx-3, 3:dimy-3, 3:dimz-3] # c    , r
                                        + 270 * u[1, 4:dimx-2, 3:dimy-3, 3:dimz-3] # c + 1, r
                                        -  27 * u[1, 5:dimx-1, 3:dimy-3, 3:dimz-3] # c + 2, r
                                        +   2 * u[1, 6:dimx,   3:dimy-3, 3:dimz-3] # c + 3, r

                                        + 270 * u[1, 3:dimx-3, 4:dimy-2, 3:dimz-3]  # c  , r+1
                                        -  27 * u[1, 3:dimx-3, 5:dimy-1, 3:dimz-3]  # c  , r+2
                                        +   2 * u[1, 3:dimx-3, 6:dimy  , 3:dimz-3]  # c  , r+3

                                        # Z-Dimension
                                        +   2 * u[1, 3:dimx-3, 3:dimy-3, 0:dimz-6] # c - 3, r
                                        -  27 * u[1, 3:dimx-3, 3:dimy-3, 1:dimz-5] # c - 2, r
                                        + 270 * u[1, 3:dimx-3, 3:dimy-3, 2:dimz-4] # c - 1, r
                                        + 270 * u[1, 3:dimx-3, 3:dimy-3, 4:dimz-2] # c + 1, r
                                        -  27 * u[1, 3:dimx-3, 3:dimy-3, 5:dimz-1] # c + 2, r
                                        +   2 * u[1, 3:dimx-3, 3:dimy-3, 6:dimz] # c + 3, r

                                        ) / 180 \
                                    + 2*u[1, 3:dimx-3, 3:dimy-3, 3:dimz-3] \
                                    -   u[2, 3:dimx-3, 3:dimy-3, 3:dimz-3]  

    update_boundary(u, boundary_size)


def update_boundary(u, sz) -> None:
    c = dimx-1
    u[0, dimx-sz-1:c, 1:dimy-1, 1:dimz-1] = u[1,  dimx-sz-2:c-1, 1:dimy-1, 1:dimz-1] + (kappa[dimx-sz-1:c, 1:dimy-1, 1:dimz-1]-1)/(kappa[ dimx-sz-1:c, 1:dimy-1, 1:dimz-1]+1) * (u[0,  dimx-sz-2:c-1, 1:dimy-1, 1:dimz-1] - u[1, dimx-sz-1:c,1:dimy-1, 1:dimz-1])
    
    c = 0
    u[0, c:sz, 1:dimy-1, 1:dimz-1]        = u[1, c+1:sz+1, 1:dimy-1, 1:dimz-1]       + (kappa[c:sz, 1:dimy-1, 1:dimz-1]-1)/(kappa[c:sz, 1:dimy-1, 1:dimz-1]+1)                * (u[0, c+1:sz+1,1:dimy-1, 1:dimz-1]       - u[1,c:sz,1:dimy-1, 1:dimz-1])

    r = dimy-1
    u[0, 1:dimx-1, dimy-1-sz:r, 1:dimz-1] = u[1, 1:dimx-1, dimy-2-sz:r-1, 1:dimz-1] + (kappa[1:dimx-1, dimy-1-sz:r, 1:dimz-1]-1)/(kappa[1:dimx-1, dimy-1-sz:r, 1:dimz-1]+1) * (u[0, 1:dimx-1, dimy-2-sz:r-1, 1:dimz-1] - u[1, 1:dimx-1, dimy-1-sz:r, 1:dimz-1])

    r = 0
    u[0, 1:dimx-1, r:sz, 1:dimz-1] = u[1, 1:dimx-1, r+1:sz+1, 1:dimz-1] + (kappa[1:dimx-1, r:sz, 1:dimz-1]-1)/(kappa[1:dimx-1, r:sz, 1:dimz-1]+1) * (u[0, 1:dimx-1, r+1:sz+1, 1:dimz-1] - u[1, 1:dimx-1, r:sz, 1:dimz-1])

    d = dimz-1
    u[0, 1:dimx-1, 1:dimy-1, dimz-1-sz:d] = u[1, 1:dimx-1, 1:dimy-1, dimz-2-sz:d-1] + (kappa[1:dimx-1, 1:dimy-1,  dimz-1-sz:d]-1)/(kappa[1:dimx-1, 1:dimy-1, dimz-1-sz:d]+1) * (u[0, 1:dimx-1, 1:dimy-1, dimz-2-sz:d-1] - u[1, 1:dimx-1, 1:dimy-1, dimz-1-sz:d])

    d = 0
    u[0, 1:dimx-1, 1:dimy-1, d:sz] = u[1, 1:dimx-1, 1:dimy-1, d+1:sz+1] + (kappa[1:dimx-1, 1:dimy-1, d:sz]-1)/(kappa[1:dimx-1, 1:dimy-1, d:sz]+1) * (u[0, 1:dimx-1, 1:dimy-1, d+1:sz+1] - u[1, 1:dimx-1, 1:dimy-1, d:sz])


def put_gauss_peak(u, x : int, y : int, z: int, height):
    w,h,d = gauss_peak.shape
    w = int(w/2)
    h = int(h/2)
    d = int(d/2)
    u[0:2, x-w:x+w, y-h:y+h, z-d:z+d] += height * gauss_peak


def place_raindrops(u):
    if (random.random()<0.000):
        w,h,d = gauss_peak.shape
        x = int(random.randrange(w, dimx-w))
        y = int(random.randrange(h, dimy-h))
        z = int(random.randrange(d, dimz-d))

        peak_ampl = 2
        put_gauss_peak(u, x, y, z, peak_ampl)


@mlab.animate(delay=20)
def update_loop():
    src = mlab.pipeline.scalar_field(u[0])
    volume = mlab.pipeline.volume(src, vmin=-30, vmax=30)

    tick = 0
    while True:
        tick += 0.06

        mlab.view(azimuth=4*tick, elevation=40, distance=500, focalpoint=(int(dimx/2), int(dimy/2), int(dimz/2)))

        place_raindrops(u)
        put_gauss_peak(u, int(dimx/2), int(dimy/2), 20, 3*math.sin(tick))
        put_gauss_peak(u, int(dimx/2), int(dimy/2), int(dimz-20), 3*math.sin(tick))
        update(u, 1)

        src.mlab_source.scalars = u[0]

        absmax = 20 #np.max(np.abs(u[0]))

        otf = PiecewiseFunction()
        for val, opacity in [(absmax, 1), (absmax * 0.3, 0), (-absmax, 1), (-absmax * 0.3, 0)]:
            otf.add_point(val, opacity)
        volume._volume_property.set_scalar_opacity(otf) 

        ctf = ColorTransferFunction()
        for p in [(absmax, 0, 0, 1), (0, 1, 1, 1), (-absmax, 1, 0, 0)]:
            ctf.add_rgb_point(*p)
        volume._volume_property.set_color(ctf)

        yield

def main():
    create_arrays()
    set_initial_conditions(u)

    animate = update_loop()
    mlab.show()

if __name__ == "__main__":
    main()