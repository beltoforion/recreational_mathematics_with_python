import pygame
import numpy as np
import random
import math
import time

hs = 1   # spatial step width
ts = 1   # time step width
dimx = 200   # width of the simulation domain
dimy = 200   # height of the simulation domain
cellsize = 3 # display size of a cell in pixel

def create_arrays():
    global velocity
    global tau
    global kappa
    global gauss_peak
    global u
    global uu

    # The three dimensional simulation grid 
    u = np.zeros((3, dimx, dimy))       

    # A second grid for comparing simulation results with another method
    uu = np.zeros((3, dimx, dimy))     

    # A field containing the velocity for each cell
    velocity = np.zeros((dimx, dimy))   

    # A field containing the factor for the Laplace Operator that  combines Velocity and Grid Constants for the Wave Equation
    tau  = np.zeros((dimx, dimy))       

    # A field containing the factor for the Laplace Operator that combines Velocity and Grid Constants for the Boundary Condition 
    kappa = np.zeros((dimx, dimy))      

    # Create a template for a gauss peak to use as a rain drop model
    sz = 10
    sigma = 1.5
    xx, yy = np.meshgrid(range(-sz, sz), range(-sz, sz))
    gauss_peak = np.zeros((sz, sz))
    gauss_peak = 300 / (sigma*2*math.pi) * (math.sqrt(2*math.pi)) * np.exp(- 0.5 * ((xx**2+yy**2)/(sigma**2)))

def set_initial_conditions(u):
    global velocity
    global tau
    global kappa
    global gauss_peak

    velocity[0:dimx,0:dimy] = 0.3            # 0.39 m/s Wave velocity of shallow water waves (lambda 0.1, depth 0.1)
    velocity[110:150,50:dimy-50] = 0.2      # will be set to a constant value of tau
    velocity[150:200,50:dimy-50] = 0.1      # will be set to a constant value of tau
    velocity[0:dimx, 150:] = 0.4     # will be set to a constant value of tau

    # compute tau and kappa from the velocity field
    tau = ( (velocity*ts) / hs )**2
    kappa = ts * velocity / hs  

    # Place a single gaussian peak at the center of the simulation
    put_gauss_peak(u, int(dimx/2), int(dimy/2), 10)

def put_gauss_peak(u, x : int, y : int, height):
    """Place a gauss shaped peak into the simulation domain.
    
        This function will put a gauss shaped peak at position x,y 
        of the simulation domain.
    """
    w,h = gauss_peak.shape
    w = int(w/2)
    h = int(h/2)

    use_multipole = False
    if use_multipole:
        # Multipole
        dist = 3
        u[0:2, x-w-dist:x+w-dist, y-h:y+h] += height * gauss_peak
        u[0:2, x-w:x+w, y-h+dist:y+h+dist] -= height * gauss_peak
        u[0:2, x-w+dist:x+w+dist, y-h:y+h] += height * gauss_peak
        u[0:2, x-w:x+w, y-h-dist:y+h-dist] -= height * gauss_peak        
    else:
        # simple peak
        u[0:2, x-w:x+w, y-h:y+h] += height * gauss_peak
            
def update(u, method):
    u[2] = u[1]
    u[1] = u[0]
    
    if method==0: 
        boundary_size = 1

        # This is the second order scheme you will most commonly see. It does not take diagonaly into account. 
        # Some waves may appear a tiny bit edgy.
        u[0, 1:dimx-1, 1:dimy-1] = tau[1:dimx-1, 1:dimy-1] \
                                   * (        u[1, 1:dimx-1, 0:dimy-2] # c, r-1 =>  1

                                        +     u[1, 0:dimx-2, 1:dimy-1] # c-1, r =>  1
                                        - 4 * u[1, 1:dimx-1, 1:dimy-1] # c,   r => -4
                                        +     u[1, 2:dimx  , 1:dimy-1] # c+1, r =>  1

                                        +     u[1, 1:dimx-1, 2:dimy]   # c, r+1 =>  1
                                     ) \
                                + 2 * u[1, 1:dimx-1, 1:dimy-1] \
                                -     u[2, 1:dimx-1, 1:dimy-1]
    elif method==1: 
        boundary_size = 1

        # This is the second order scheme with a laplacian that takes the diagonals into account.
        # The resulting wave shape will look a bit better under certain conditions but the accuracy 
        # is still low. In most cases you will hardly see a difference to #1
        u[0, 1:dimx-1, 1:dimy-1] = tau[1:dimx-1, 1:dimy-1] \
                                   * (    0.25 * u[1, 0:dimx-2, 0:dimy-2] # c-1, r-1 =>  1
                                        + 0.5  * u[1, 1:dimx-1, 0:dimy-2] # c,   r-1 =>  1
                                        + 0.25 * u[1, 2:dimx  , 0:dimy-2] # c+1, r-1 =>  1

                                        + 0.5  * u[1, 0:dimx-2, 1:dimy-1] # c-1, r =>  1
                                        - 3    * u[1, 1:dimx-1, 1:dimy-1] # c,   r => -8
                                        + 0.5  * u[1, 2:dimx  , 1:dimy-1] # c+1, r =>  1

                                        + 0.25 * u[1, 0:dimx-2, 2:dimy]   # c-1, r+1 =>  1
                                        + 0.5  * u[1, 1:dimx-1, 2:dimy]   # c,   r+1 =>  1
                                        + 0.25 * u[1, 2:dimx  , 2:dimy]   # c+1, r+1 =>  1
                                     ) \
                                + 2 * u[1, 1:dimx-1, 1:dimy-1] \
                                -     u[2, 1:dimx-1, 1:dimy-1]
    elif method==2: # ok, (4)th Order https://www.ams.org/journals/mcom/1988-51-184/S0025-5718-1988-0935077-0/S0025-5718-1988-0935077-0.pdf; Page 702
        boundary_size = 2
        u[0, 2:dimx-2, 2:dimy-2]  = tau[2:dimx-2, 2:dimy-2]\
                                    * ( -  1 * u[1, 2:dimx-2, 0:dimy-4]  # c    , r-2 => -1
                                        + 16 * u[1, 2:dimx-2, 1:dimy-3]  # c    , r-1 => 16                                       

                                        -  1 * u[1, 0:dimx-4, 2:dimy-2]  # c - 2, r => -1
                                        + 16 * u[1, 1:dimx-3, 2:dimy-2]  # c - 1, r => 16
                                        - 60 * u[1, 2:dimx-2, 2:dimy-2]  # c    , r => -60
                                        + 16 * u[1, 3:dimx-1, 2:dimy-2]  # c+1  , r => 16
                                        -  1 * u[1, 4:dimx,   2:dimy-2]  # c+2  , r => -1

                                        + 16 * u[1, 2:dimx-2, 3:dimy-1]  # c    , r+1 => 16                                       
                                        - 1  * u[1, 2:dimx-2, 4:dimy]    # c    , r+2 => -1 
                                        ) / 12 \
                                    + 2*u[1, 2:dimx-2, 2:dimy-2] \
                                    -   u[2, 2:dimx-2, 2:dimy-2]         
    elif method==3: # ok, (6th) https://www.ams.org/journals/mcom/1988-51-184/S0025-5718-1988-0935077-0/S0025-5718-1988-0935077-0.pdf; Page 702
        boundary_size = 3
        u[0, 3:dimx-3, 3:dimy-3]  = tau[3:dimx-3, 3:dimy-3]\
                                    * (     2 * u[1, 3:dimx-3, 0:dimy-6]  # c,   r-3
                                        -  27 * u[1, 3:dimx-3, 1:dimy-5]  # c,   r-2
                                        + 270 * u[1, 3:dimx-3, 2:dimy-4]  # c,   r-1

                                        +   2 * u[1, 0:dimx-6, 3:dimy-3] # c - 3, r
                                        -  27 * u[1, 1:dimx-5, 3:dimy-3] # c - 2, r
                                        + 270 * u[1, 2:dimx-4, 3:dimy-3] # c - 1, r
                                        - 980 * u[1, 3:dimx-3, 3:dimy-3] # c    , r
                                        + 270 * u[1, 4:dimx-2, 3:dimy-3] # c + 1, r
                                        -  27 * u[1, 5:dimx-1, 3:dimy-3] # c + 2, r
                                        +   2 * u[1, 6:dimx,   3:dimy-3] # c + 3, r

                                        + 270 * u[1, 3:dimx-3, 4:dimy-2]  # c  , r+1
                                        -  27 * u[1, 3:dimx-3, 5:dimy-1]  # c  , r+2
                                        +   2 * u[1, 3:dimx-3, 6:dimy  ]  # c  , r+3
                                        ) / 180 \
                                    + 2*u[1, 3:dimx-3, 3:dimy-3] \
                                    -   u[2, 3:dimx-3, 3:dimy-3]  
    elif method==4: # ok, (8th) https://www.ams.org/journals/mcom/1988-51-184/S0025-5718-1988-0935077-0/S0025-5718-1988-0935077-0.pdf; Page 702
        boundary_size = 4
        u[0, 4:dimx-4, 4:dimy-4]  = tau[4:dimx-4, 4:dimy-4]\
                                    * ( -  1/560 * u[1, 4:dimx-4, 0:dimy-8]  # c,   r-4
                                        +  8/315 * u[1, 4:dimx-4, 1:dimy-7]  # c,   r-3
                                        -    1/5 * u[1, 4:dimx-4, 2:dimy-6]  # c,   r-2
                                        +    8/5 * u[1, 4:dimx-4, 3:dimy-5]  # c,   r-1

                                        - 1/560  * u[1, 0:dimx-8, 4:dimy-4]  # c - 4, r
                                        + 8/315  * u[1, 1:dimx-7, 4:dimy-4]  # c - 3, r
                                        -   1/5  * u[1, 2:dimx-6, 4:dimy-4]  # c - 2, r
                                        +   8/5  * u[1, 3:dimx-5, 4:dimy-4]  # c - 1, r
                                        - 410/72 * u[1, 4:dimx-4, 4:dimy-4]  # c    , r
                                        +   8/5  * u[1, 5:dimx-3, 4:dimy-4]  # c + 1, r
                                        -   1/5  * u[1, 6:dimx-2, 4:dimy-4]  # c + 2, r
                                        + 8/315  * u[1, 7:dimx-1, 4:dimy-4]  # c + 3, r
                                        - 1/560  * u[1, 8:dimx  , 4:dimy-4]  # c + 4, r

                                        +    8/5 * u[1, 4:dimx-4, 5:dimy-3]  # c  , r+1
                                        -    1/5 * u[1, 4:dimx-4, 6:dimy-2]  # c  , r+2
                                        +  8/315 * u[1, 4:dimx-4, 7:dimy-1]  # c  , r+3
                                        -  1/560 * u[1, 4:dimx-4, 8:dimy  ]  # c  , r+4
                                        ) \
                                    + 2*u[1, 4:dimx-4, 4:dimy-4] \
                                    -   u[2, 4:dimx-4, 4:dimy-4]  

    # Absorbing Boundary Conditions:
    mur = True
    if mur==True:
        update_boundary(u, boundary_size)

def update_boundary(u, sz) -> None:
    """Update the boundary cells. 
    
        Implement MUR boundary conditions. This represents an open boundary were waves can leave the
        simulation domain with little remaining reflection artifacts. Although this is of a low error 
        order it is good enough for this simulation.
    """
    c = dimx-1
    u[0, dimx-sz-1:c, 1:dimy-1] = u[1,  dimx-sz-2:c-1, 1:dimy-1] + (kappa[dimx-sz-1:c, 1:dimy-1]-1)/(kappa[ dimx-sz-1:c, 1:dimy-1]+1) * (u[0,  dimx-sz-2:c-1,1:dimy-1] - u[1, dimx-sz-1:c,1:dimy-1])
    
    c = 0
    u[0, c:sz, 1:dimy-1]        = u[1, c+1:sz+1, 1:dimy-1]       + (kappa[c:sz, 1:dimy-1]-1)/(kappa[c:sz, 1:dimy-1]+1)                * (u[0, c+1:sz+1,1:dimy-1]       - u[1,c:sz,1:dimy-1])

    r = dimy-1
    u[0, 1:dimx-1, dimy-1-sz:r] = u[1, 1:dimx-1, dimy-2-sz:r-1] + (kappa[1:dimx-1, dimy-1-sz:r]-1)/(kappa[1:dimx-1, dimy-1-sz:r]+1) * (u[0, 1:dimx-1, dimy-2-sz:r-1] - u[1, 1:dimx-1, dimy-1-sz:r])

    r = 0
    u[0, 1:dimx-1, r:sz] = u[1, 1:dimx-1, r+1:sz+1] + (kappa[1:dimx-1, r:sz]-1)/(kappa[1:dimx-1, r:sz]+1) * (u[0, 1:dimx-1, r+1:sz+1] - u[1, 1:dimx-1, r:sz])

def place_raindrops(u, uu, tick):
    if (random.random()<0.01):
        w,h = gauss_peak.shape
        x = int(random.randrange(w+w/2, dimx-h-h/2))
        y = int(random.randrange(w+w/2, dimy-h-h/2))

        height = 10
        put_gauss_peak(u, x, y, height)
        put_gauss_peak(uu, x, y, height)

def main():
    pygame.init()
    pygame.font.init()
                   
    my_font = pygame.font.SysFont('Consolas', 15)    
    display = pygame.display.set_mode((2*dimx*cellsize, dimy*cellsize))
    pygame.display.set_caption("Solving the 2d Wave Equation")

    create_arrays()
    set_initial_conditions(u)
    set_initial_conditions(uu)

    pixeldata = np.zeros((3*dimx, dimy, 3), dtype=np.uint8 )

    tick = 0
    last_tick = 0
    fps = 0
    start_time = time.time()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        tick = tick + 1
        #place_raindrops(u, uu, tick)

        update(u, 0)
        update(uu, 4)        

        vfak = 4
        pixeldata[1:dimx, 1:dimy, 0] = np.clip(u[0, 1:dimx, 1:dimy] + 128 * vfak * velocity[1:dimx, 1:dimy], 0, 255)
        pixeldata[1:dimx, 1:dimy, 1] = np.clip(u[1, 1:dimx, 1:dimy] + 128 * vfak * velocity[1:dimx, 1:dimy], 0, 255)
        pixeldata[1:dimx, 1:dimy, 2] = np.clip(u[2, 1:dimx, 1:dimy] + 128 * vfak * velocity[1:dimx, 1:dimy], 0, 255)

        pixeldata[dimx+1:2*dimx, 1:dimy, 0] = np.clip(uu[0, 1:dimx, 1:dimy] + 128 * vfak * velocity[1:dimx, 1:dimy], 0, 255)
        pixeldata[dimx+1:2*dimx, 1:dimy, 1] = np.clip(uu[1, 1:dimx, 1:dimy] + 128 * vfak * velocity[1:dimx, 1:dimy], 0, 255) 
        pixeldata[dimx+1:2*dimx, 1:dimy, 2] = np.clip(uu[2, 1:dimx, 1:dimy] + 128 * vfak * velocity[1:dimx, 1:dimy], 0, 255) 

        surf = pygame.surfarray.make_surface(pixeldata)
        display.blit(pygame.transform.scale(surf, (3*dimx * cellsize, dimy * cellsize)), (0, 0))

        text_surface = my_font.render('2D Wave Equation - Explicit Euler (Radiating Boundary Conditions)', True, (255, 255, 255))
        display.blit(text_surface, (5,5))

        current_time = time.time() 
        if current_time - start_time > 0.5:
            fps = (tick-last_tick) / (current_time - start_time) 
            start_time = time.time()
            last_tick = tick

        text_surface = my_font.render(f'FPS: {fps:.1f}; t={tick*ts:.2f} s; area={dimx*hs}x{dimy*hs} m', True, (255, 255, 255))
        display.blit(text_surface, (5, dimy*cellsize - 20))

        pygame.display.update()

if __name__ == "__main__":
    main()