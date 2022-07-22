import pygame
import numpy as np
import random
import math

h = 1        # spatial step width
k = 1        # time step width
dimx = 200   # width of the simulation domain
dimy = 200   # height of the simulation domain
cellsize = 3 # display size of a cell in pixel

def init_simulation():
    u = np.zeros((3, dimx, dimy))   # The three dimensional simulation grid 
    c = 0.4                         # The "original" wave propagation speed
    tau = ( (c*k) / h )**2          # wave propagation speed scaled to the step widths
    alpha = np.zeros((dimx, dimy))  # wave propagation velocities of the entire simulation domain
    alpha[0:dimx,0:dimy] = tau      # will be set to a constant value of tau

    # alpha[100:150,50:dimy-50] = tau/2      # will be set to a constant value of tau
    # alpha[150:200,50:dimy-50] = tau/4      # will be set to a constant value of tau
#    alpha[200:,0:dimy] = 2*tau     # will be set to a constant value of tau

    # Create a template for a gauss peak to use as a rain drop model
    global gauss_peak
    sz = 10
    sigma=2.0
    xx, yy = np.meshgrid(range(-sz, sz), range(-sz, sz))
    gauss_peak = np.zeros((sz, sz), dtype=np.float)
    gauss_peak = 300 * 1 / (sigma*2*math.pi) * (math.sqrt(2*math.pi)) * np.exp(- 0.5 * ((xx**2+yy**2)/(sigma**2)))

    x = int(dimx/2)
    y = int(dimy/2)
    u[0:2, x-sz:x+sz, y-sz:y+sz] += 10 * gauss_peak

    return u, alpha


def update(u, alpha, algo):
    u[2] = u[1]
    u[1] = u[0]

    u[0, 1:dimx-1, 1:dimy-1] = alpha[1:dimx-1, 1:dimy-1] \
                                * (   0.25 * u[1, 0:dimx-2, 0:dimy-2] # c-1, r-1 =>  1
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

    if algo==0: # ok, 2nd Order
        u[0, 1:dimx-1, 1:dimy-1] = alpha[1:dimx-1, 1:dimy-1] \
                                   * (        u[1, 1:dimx-1, 0:dimy-2] # c, r-1 =>  1

                                        +     u[1, 0:dimx-2, 1:dimy-1] # c-1, r =>  1
                                        - 4 * u[1, 1:dimx-1, 1:dimy-1] # c,   r => -4
                                        +     u[1, 2:dimx  , 1:dimy-1] # c+1, r =>  1

                                        +     u[1, 1:dimx-1, 2:dimy]   # c, r+1 =>  1
                                     ) \
                                + 2 * u[1, 1:dimx-1, 1:dimy-1] \
                                -     u[2, 1:dimx-1, 1:dimy-1]
    elif algo==1: # ok, 2nd Order
        u[0, 1:dimx-1, 1:dimy-1] = alpha[1:dimx-1, 1:dimy-1] \
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
    elif algo==2: # ok, (4)th Order https://www.ams.org/journals/mcom/1988-51-184/S0025-5718-1988-0935077-0/S0025-5718-1988-0935077-0.pdf; Page 702
        u[0, 2:dimx-2, 2:dimy-2]  = alpha[2:dimx-2, 2:dimy-2]\
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
    elif algo==3: # ok, (6th) https://www.ams.org/journals/mcom/1988-51-184/S0025-5718-1988-0935077-0/S0025-5718-1988-0935077-0.pdf; Page 702
        u[0, 3:dimx-3, 3:dimy-3]  = alpha[3:dimx-3, 3:dimy-3]\
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
    elif algo==4: # ok, (8th) https://www.ams.org/journals/mcom/1988-51-184/S0025-5718-1988-0935077-0/S0025-5718-1988-0935077-0.pdf; Page 702
        u[0, 4:dimx-4, 4:dimy-4]  = alpha[4:dimx-4, 4:dimy-4]\
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

    # Absorbind Boundary Conditions:
    mur = False
    if mur==True:
        fak2 = k * 0.35 / h                
        c = dimx-1
        u[0, c, 1:dimy-1] = u[1, c-1, 1:dimy-1] + (fak2-1)/(fak2+1) * (u[0, c-1,1:dimy-1] - u[1,c,1:dimy-1])
        
        c = 0
        u[0, c, 1:dimy-1] = u[1, c+1, 1:dimy-1] + (fak2-1)/(fak2+1) * (u[0, c+1,1:dimy-1] - u[1,c,1:dimy-1])

        r = dimy-1
        u[0, 1:dimx-1, r] = u[1, 1:dimx-1, r-1] + (fak2-1)/(fak2+1) * (u[0, 1:dimx-1, r-1] - u[1, 1:dimx-1, r])

        r = 0
        u[0, 1:dimx-1, r] = u[1, 1:dimx-1, r+1] + (fak2-1)/(fak2+1) * (u[0, 1:dimx-1, r+1] - u[1, 1:dimx-1, r])


def place_raindrops(u, uu, tick):
    if (random.random()<0.01):
        w,h = gauss_peak.shape
        w = int(w/2)
        h = int(h/2)
        height = 10
        x = random.randrange(w+10, dimx-h-10)
        y = random.randrange(w+10, dimy-h-10)
        u[0:3, x-w:x+h, y-w:y+h] += height * gauss_peak
        uu[0:3, x-w:x+h, y-w:y+h] += height * gauss_peak

def main():
    pygame.init()
    pygame.font.init()
                   
    my_font = pygame.font.SysFont('Consolas', 15)    
    display = pygame.display.set_mode((2*dimx*cellsize, dimy*cellsize))
    pygame.display.set_caption("Solving the 2d Wave Equation")

    u, alpha = init_simulation()
    uu, alpha = init_simulation()

    pixeldata = np.zeros((3*dimx, dimy, 3), dtype=np.uint8 )

    tick = 0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        tick = tick + 1
        place_raindrops(u, uu, tick)

        update(u, alpha, 1)
        update(uu, alpha, 4)

        pixeldata[1:dimx, 1:dimy, 0] = np.clip(u[0, 1:dimx, 1:dimy] + 128, 0, 255)
        pixeldata[1:dimx, 1:dimy, 1] = np.clip(u[1, 1:dimx, 1:dimy] + 128, 0, 255)
        pixeldata[1:dimx, 1:dimy, 2] = np.clip(u[2, 1:dimx, 1:dimy] + 128, 0, 255)

        pixeldata[dimx+1:2*dimx, 1:dimy, 0] = np.clip(uu[0, 1:dimx, 1:dimy] + 128, 0, 255)
        pixeldata[dimx+1:2*dimx, 1:dimy, 1] = np.clip(uu[1, 1:dimx, 1:dimy] + 128, 0, 255) 
        pixeldata[dimx+1:2*dimx, 1:dimy, 2] = np.clip(uu[2, 1:dimx, 1:dimy] + 128, 0, 255) 

        surf = pygame.surfarray.make_surface(pixeldata)
        display.blit(pygame.transform.scale(surf, (3*dimx * cellsize, dimy * cellsize)), (0, 0))

        text_surface = my_font.render('2D Wave Equation - Explicit Euler', True, (255, 255, 255))
        display.blit(text_surface, (5,5))

        pygame.display.update()

if __name__ == "__main__":
    main()