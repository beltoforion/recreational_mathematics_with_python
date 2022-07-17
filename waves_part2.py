import pygame
import numpy as np
import random
import math

h = 1        # spatial step width
k = 1        # time step width
dimx = 400   # width of the simulation domain
dimy = 400   # height of the simulation domain
cellsize = 1 # display size of a cell in pixel

def init_simulation():
    u = np.zeros((3, dimx, dimy))   # The three dimensional simulation grid 
    c = 0.35                         # The "original" wave propagation speed
    tau = ( (c*k) / h )**2          # wave propagation speed scaled to the step widths
    alpha = np.zeros((dimx, dimy))  # wave propagation velocities of the entire simulation domain
    alpha[0:dimx,0:dimy] = tau      # will be set to a constant value of tau

#    alpha[int(dimx/5)-2:int(dimx/5)+2, 0:int(dimy/2-15)] = 0
#    alpha[int(dimx/5)-2:int(dimx/5)+2, int(dimy/2+15):dimy] = 0    
#    alpha[500:520, 0:150] = 0
#    alpha[900:920, 70:dimy] = 0
#    alpha[0:int(dimx/2), 0:dimy] = tau/3

    return u, alpha


def update(u, alpha, algo):
    buf = u[2]
    u[2] = u[1]
    u[1] = u[0]
    u[0] = buf

    if algo==0: # ok
        u[0, 1:dimx-1, 1:dimy-1] = alpha[1:dimx-1, 1:dimy-1] \
                                   * (        u[1, 1:dimx-1, 0:dimy-2] # c, r-1 =>  1

                                        +     u[1, 0:dimx-2, 1:dimy-1] # c-1, r =>  1
                                        - 4 * u[1, 1:dimx-1, 1:dimy-1] # c,   r => -4
                                        +     u[1, 2:dimx  , 1:dimy-1] # c+1, r =>  1

                                        +     u[1, 1:dimx-1, 2:dimy]   # c, r+1 =>  1
                                     ) \
                                + 2 * u[1, 1:dimx-1, 1:dimy-1] \
                                -     u[2, 1:dimx-1, 1:dimy-1]
    elif algo==1: # ok, sieht einen tick besser aus
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
    elif algo==2: # NICHT OK!
        # zu schnell
        # Quelle: https://stackoverflow.com/questions/8070586/bigger-mask-size-for-laplacian-filter
        u[0, 1:dimx-1, 1:dimy-1] = alpha[1:dimx-1, 1:dimy-1] \
                                   * (  - 0.157807 * u[1, 0:dimx-2, 0:dimy-2] # c-1, r-1 =>  1
                                        + 1.84219  * u[1, 1:dimx-1, 0:dimy-2] # c,   r-1 =>  1
                                        - 0.157807 * u[1, 2:dimx  , 0:dimy-2] # c+1, r-1 =>  1

                                        + 1.84219  * u[1, 0:dimx-2, 1:dimy-1] # c-1, r =>  1
                                        - 6.73754  * u[1, 1:dimx-1, 1:dimy-1] # c,   r => -8
                                        + 1.84219  * u[1, 2:dimx  , 1:dimy-1] # c+1, r =>  1

                                        - 0.157807 * u[1, 0:dimx-2, 2:dimy]   # c-1, r+1 =>  1
                                        + 1.84219  * u[1, 1:dimx-1, 2:dimy]   # c,   r+1 =>  1
                                        - 0.157807 * u[1, 2:dimx  , 2:dimy]   # c+1, r+1 =>  1
                                     ) \
                                + 2 * u[1, 1:dimx-1, 1:dimy-1] \
                                -     u[2, 1:dimx-1, 1:dimy-1]
    elif algo==3: # ok
        # Quelle: Machnumber.java
        # ok, aber kein Unterschied erkennbar
        u[0, 1:dimx-1, 1:dimy-1] = alpha[1:dimx-1, 1:dimy-1] \
                                   * (          1  * u[1, 0:dimx-2, 0:dimy-2] # c-1, r-1 =>  1
                                        +       4  * u[1, 1:dimx-1, 0:dimy-2] # c,   r-1 =>  1
                                        +       1  * u[1, 2:dimx  , 0:dimy-2] # c+1, r-1 =>  1

                                        +       4  * u[1, 0:dimx-2, 1:dimy-1] # c-1, r =>  1
                                        -      20  * u[1, 1:dimx-1, 1:dimy-1] # c,   r => -8
                                        +       4  * u[1, 2:dimx  , 1:dimy-1] # c+1, r =>  1

                                        +       1  * u[1, 0:dimx-2, 2:dimy]   # c-1, r+1 =>  1
                                        +       4  * u[1, 1:dimx-1, 2:dimy]   # c,   r+1 =>  1
                                        +       1  * u[1, 2:dimx  , 2:dimy]   # c+1, r+1 =>  1
                                     ) / 6 \
                                + 2 * u[1, 1:dimx-1, 1:dimy-1] \
                                -     u[2, 1:dimx-1, 1:dimy-1]
    elif algo==4:
        u[0, 2:dimx-2, 2:dimy-2]  = alpha[2:dimx-2, 2:dimy-2]\
                                    * (        u[1, 2:dimx-2, 0:dimy-4]  # c,   r-2 => 1

                                        +      u[1, 1:dimx-3, 1:dimy-3]  # c-1, r-1 => 1
                                        +  2 * u[1, 2:dimx-2, 1:dimy-3]  # c  , r-1 => 2                                       
                                        +      u[1, 3:dimx-1, 1:dimy-3]  # c+1, r-1 => 1

                                        +      u[1, 0:dimx-4, 2:dimy-2] # c - 2, r => 1
                                        +  2 * u[1, 1:dimx-3, 2:dimy-2] # c - 1, r => 2
                                        - 17 * u[1, 2:dimx-2, 2:dimy-2] # c    , r => -17
                                        +  2 * u[1, 3:dimx-1, 2:dimy-2] # c+1  , r => 2
                                        +      u[1, 4:dimx,   2:dimy-2] # c+2  , r => 1

                                        +      u[1, 1:dimx-3, 3:dimy-1]  # c-1, r+1 => 1
                                        +  2 * u[1, 2:dimx-2, 3:dimy-1]  # c  , r+1 => 2                                       
                                        +      u[1, 3:dimx-1, 3:dimy-1]  # c+1, r+1 => 1

                                        +      u[1, 2:dimx-2, 4:dimy]  # c,   r+2 => 1 
                                        ) \
                                    + 2*u[1, 2:dimx-2, 2:dimy-2] \
                                    -   u[2, 2:dimx-2, 2:dimy-2]
    elif algo==5:
        u[0, 2:dimx-2, 2:dimy-2]  = alpha[2:dimx-2, 2:dimy-2]\
                                    * (        u[1, 2:dimx-2, 0:dimy-4]  # c,   r-2 => 1

                                        +      u[1, 1:dimx-3, 1:dimy-3]  # c-1, r-1 => 1
                                        +  2 * u[1, 2:dimx-2, 1:dimy-3]  # c  , r-1 => 2                                       
                                        +      u[1, 3:dimx-1, 1:dimy-3]  # c+1, r-1 => 1

                                        +      u[1, 0:dimx-4, 2:dimy-2] # c - 2, r => 1
                                        +  2 * u[1, 1:dimx-3, 2:dimy-2] # c - 1, r => 2
                                        - 17 * u[1, 2:dimx-2, 2:dimy-2] # c    , r => -17
                                        +  2 * u[1, 3:dimx-1, 2:dimy-2] # c+1  , r => 2
                                        +      u[1, 4:dimx,   2:dimy-2] # c+2  , r => 1

                                        +      u[1, 1:dimx-3, 3:dimy-1]  # c-1, r+1 => 1
                                        +  2 * u[1, 2:dimx-2, 3:dimy-1]  # c  , r+1 => 2                                       
                                        +      u[1, 3:dimx-1, 3:dimy-1]  # c+1, r+1 => 1

                                        +      u[1, 2:dimx-2, 4:dimy]  # c,   r+2 => 1 
                                        ) \
                                    + 2*u[1, 2:dimx-2, 2:dimy-2] \
                                    -   u[2, 2:dimx-2, 2:dimy-2]                                    

    u[0, 1:dimx-1, 1:dimy-1] *= 0.998


def place_raindrops(u, uu, tick):
#    u[0, 50, int(dimy/2)-20] = math.sin(tick * 0.4) * 200
#    u[0, 50, int(dimy/2)+20] = math.sin(tick * 0.4)* 200
#    grid[0, 50, 0:dimy] = math.sin(tick * 0.1) * 200

    if (random.random()<0.02):
        x = random.randrange(5, dimx-5)
        y = random.randrange(5, dimy-5)
        u[0, x-3:x+3, y-3:y+3] = 120
        uu[0, x-3:x+3, y-3:y+3] = 120


def main():
    pygame.init()
    display = pygame.display.set_mode((3*dimx*cellsize, dimy*cellsize))
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
        update(uu, alpha, 3)

        pixeldata[1:dimx, 1:dimy, 0] = np.clip(u[0, 1:dimx, 1:dimy] + 128, 0, 255)
        pixeldata[1:dimx, 1:dimy, 1] = np.clip(u[0, 1:dimx, 1:dimy] + 128, 0, 255)
        pixeldata[1:dimx, 1:dimy, 2] = np.clip(u[0, 1:dimx, 1:dimy] + 128, 0, 255)

        pixeldata[dimx+1:2*dimx, 1:dimy, 0] = np.clip(uu[0, 1:dimx, 1:dimy] + 128, 0, 255)
        pixeldata[dimx+1:2*dimx, 1:dimy, 1] = np.clip(uu[0, 1:dimx, 1:dimy] + 128, 0, 255) 
        pixeldata[dimx+1:2*dimx, 1:dimy, 2] = np.clip(uu[0, 1:dimx, 1:dimy] + 128, 0, 255) 

        pixeldata[2*dimx+1:3*dimx, 1:dimy, 0] = np.clip(uu[0, 1:dimx, 1:dimy] - u[0, 1:dimx, 1:dimy] + 128, 0, 255)
        pixeldata[2*dimx+1:3*dimx, 1:dimy, 1] = np.clip(uu[0, 1:dimx, 1:dimy] - u[0, 1:dimx, 1:dimy]+ 128, 0, 255) 
        pixeldata[2*dimx+1:3*dimx, 1:dimy, 2] = np.clip(uu[0, 1:dimx, 1:dimy] - u[0, 1:dimx, 1:dimy]+ 128, 0, 255) 



        surf = pygame.surfarray.make_surface(pixeldata)
        display.blit(pygame.transform.scale(surf, (3*dimx * cellsize, dimy * cellsize)), (0, 0))
        pygame.display.update()

if __name__ == "__main__":
    main()