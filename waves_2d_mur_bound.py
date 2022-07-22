import pygame
import numpy as np
import random
import math

h = 1        # spatial step width
k = 1        # time step width
c = 0.4  
dimx = 300   # width of the simulation domain
dimy = 300   # height of the simulation domain
cellsize = 2 # display size of a cell in pixel

def init_simulation():
    u = np.zeros((5, dimx, dimy))   # The three dimensional simulation grid 
                          # The "original" wave propagation speed
    tau = ( (c*k) / h )**2          # wave propagation speed scaled to the step widths
    alpha = np.zeros((dimx, dimy))  # wave propagation velocities of the entire simulation domain
    alpha[0:dimx,0:dimy] = tau      # will be set to a constant value of tau

    # Create a template for a gauss peak to use as a rain drop model
    global gauss_peak
    sz = 10
    sigma=3.0
    xx, yy = np.meshgrid(range(-sz, sz), range(-sz, sz))
    gauss_peak = np.zeros((sz, sz), dtype=np.float)
    gauss_peak = 300 * 1 / (sigma*2*math.pi) * (math.sqrt(2*math.pi)) * np.exp(- 0.5 * ((xx**2+yy**2)/(sigma**2)))

    x = int(dimx/2)
    y = int(dimy/2)
    u[0, x-sz:x+sz, y-sz:y+sz] += 10 * gauss_peak
    u[1, x-sz:x+sz, y-sz:y+sz] += 10 * gauss_peak
    u[2, x-sz:x+sz, y-sz:y+sz] += 10 * gauss_peak    
    u[3, x-sz:x+sz, y-sz:y+sz] += 10 * gauss_peak    
    u[4, x-sz:x+sz, y-sz:y+sz] += 10 * gauss_peak    
    return u, alpha

def place_raindrops(u, uu):
    if (random.random()<0.02):
        x = random.randrange(10, dimx-10)
        y = random.randrange(10, dimy-10)
        u[0:4, x-2:x+2, y-2:y+2] = 120
        uu[0:4, x-2:x+2, y-2:y+2] = 120        


def update(u, alpha, algo):
    u[4] = u[3]
    u[3] = u[2]
    u[2] = u[1]
    u[1] = u[0]

    tau = ( (0.4*k) / h )**2  

    if algo==0: # ok, 2nd Order
        u[0, 1:dimx-1, 1:dimy-1] = tau \
                                   * (        u[1, 1:dimx-1, 0:dimy-2] # c, r-1 =>  1

                                        +     u[1, 0:dimx-2, 1:dimy-1] # c-1, r =>  1
                                        - 4 * u[1, 1:dimx-1, 1:dimy-1] # c,   r => -4
                                        +     u[1, 2:dimx  , 1:dimy-1] # c+1, r =>  1

                                        +     u[1, 1:dimx-1, 2:dimy]   # c, r+1 =>  1
                                     ) \
                                + 2 * u[1, 1:dimx-1, 1:dimy-1] \
                                -     u[2, 1:dimx-1, 1:dimy-1]

        fak2 = k*0.4/h                
        for r in range(1, dimy-1):
            c=dimx-1
            u[0, c, r] = u[1, c-1, r] + (fak2-1)/(fak2+1) * (u[0, c-1,r] - u[1,c,r])

            c=0
            u[0, c, r] = u[1, c+1, r] + (fak2-1)/(fak2+1) * (u[0, c+1,r] - u[1,c,r])

        for c in range(1, dimy-1):
            r=dimx-1
            u[0, c, r] = u[1, c, r-1] + (fak2-1)/(fak2+1) * (u[0, c,r-1] - u[1,c,r])

            r=0
            u[0, c, r] = u[1, c, r+1] + (fak2-1)/(fak2+1) * (u[0, c,r+1] - u[1,c,r])

    if algo==1: 
        u[0, 1:dimx-1, 1:dimy-1] = tau \
                                   * (        u[1, 1:dimx-1, 0:dimy-2] # c, r-1 =>  1

                                        +     u[1, 0:dimx-2, 1:dimy-1] # c-1, r =>  1
                                        - 4 * u[1, 1:dimx-1, 1:dimy-1] # c,   r => -4
                                        +     u[1, 2:dimx  , 1:dimy-1] # c+1, r =>  1

                                        +     u[1, 1:dimx-1, 2:dimy]   # c, r+1 =>  1
                                     ) \
                                + 2 * u[1, 1:dimx-1, 1:dimy-1] \
                                -     u[2, 1:dimx-1, 1:dimy-1]
    if algo==2: #
        u[0, 1:dimx-1, 1:dimy-1] = tau \
                                   * (        u[1, 1:dimx-1, 0:dimy-2] # c, r-1 =>  1

                                        +     u[1, 0:dimx-2, 1:dimy-1] # c-1, r =>  1
                                        - 4 * u[1, 1:dimx-1, 1:dimy-1] # c,   r => -4
                                        +     u[1, 2:dimx  , 1:dimy-1] # c+1, r =>  1

                                        +     u[1, 1:dimx-1, 2:dimy]   # c, r+1 =>  1
                                     ) \
                                + 2 * u[1, 1:dimx-1, 1:dimy-1] \
                                -     u[2, 1:dimx-1, 1:dimy-1]                                


def main():
    pygame.init()
    pygame.font.init()
                   
    my_font = pygame.font.SysFont('Consolas', 15)    
    display = pygame.display.set_mode((2*dimx*cellsize, dimy*cellsize))
    pygame.display.set_caption("Solving the 2d Wave Equation")

    u, alpha = init_simulation()
    uu, alpha = init_simulation()
    update(u, alpha, 0)
    uu[1] = np.copy(u[0])
    update(u, alpha, 0)
    uu[2]  = np.copy(u[0])
    update(u, alpha, 0)
    uu[3] = np.copy(u[0])
    update(u, alpha, 0)
    uu[4] = np.copy(u[0])

    pixeldata = np.zeros((3*dimx, dimy, 3), dtype=np.uint8 )

    tick = 0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    uu[0] = np.copy(u[0])
                    uu[1] = np.copy(u[1])
                    uu[2] = np.copy(u[2])
                    uu[3] = np.copy(u[3])
                    uu[4] = np.copy(u[4])

        tick = tick + 1
        place_raindrops(u, uu)

        update(u, alpha, 0)
        update(uu, alpha, 1)

        pixeldata[1:dimx, 1:dimy, 0] = np.clip(u[0, 1:dimx, 1:dimy] + 128, 0, 255)
        pixeldata[1:dimx, 1:dimy, 1] = np.clip(u[1, 1:dimx, 1:dimy] + 128, 0, 255)
        pixeldata[1:dimx, 1:dimy, 2] = np.clip(u[2, 1:dimx, 1:dimy] + 128, 0, 255)

        pixeldata[dimx+1:2*dimx, 1:dimy, 0] = np.clip(uu[0, 1:dimx, 1:dimy] + 128, 0, 255)
        pixeldata[dimx+1:2*dimx, 1:dimy, 1] = np.clip(uu[1, 1:dimx, 1:dimy] + 128, 0, 255) 
        pixeldata[dimx+1:2*dimx, 1:dimy, 2] = np.clip(uu[2, 1:dimx, 1:dimy] + 128, 0, 255) 

        surf = pygame.surfarray.make_surface(pixeldata)
        display.blit(pygame.transform.scale(surf, (3*dimx * cellsize, dimy * cellsize)), (0, 0))

        text_surface = my_font.render('2D Wave Equation', True, (255, 255, 255))
        display.blit(text_surface, (5,5))

        pygame.display.update()

if __name__ == "__main__":
    main()