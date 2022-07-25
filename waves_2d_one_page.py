import pygame
import numpy as np
import random

h = 1        # spatial step width
k = 1        # time step width
c = 0.2
kappa = k * c / h  
alpha = ( (c*k) / h )**2
dimx = int(300)   # width of the simulation domain
dimy = int(300)   # height of the simulation domain
cellsize = 2 # display size of a cell in pixel

def init_simulation():
    u = np.zeros((3, dimx, dimy))   # The three dimensional simulation grid 
    u[0:2, 295:299, 0:dimy] = 200
    return u

def update(u):
    u[2] = u[1]
    u[1] = u[0]

    # Solving the Wave Equation
    u[0, 1:dimx-1, 1:dimy-1]  = alpha * (u[1, 0:dimx-2, 1:dimy-1] + 
                                         u[1, 2:dimx,   1:dimy-1] + 
                                         u[1, 1:dimx-1, 0:dimy-2] + 
                                         u[1, 1:dimx-1, 2:dimy] - 4*u[1, 1:dimx-1, 1:dimy-1]) \
                                    + 2 * u[1, 1:dimx-1, 1:dimy-1] - u[2, 1:dimx-1, 1:dimy-1]
    u[0, 1:dimx-1, 1:dimy-1] *= 0.9995                                   

    # Place Walls
    u[0:2, 240:250, 10:90] = 0
    u[0:2, 240:250, 110:190] = 0    
    u[0:2, 240:250, 210:290] = 0        

#    u[0:2, 80:90, int(dimy/2)-30:int(dimy/2)+30] = 0
#    u[0:2, 140:210, int(dimy/2)-20] = 0
#    u[0:2, 140:210, int(dimy/2)+20] = 0

    # Absorbing Boundary Conditions on the left hand side
    u[0, 0, 1:dimy-1]      = u[1,      1, 1:dimy-1] + (kappa-1)/(kappa+1) * (u[0,      1, 1:dimy-1] - u[1,      1, 1:dimy-1])

def main():
    pygame.init()
    display = pygame.display.set_mode((dimx*cellsize, dimy*cellsize))
    pygame.display.set_caption("Solving the 2d Wave Equation")

    u = init_simulation()
    pixeldata = np.zeros((dimx, dimy, 3), dtype=np.uint8 )

    tick = 0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        tick += 1
    
#        u[0:2, 260, int(150 + np.sin(tick * 0.001) * 100)] = np.sin(tick * 0.1) * 200
        u[0:2, 295:299, 0:dimy] = np.sin(tick * 0.1) * 20
        update(u)

        pixeldata[1:dimx, 1:dimy, 0] = np.clip((u[0, 1:dimx, 1:dimy]>0) * 5 * u[0, 1:dimx, 1:dimy]+u[1, 1:dimx, 1:dimy]+u[2, 1:dimx, 1:dimy], 0, 255)
        pixeldata[1:dimx, 1:dimy, 1] = 0 
        pixeldata[1:dimx, 1:dimy, 2] = np.clip((u[0, 1:dimx, 1:dimy]<=0) * -5 * u[0, 1:dimx, 1:dimy] + u[1, 1:dimx, 1:dimy] + u[2, 1:dimx, 1:dimy], 0, 255)

        surf = pygame.surfarray.make_surface(pixeldata)
        display.blit(pygame.transform.scale(surf, (dimx * cellsize, dimy * cellsize)), (0, 0))
        pygame.display.update()

if __name__ == "__main__":
    main()