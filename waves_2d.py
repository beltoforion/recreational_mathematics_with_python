import pygame
import numpy as np

h = 1        # spatial step width
k = 1        # time step width
c = 0.3      # wave velocity
dimx = dimy = int(600)
tau = ( (c*k) / h )**2
kappa = k * c / h  
cellsize = 1

# Place Obstacles
alpha = np.zeros((dimx, dimy))
alpha[:] = tau
alpha[560:570, 0:180] = alpha[560:570, 220:380] = alpha[560:570, 410:dimy] = 0
alpha[260:280, 400:500] = alpha[120:140, 120:220] = alpha[360:380, 60:160] = 0

def update(u):
    u[2] = u[1]
    u[1] = u[0]

    # Solving the Wave Equation
    u[0, 2:dimx-2, 2:dimy-2]  = alpha[2:dimx-2, 2:dimy-2] * ( -  1 * u[1, 2:dimx-2, 0:dimy-4] + 16 * u[1, 2:dimx-2, 1:dimy-3]
                                          -  1 * u[1, 0:dimx-4, 2:dimy-2] + 16 * u[1, 1:dimx-3, 2:dimy-2] 
                                          - 60 * u[1, 2:dimx-2, 2:dimy-2] + 16 * u[1, 3:dimx-1, 2:dimy-2]
                                          -  1 * u[1, 4:dimx,   2:dimy-2] + 16 * u[1, 2:dimx-2, 3:dimy-1] 
                                          -  1 * u[1, 2:dimx-2, 4:dimy] ) / 12 \
                                + 2*u[1, 2:dimx-2, 2:dimy-2] -   u[2, 2:dimx-2, 2:dimy-2]                                     

    # Absorbing Boundary Conditions
    u[0, dimx-3:dimx-1, 1:dimy-1] = u[1,  dimx-4:dimx-2, 1:dimy-1] + (kappa-1)/(kappa+1) * (u[0,  dimx-4:dimx-2, 1:dimy-1] - u[1, dimx-3:dimx-1,1:dimy-1])
    u[0,           0:2, 1:dimy-1] = u[1,            1:3, 1:dimy-1] + (kappa-1)/(kappa+1) * (u[0,            1:3, 1:dimy-1] - u[1,0:2,1:dimy-1])
    u[0, 1:dimx-1, dimy-3:dimy-1] = u[1,  1:dimx-1, dimy-4:dimy-2] + (kappa-1)/(kappa+1) * (u[0, 1:dimx-1,  dimy-4:dimy-2] - u[1, 1:dimx-1, dimy-3:dimy-1])
    u[0, 1:dimx-1, 0:2] = u[1, 1:dimx-1, 1:3] + (kappa-1)/(kappa+1) * (u[0, 1:dimx-1, 1:3] - u[1, 1:dimx-1, 0:2])

def main():
    pygame.init()
    display = pygame.display.set_mode((dimx*cellsize, dimy*cellsize))
    pygame.display.set_caption("Solving the 2d Wave Equation")

    u = np.zeros((3, dimx, dimy))
    pixeldata = np.zeros((dimx, dimy, 3), dtype=np.uint8 )

    tick = 0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        tick += 1
        u[0:2, 590:598, 0:dimy] = np.sin(tick * 0.08) * 20
        update(u)

        pixeldata[1:dimx, 1:dimy, 0] = np.clip((u[0, 1:dimx, 1:dimy]>0) * 10 * u[0, 1:dimx, 1:dimy]+u[1, 1:dimx, 1:dimy]+u[2, 1:dimx, 1:dimy], 0, 255)
        pixeldata[1:dimx, 1:dimy, 2] = 0 #255-np.clip(np.abs(u[0, 1:dimx, 1:dimy]) * 10, 0, 255)
        pixeldata[1:dimx, 1:dimy, 1] = np.clip((u[0, 1:dimx, 1:dimy]<=0) * -10 * u[0, 1:dimx, 1:dimy] + u[1, 1:dimx, 1:dimy] + u[2, 1:dimx, 1:dimy], 0, 255) 

        surf = pygame.surfarray.make_surface(pixeldata)
        display.blit(pygame.transform.scale(surf, (dimx * cellsize, dimy * cellsize)), (0, 0))
        pygame.display.update()

if __name__ == "__main__":
    main()