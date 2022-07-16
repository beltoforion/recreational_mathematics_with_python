import pygame
import numpy as np

grid = None
h = 1
k = 1
c = 0.7
tau = (c * c * k * k) / (h * h)

def init(dimx, dimy):
    global grid
    grid = np.zeros((3, dimx, dimy))

    sigma=5
    xx, yy = np.meshgrid(range(0, dimx), range(0, dimy))
    grid[0:2, xx, yy] = 400 * np.exp(-(np.sqrt((dimx/2-xx)**2 + (dimy/2-yy)**2)**2)/(2*sigma**2))

def update(dimx, dimy):
    buf = grid[2]
    grid[2] = grid[1]
    grid[1] = grid[0]
    grid[0] = buf

#    for c in range(1, dimx-1):
#        for r in range(1, dimy-1):
#            grid[0, c, r]  = tau * (grid[1, c-1, r] + grid[1, c+1, r] + grid[1, c, r-1] + grid[1, c, r+1] - 4*grid[1, c, r])
#            grid[0, c, r] += 2 * grid[1, c, r] - grid[2, c, r]

    grid[0, 1:dimx-1, 1:dimy-1]  = tau * (grid[1, 0:dimx-2, 1:dimy-1] + \
                                          grid[1, 2:dimx,   1:dimy-1] + \
                                          grid[1, 1:dimx-1, 0:dimy-2] + \
                                          grid[1, 1:dimx-1, 2:dimy] - 4*grid[1, 1:dimx-1, 1:dimy-1]) \
                                    + 2 * grid[1, 1:dimx-1, 1:dimy-1] - grid[2, 1:dimx-1, 1:dimy-1]
    grid[0, 1:dimx-1, 1:dimy-1] *= 0.998

def main(dimx, dimy, cellsize):
    pygame.init()

    display = pygame.display.set_mode((dimx*cellsize, dimy*cellsize))
    pygame.display.set_caption("Water Effect - Solving the 2d Wave Equation")

    init(dimx, dimy)
    pixeldata = np.zeros((dimx, dimy, 3), dtype=np.uint8 )

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        update(dimx, dimy)

        pixeldata[1:dimx, 1:dimy, 0] = grid[0, 1:dimx, 1:dimy] + 100
        pixeldata[1:dimx, 1:dimy, 1] = grid[0, 1:dimx, 1:dimy] + 100
        pixeldata[1:dimx, 1:dimy, 2] = grid[0, 1:dimx, 1:dimy] + 100

        surf = pygame.surfarray.make_surface(pixeldata)
        display.blit(pygame.transform.scale(surf, (dimx * cellsize, dimy * cellsize)), (0, 0))

        pygame.display.update()

if __name__ == "__main__":
    main(300, 300, 2)