import pygame
import math
import numpy as np
#import pygame.surfarray as surfarray

grid = None


def init(dimx, dimy):
    global grid

    grid = np.zeros((3, dimx, dimy))

    sigma=2
    for c in range(0, dimx):
        for r in range(0, dimy):
            dist = math.sqrt((dimx/2-c)**2 + (dimy/2-r)**2)
            grid[0:2, c, r] = 400 * math.exp(-(dist**2)/(2*sigma**2))

def update(surface, dimx, dimy):
    buf = grid[2]
    grid[2] = grid[1]
    grid[1] = grid[0]
    grid[0] = buf

    h = 1
    k = 1
    c = 0.7
    tau = (c * c * k * k) / (h * h)

    for c in range(1, dimx-1):
        for r in range(1, dimy-1):
            grid[0, c, r]  = grid[1, c-1, r] + grid[1, c+1, r] + grid[1, c, r-1] + grid[1, c, r+1] - 4*grid[1, c, r]
            grid[0, c, r] *= tau
            grid[0, c, r] += 2 * grid[1, c, r] - grid[2, c, r]
            val = int(max(min(100 + grid[0, c, r], 255), 0))
            surface.set_at((c, r), (val, val, val))
 

def render(display, surface, dimx, dimy, cellsize):
    tmp = pygame.transform.scale(surface, (dimx * cellsize, dimy * cellsize))
    display.blit(tmp, (0, 0))
    pygame.display.update()


def main(dimx, dimy, cellsize):
    pygame.init()
    display = pygame.display.set_mode((dimx*cellsize, dimy*cellsize))
    pygame.display.set_caption("Water Effect - Solving the 2d Wave Equation")

    surface_water = pygame.Surface((dimx, dimy))

    init(dimx, dimy)

    ct = 0
    while True:
        ct = ct + 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        update(surface_water, dimx, dimy)
        render(display, surface_water, dimx, dimy, cellsize)
        print(f"{ct}")


if __name__ == "__main__":
    main(100, 100, 5)