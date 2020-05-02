import pygame
import random
import math
import numpy as np

grid = None


def init(dimx, dimy):
    global grid

    grid = np.zeros((3, dimx, dimy))

    for c in range(1, dimx):
        for r in range(1, dimy):
            if math.sqrt(math.pow(dimx/2-c, 2) - math.pow(dimy/2-r, 2)) < 10:
                grid[0:1, c, r] = 1
            else:
                grid[0:1, c, r] = 0


def update(surface, dimx, dimy):
    buf = grid[2]
    grid[2] = grid[1]
    grid[1] = grid[0]
    grid[0] = buf

    h = 0.1
    k = 0.05

    tau = k * k / h * h
    tau5 = k * k / (144 * h * h)

    for c in range(0, dimx):
        for r in range(0, dimy):
            grid[0, c, r] = tau5 * (     grid[1, c-4, r] -  16 * grid[1, c-3, r] + 64 * grid[1, c-2, r] +
                                    16 * grid[1, c-1, r] - 130 * grid[1, c, r]   + 16 * grid[1, c+1, r] +
                                    64 * grid[1, c+2, r] -  16 * grid[1, c+3, r] +      grid[1, c+4, r])


def render(display, surface_fire, dimx, dimy, cellsize):
    display.blit(pygame.transform.scale(surface_fire, (dimx * cellsize, dimy * cellsize)), (0, 0))
    pygame.display.update()


def main(dimx, dimy, cellsize):
    pygame.init()
    display = pygame.display.set_mode((dimx*cellsize, dimy*cellsize))
    pygame.display.set_caption("Water Effect - Solving the 2d Wave Equation")

    surface_water = pygame.Surface((dimx, dimy))
    init(dimx, dimy)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        update(surface_water, dimx, dimy)
        render(display, surface_water, dimx, dimy, cellsize)


if __name__ == "__main__":
    main(100, 100, 4)