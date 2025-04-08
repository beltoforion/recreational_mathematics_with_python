import pygame
import random
import numpy as np

cells = None
fire_colors = []


def init(dimx, dimy):
    global fire_colors, cells

    cells = np.zeros((dimx, dimy))

    for c in range(0, dimx):
        cells[c, 0] = random.random()

    # create color palette
    p1, p2, p3 = (0,0,0), (80, 0, 0), (255, 255, 128)
    for i in range(0, 256):
        if i < 128:
            r, g, b = (p1[0] + (p2[0] - p1[0]) * i/128,
                       p1[1] + (p2[1] - p1[1]) * i/128,
                       p1[2] + (p2[2] - p1[2]) * i/128)
        else:
            r, g, b = (p2[0] + (p3[0] - p2[0]) * (i-128)/128,
                       p2[1] + (p3[1] - p2[1]) * (i-128)/128,
                       p2[2] + (p3[2] - p2[2]) * (i-128)/128)
        fire_colors.append((int(r), int(g), int(b)))


def update(surface, cells, dimx, dimy):
    old = np.copy(cells)
    for r in range(0, dimy-1):
        for c in range(0, dimx-1):
            if r == 0 and 5 < c < dimx-5:
                cells[c, r] = cells[c, 1] + random.random() * 0.9
            else:
                intens = 4.1 + 0.3 * ((abs(c - dimx/2) / (dimx/2)) )
                cells[c, r] = (old[c - 1, r - 1] + old[c, r - 1] +
                               old[c + 1, r - 1] + old[c, r - 2]) / intens

            val = min(int(255 * cells[c, r - 1]), 255) * (r > 2)
            surface.set_at((c, r), fire_colors[val])


def render(display, surface_fire, dimx, dimy, cellsize):
    tmp = pygame.transform.scale(surface_fire, (dimx * cellsize, dimy * cellsize))
    tmp = pygame.transform.flip(tmp, False, True)
    display.blit(tmp, (0, 0))
    pygame.display.update()


def main(dimx, dimy, cellsize):
    pygame.init()
    display = pygame.display.set_mode((dimx*cellsize, dimy*cellsize))
    pygame.display.set_caption("Fire Effect")

    surface_fire = pygame.Surface((dimx, dimy))
    init(dimx, dimy)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        update(surface_fire, cells, dimx, dimy)
        render(display, surface_fire, dimx, dimy, cellsize)


if __name__ == "__main__":
    main(100, 70, 6)