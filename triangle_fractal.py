import pygame
import time
from random import randint
from pygame.locals import *

surface = None


class Point:
    def __init__(self, pos, col):
        self.pos = pos
        self.col = col


def subdivide(v, p, q, r, col):
    if (abs(q.pos[0] - p.pos[0]) < 1 and abs(q.pos[1] - p.pos[1]) < 1):
        pixel_col = (min(col[0], 255), min(col[1], 255), min(col[2], 255))
        surface.set_at( (int(p.pos[0]), int(p.pos[1])), pixel_col)
        return

    col = (randint(col[0], col[0] + 6), randint(col[1], col[1] + 6), randint(col[2], col[2] + 6))
    m = Point(((p.pos[0] + q.pos[0]) / 2, (p.pos[1] + q.pos[1]) / 2), col)
    n = Point(((q.pos[0] + r.pos[0]) / 2, (q.pos[1] + r.pos[1]) / 2), col)
    o = Point(((p.pos[0] + r.pos[0]) / 2, (p.pos[1] + r.pos[1]) / 2), col)

    color_switcher = {
        0: p.col,
        1: col
    }

    subdivide(v, m, n, o, color_switcher.get(v[0]))
    subdivide(v, p, o, m, color_switcher.get(v[1]))
    subdivide(v, m, n, q, color_switcher.get(v[2]))
    subdivide(v, o, r, n, color_switcher.get(v[3]))


def main(width, height):
    global surface

    pygame.init()

    surface = pygame.display.set_mode((width, height))
    pygame.display.set_caption('Triangular Fraktal')

    col1 = (200, 180, 60)
    col2 = (100, 30, 30)

    p = Point((20, 20), col1)
    q = Point((width/2, height-20), col1)
    r = Point((width-20, 20), col1)

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                return

        v = [0, 0, 0, 0]
        for i in range(0, 16):
            for j in range(0, 4):
                v[j] = int(bool(i & (1 << j)))

            surface.fill((0, 0, 20))
            subdivide(v, p, q, r, col2)
            print('v = ({}, {}, {}, {})'.format(v[0], v[1], v[2], v[3]))
            pygame.display.update()
            #pygame.image.save(surface, 'triangle_{}{}{}{}.jpg'.format(v[0], v[1], v[2], v[3]))


if __name__ == "__main__":
    main(1024, 768)
