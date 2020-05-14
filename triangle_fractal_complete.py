import pygame
import itertools
from random import randint
from pygame.locals import *

surface = None


class Point:
    def __init__(self, pos, col):
        self.pos = pos
        self.col = col


def subdivide(v, k, p, q, r, col, level):
    if level>=9:
        pixel_col = (min(col[0], 255), min(col[1], 255), min(col[2], 255))
#        surface.set_at( (int(p.pos[0]), int(p.pos[1])), pixel_col)
        pygame.draw.polygon(surface, pixel_col, ((int(p.pos[0]), int(p.pos[1])), (int(q.pos[0]), int(q.pos[1])), (int(r.pos[0]), int(r.pos[1]))))
        return


#    pixel_col = (min(col[0], 255), min(col[1], 255), min(col[2], 255))
#    pygame.draw.polygon(surface, pixel_col, ((int(p.pos[0]), int(p.pos[1])), (int(q.pos[0]), int(q.pos[1])), (int(r.pos[0]), int(r.pos[1]))))

    col = (randint(col[0], col[0] + 6), randint(col[1], col[1] + 6), randint(col[2], col[2] + 6))
    m = Point(((p.pos[0] + q.pos[0]) / 2, (p.pos[1] + q.pos[1]) / 2), col)
    n = Point(((q.pos[0] + r.pos[0]) / 2, (q.pos[1] + r.pos[1]) / 2), col)
    o = Point(((p.pos[0] + r.pos[0]) / 2, (p.pos[1] + r.pos[1]) / 2), col)

    l = list(itertools.permutations([m, n, o]))
    idx = k[0]
    subdivide(v, k, l[idx][0], l[idx][1], l[idx][2], p.col if v[0] == 0 else col, level + 1)

    l = list(itertools.permutations([p, o, m]))
    idx = k[1]
    subdivide(v, k, l[idx][0], l[idx][1], l[idx][2], p.col if v[0] == 0 else col, level + 1)

    l = list(itertools.permutations([m, n, q]))
    idx = k[2]
    subdivide(v, k, l[idx][0], l[idx][1], l[idx][2], p.col if v[0] == 0 else col, level + 1)

    l = list(itertools.permutations([o, r, n]))
    idx = k[3]
    subdivide(v, k, l[idx][0], l[idx][1], l[idx][2], p.col if v[0] == 0 else col, level + 1)


def main(width, height):
    global surface

    pygame.init()

    surface = pygame.display.set_mode((width, height))
    pygame.display.set_caption('Triangular Fraktal')

    col1 = (200, 180, 60)
    col2 = (100, 30, 30)

    p = Point((20, 20), col1)
    q = Point((width/2, height-20),col1)
    r = Point((width-20, 20), col1)

    v = [0, 0, 0, 1]
    k = [0, 4, 0, 3]
    surface.fill((0, 0, 20))
    subdivide(v, k, p, q, r, col2, 0)
    print('v = ({}, {}, {}, {})'.format(v[0], v[1], v[2], v[3]))
    pygame.display.update()
    pygame.image.save(surface, './triangle_{}{}{}{}_{}{}{}{}.jpg'.format(v[0], v[1], v[2], v[3], k[0], k[1], k[2], k[3]))

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                return

    return


    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                return

        v = [0, 0, 0, 0]
        k = [0, 0, 0, 0]
        for i in range(0, 16):
            for j in range(0, 4):
                v[j] = int(bool(i & (1 << j)))

            for j1 in range(0,6):
                for j2 in range(0, 6):
                    for j3 in range(0, 6):
                        for j4 in range(0, 6):
                            k = [j1, j2, j3, j4]

                            surface.fill((0, 0, 20))
                            subdivide(v, k, p, q, r, col2, 0)
                            print('v = ({}, {}, {}, {})'.format(v[0], v[1], v[2], v[3]))
                            pygame.display.update()
                            pygame.image.save(surface, './out/triangle_{}{}{}{}_{}{}{}{}.jpg'.format(v[0], v[1], v[2], v[3], k[0], k[1], k[2], k[3]))


if __name__ == "__main__":
    main(1024, 768)
