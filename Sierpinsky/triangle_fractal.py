import pygame
import os
from random import randint

out_dir = f'{os.path.dirname(__file__)}/out'


class Point:
    def __init__(self, pos, col):
        self.pos = pos
        self.col = col


def subdivide(surface, v, p, q, r, col, level):
    if level >= 10:
        pixel_col = (min(col[0], 255), min(col[1], 255), min(col[2], 255))
        pygame.draw.polygon(surface, pixel_col, ((int(p.pos[0]), int(p.pos[1])),
                                                 (int(q.pos[0]), int(q.pos[1])),
                                                 (int(r.pos[0]), int(r.pos[1]))))
        return

    col = (randint(col[0], col[0] + 10), randint(col[1], col[1] + 10), randint(col[2], col[2] + 10))
    m = Point(((p.pos[0] + q.pos[0]) / 2, (p.pos[1] + q.pos[1]) / 2), col)
    n = Point(((q.pos[0] + r.pos[0]) / 2, (q.pos[1] + r.pos[1]) / 2), col)
    o = Point(((p.pos[0] + r.pos[0]) / 2, (p.pos[1] + r.pos[1]) / 2), col)

    subdivide(surface, v, m, n, o, p.col if v[0] == 0 else col, level + 1)
    subdivide(surface, v, p, o, m, p.col if v[1] == 0 else col, level + 1)
    subdivide(surface, v, m, n, q, p.col if v[2] == 0 else col, level + 1)
    subdivide(surface, v, o, r, n, p.col if v[3] == 0 else col, level + 1)



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
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        v = [1, 1, 0, 0]
        for i in range(0, 16):
            for j in range(0, 4):
                v[j] = int(bool(i & (1 << j)))

            surface.fill((255, 255, 255))
            subdivide(surface, v, p, q, r, col2, 0)
            print(f'v = ({v[0]}, {v[1]}, {v[2]}, {v[3]}')
            pygame.display.update()
            pygame.image.save(surface, f'{out_dir}/triangle_{v[0]}{v[1]}{v[2]}{v[3]}.jpg')


if __name__ == "__main__":
    main(600, 480)
