import pygame
import random
import math
import colorsys
from pygame.locals import *

idx = [0, 0, 0]


def mark_pixel(surface, pos, plane):
    col = surface.get_at(pos)
    v = min(col[plane] + 4, 255)
    surface.set_at(pos, (v if plane == 0 else col[0],
                         v if plane == 1 else col[1],
                         v if plane == 2 else col[2]))


def random_point_index(p):
    global idx
    idx[2] = idx[1]
    idx[1] = idx[0]

    while True:
        idx[0] = random.randint(0, len(p) - 1)
        dst = abs(idx[0] - idx[1])
        if dst == 0:
            continue
        else:
            break

    return idx[0]


def init_polygon(width, height, n):
    delta_angle = 360/n
    r = width/2 - 10
    p = []

    for i in range(0, n):
        angle = (180 + i*delta_angle) * math.pi / 180
        color = colorsys.hsv_to_rgb((i*delta_angle)/360, 0.8, 1)
        p.append(((width/2 + r*math.sin(angle),
                   height/2 + r*math.cos(angle)),
                  (int(color[0]*255), int(color[1]*255), int(color[2]*255))))
    return p


def main(width, height, n, r):
    pygame.init()
    surface = pygame.display.set_mode((width, height))
    pygame.display.set_caption('Das Chaos Spiel')

    p = init_polygon(width, height, n)
    x = [random.randint(0, width), random.randint(0, width), random.randint(0, width)]
    y = [random.randint(0, height), random.randint(0, height), random.randint(0, height)]
    plane_randomness = [0.02, 0.04, 0.08]

    step = 0
    while True:
        step = step + 1
        i = random_point_index(p)
        for plane_idx in (range(0,3)):
            rr = random.random() * plane_randomness[plane_idx]
            x[plane_idx] += (p[i][0][0] - x[plane_idx]) * (r + rr)
            y[plane_idx] += (p[i][0][1] - y[plane_idx]) * (r + rr)

            mark_pixel(surface, (int(x[plane_idx]), int(y[plane_idx])), plane_idx)

        if step % 5000 == 0:
            pygame.display.update()

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.image.save(surface, 'chaosspiel.jpg')
                pygame.quit()
                return


if __name__ == "__main__":
    n = 5; main(500, 500, n, 0.5)