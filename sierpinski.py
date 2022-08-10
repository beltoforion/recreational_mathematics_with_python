import pygame
from pygame.locals import *

def subdivide(surface, p1, p2, p3, level):
    if level >= 9:
        return

    pygame.draw.lines(surface, (255, 255, 255), True, (p1, p2, p3))
    pygame.display.update()

    subdivide(surface, p1, ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2), ((p1[0] + p3[0]) / 2, (p1[1] + p3[1]) / 2), level + 1)
    subdivide(surface, p2, ((p2[0] + p3[0]) / 2, (p2[1] + p3[1]) / 2), ((p2[0] + p1[0]) / 2, (p2[1] + p1[1]) / 2), level + 1)
    subdivide(surface, p3, ((p3[0] + p2[0]) / 2, (p3[1] + p2[1]) / 2), ((p3[0] + p1[0]) / 2, (p3[1] + p1[1]) / 2), level + 1)

def main():
    pygame.init()
    surface = pygame.display.set_mode((800, 600))

    subdivide(surface, (400, 100), (100, 500), (700, 500), 0)

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                return

if __name__ == "__main__":
    main()