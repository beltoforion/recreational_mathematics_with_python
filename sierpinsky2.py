import pygame
from pygame.locals import *


def subdivide(surface, level, p1, p2, p3):
    b = 255
    pygame.draw.line(surface, (b, b, b), p1, p2)
    pygame.draw.line(surface, (b, b, b), p2, p3)
    pygame.draw.line(surface, (b, b, b), p3, p1)
    pygame.display.update()

    if level < 9:
        subdivide(surface, level + 1, p1, ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2), ((p1[0] + p3[0]) / 2, (p1[1] + p3[1]) / 2))
        subdivide(surface, level + 1, p2, ((p2[0] + p3[0]) / 2, (p2[1] + p3[1]) / 2), ((p2[0] + p1[0]) / 2, (p2[1] + p1[1]) / 2))
        subdivide(surface, level + 1, p3, ((p3[0] + p2[0]) / 2, (p3[1] + p2[1]) / 2), ((p3[0] + p1[0]) / 2, (p3[1] + p1[1]) / 2))


def main():
    pygame.init()
    surface = pygame.display.set_mode((800, 600))
    pygame.display.set_caption('Sierpinsky Dreieck (II)')

    subdivide(surface, 0, (400, 100), (100, 500), (700, 500))

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                return

        pygame.display.update()


if __name__ == "__main__":
    main()