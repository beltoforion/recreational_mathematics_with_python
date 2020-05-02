import pygame
import random
from pygame.locals import *

def main():
    pygame.init()
    surface = pygame.display.set_mode((800, 600))
    pygame.display.set_caption('Sierpinsky Dreieck (I)')

    p = [(400, 50), (50, 550), (750, 550)]
    x = 400
    y = 300

    while True:
        i = random.randint(0, 2)
        xt = p[i][0]
        yt = p[i][1]

        x += int((xt - x) / 2)
        y += int((yt - y) / 2)

        surface.set_at((x, y), (255, 255, 255))

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                return

        pygame.display.update()

if __name__ == "__main__":
    main()