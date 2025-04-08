import pygame
import numpy as np
from numba import jit


@jit
def trap_function(z, petals=5):
    return abs(np.sin(petals * np.angle(z)) * np.abs(z))


@jit
def compute_mandelbrot(dimx : int, dimy : int, maxiter : int, rmin : float, rmax : float, imin : float, imax : float):
    istep = (imax - imin) / dimy
    rstep = (rmax - rmin) / dimx
    mandelbrot_set = np.zeros((dimy, dimx), dtype=np.float64)

    escape_radius = 100
    petals = 9
    for row in range(dimy):
        for col in range(dimx):
            z = 0
            c = (rmin + col * rstep) + 1j * (imin + row * istep)

            dist = 1e10
            for k in range(maxiter):
                z = z*z + c

                dist = min(dist, trap_function(z, petals))

                if abs(z) > escape_radius:
                    mandelbrot_set[row, col] = k + 1 - np.log(np.log(np.abs(z))) / np.log(2)
                    mandelbrot_set[row, col] += 0.5*dist
                    break

            else:
                mandelbrot_set[row, col] = dist * 40

    return mandelbrot_set


def main(width, height, max_iter):
    pygame.init()
    pygame.display.set_caption("Mandelbrot-Set")
    screen = pygame.display.set_mode((width, height))

    imin, imax = -1.3, 1.3
    rmin, rmax = -(0.5*(imax-imin)) * (width/height)-.5, (0.5*(imax-imin)) * (width/height)-.5

    mandelbrot_array = compute_mandelbrot(width, height, max_iter, rmin, rmax, imin, imax)
    mandelbrot_array = np.log(mandelbrot_array + 1)
    mandelbrot_array /= np.max(mandelbrot_array)

    mask = mandelbrot_array != 0
    start, freq = 1, 9
    r = mask * (0.5 + 0.5*np.cos(start + mandelbrot_array * freq + 0))
    g = mask * (0.5 + 0.5*np.cos(start + mandelbrot_array * freq + 0.6))
    b = mask * (0.5 + 0.5*np.cos(start + mandelbrot_array * freq + 1.0))

    r = (r / np.max(r) * 255).astype(np.uint8)
    g = (g / np.max(g) * 255).astype(np.uint8)
    b = (b / np.max(b) * 255).astype(np.uint8)

    surface = pygame.surfarray.make_surface(np.stack([r.T, g.T, b.T], axis=-1))
    screen.blit(surface, (0, 0)) 
    pygame.display.update()
    pygame.image.save(screen, f"mandelbrot.png")

    while not any(event.type == pygame.QUIT for event in pygame.event.get()):
        pass

    pygame.quit()


if __name__=="__main__":
    main(2*1920, 2*1080, 500)