import pygame
import numpy as np
from numba import jit


@jit
def compute_mandelbrot(dimx : int, dimy : int, maxiter : int, rmin : float, rmax : float, imin : float, imax : float):
    istep = (imax - imin) / dimy
    rstep = (rmax - rmin) / dimx
    mandelbrot_set = np.zeros((dimy, dimx), dtype=np.float64)

    for row in range(dimy):
        for col in range(dimx):
            z = 0
            c = (rmin + col * rstep) + 1j * (imin + row * istep)

            trapped = False
            trap_size = 0.025
            trap_dist = 0
            trap_radius = 0.27
            for k in range(maxiter):
                z = z*z + c

                # circular orbit trap
                if (trap_radius-trap_size) < abs(z) < (trap_radius + trap_size):
                    trap_dist = abs(trap_radius - abs(z))
                    trapped = True

                # real axis trap
                if abs(z.real) < trap_size:
                    trap_dist = abs(z.real)
                    trapped = True
                
                # imaginary axis trap
                if abs(z.imag) < trap_size:
                    trap_dist = abs(z.imag)
                    trapped = True

                if abs(z) > 2 or trapped:
                    break

            mandelbrot_set[row, col] = 0 if not trapped else trap_dist

    return mandelbrot_set


def main(width, height, max_iter):
    pygame.init()
    pygame.display.set_caption("Mandelbrot-Set")
    screen = pygame.display.set_mode((width, height))

    imin, imax = -1.3, 1.3
    rmin, rmax = -(0.5*(imax-imin)) * (width/height)-.5, (0.5*(imax-imin)) * (width/height)-.5

    mandelbrot_array = compute_mandelbrot(width, height, max_iter, rmin, rmax, imin, imax)
    mandelbrot_array /= np.max(mandelbrot_array)

    mask = mandelbrot_array != 0
    start, freq = 1.2, 0.7
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
    main(1000, 1000, 100)