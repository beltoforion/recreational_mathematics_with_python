import pygame
import colorsys
import numpy as np
from numba import jit

@jit
def compute_mandelbrot(dimx : int, dimy : int, maxiter : int, rmin : float, rmax : float, imin : float, imax : float, color_members : bool):
    istep = (imax - imin) / dimy
    rstep = (rmax - rmin) / dimx

    mandelbrot_set = np.zeros((dimy, dimx), dtype=np.float64)

    for row in range(dimy):
        for col in range(dimx):
            c = (rmin + col * rstep) + 1j * (imin + row * istep)
            z = 0

            escape_radius = 1000
            dist = 1e10
            points = [complex(0, 0), complex(-1, 0), complex(1, 1), complex(1, -1)]
            #points = [complex(0, 0)]
            for k in range(maxiter):
                z = z*z + c
                #z = z*z*z + c
                #z = z*z*z + (c-1)*z-c

                if abs(z) > escape_radius:
                    break

                # Compute distance to the closest point
                for point in points:
                    dist = min(dist, abs(z-point))

            if abs(z) > escape_radius:
                value = k + 1 - np.log(np.log(np.abs(z))) / np.log(2)
                value = dist
            else:
                value = dist if color_members else 0
                
            mandelbrot_set[row, col] = value

    return mandelbrot_set


def main(width, height, max_iter):
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Mandelbrot-Set")

    rmin = -2;    rmax = 1
    imin = -1.5;  imax = 1.5

    width_mandelbrot = rmax - rmin
    height_mandelbrot = imax - imin
    aspect_mandelbrot = width_mandelbrot / height_mandelbrot
    aspect_window = width / height

    if aspect_window > aspect_mandelbrot:
        new_height_mandelbrot = height_mandelbrot
        new_width_mandelbrot = height_mandelbrot * aspect_window
        rmin = rmin - (new_width_mandelbrot - width_mandelbrot) / 2
        rmax = rmax + (new_width_mandelbrot - width_mandelbrot) / 2
    elif aspect_window < aspect_mandelbrot:
        new_width_mandelbrot = width_mandelbrot
        new_height_mandelbrot = width_mandelbrot / aspect_window
        imin = imin - (new_height_mandelbrot - height_mandelbrot) / 2
        imax = imax + (new_height_mandelbrot - height_mandelbrot) / 2

    color_scheme = 1
    color_members = True

    # compute mandelbrot, take log of value, scale to 0-1
    mandelbrot_array = compute_mandelbrot( width, height, max_iter, rmin, rmax, imin, imax, color_members)
    mandelbrot_array = np.log(mandelbrot_array + 1)
    mandelbrot_array = mandelbrot_array / np.max(mandelbrot_array)

    if color_scheme==1:
        start = 2.5
        freq = 20
        r = (0.5 + 0.5*np.cos(start + mandelbrot_array * freq + 0))
        g = (0.5 + 0.5*np.cos(start + mandelbrot_array * freq + 0.6))
        b = (0.5 + 0.5*np.cos(start + mandelbrot_array * freq + 1.0))
        print(f"r_val={np.max(r)}, g_val={np.max(g)}, b_val={np.max(b)}")
    elif color_scheme==2:
        # use color circle to convert values to hsv colors
        r = np.zeros_like(mandelbrot_array)
        g = np.zeros_like(mandelbrot_array)
        b = np.zeros_like(mandelbrot_array)
        for i in range(mandelbrot_array.shape[0]):
            for j in range(mandelbrot_array.shape[1]):
                if mandelbrot_array[i, j] == 0:
                    r[i, j] = 0
                    g[i, j] = 0
                    b[i, j] = 0
                else:
                    h = 10 * mandelbrot_array[i, j] * 360
                    s = 1
                    v = 1
                    r[i, j], g[i, j], b[i, j] = colorsys.hsv_to_rgb(h/360, s, v)
    else:
        r = mandelbrot_array
        g = mandelbrot_array
        b = mandelbrot_array
    
    r = (r / np.max(r) * 255).astype(np.uint8)
    g = (g / np.max(g) * 255).astype(np.uint8)
    b = (b / np.max(b) * 255).astype(np.uint8)
    print(f"r_val={np.max(r)}, g_val={np.max(g)}, b_val={np.max(b)}")

    screen.blit(pygame.surfarray.make_surface(np.stack([r.T,g.T,b.T], axis=-1)), (0, 0)) 
    pygame.display.update()

    running = True
    save_img_count = 0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # Fenster schließen
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    save_img_count += 1
                    pygame.image.save(screen, f"mandelbrot_{save_img_count}.png")
    
    pygame.quit()


if __name__=="__main__":
    main(1000, 1000, 1000)