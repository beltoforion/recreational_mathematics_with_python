import pygame
import numpy as np
import random
from numba import jit, njit

@njit
def compute_mandelbrot_point(c : complex, power : float, istep : float, rstep : float, imin : float, rmin : float, maxiter : int, mandelbrot_prob : np.ndarray):
    z = 0
    
    # array for storing traces
    trace = np.zeros((maxiter, 2), dtype=np.float32)

    for k in range(maxiter):
        z = z*z + c

        col = (z.real - rmin) / rstep
        row = (z.imag - imin) / istep

        trace[k] = [col, row]               
        
        if abs(z) > 3.2:
            for i in range(k):
                col = trace[i, 0]
                row = trace[i, 1]
                if (row >= 0 and row < mandelbrot_prob.shape[0]) and (col >= 0 and col < mandelbrot_prob.shape[1]):
                    mandelbrot_prob[int(row), int(col)] += 1
            return k

    return maxiter    

@jit
def compute_mandelbrot(mandelbrot_prob, power : float, width : int, height : int, maxiter : int, rmin : float, rmax : float, imin : float, imax : float):
    istep = (imax - imin) / height
    rstep = (rmax - rmin) / width

    for n in range(100000):
        rnd_col = random.uniform(rmin, rmax)
        rnd_row = random.uniform(imin, imax)

        c = rnd_col + 1j * rnd_row
        compute_mandelbrot_point(c, power, istep, rstep, imin, rmin, maxiter, mandelbrot_prob)

def create_image_map(prob_data, gamma : float = 5.2):
    prob_map = prob_data.copy()
    prob_map = np.log(prob_map + 1)
    prob_map = prob_map - np.quantile(prob_map, 0.1)
    prob_map[prob_map < 0] = 0

    norm_map = prob_map / np.max(prob_map)
    norm_map = (norm_map * 255).astype(np.uint8)

    return norm_map

def main(width : int = 1000, height : int =1000) -> None:
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Mandelbrot-Set")

    rmin = -2;   rmax = 1
    rmin = -1.8;   rmax = 1.8
    imin = -1.5; imax = 1.5

#    rmin = -1.74;   rmax = -1.78
#    imin = -.02; imax = .02

#    rmin = -1.753;   rmax = -1.767
#    imin = -.007; imax = .007

#    f = 1
#    rmin=-0.750222*f; rmax=-0.749191*f
#    imax=0.031161*f; imin=0.031752*f

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

    mandelbrot_prob_red = np.zeros((height, width), dtype=np.int32)
    mandelbrot_prob_green = np.zeros((height, width), dtype=np.int32)
    mandelbrot_prob_blue = np.zeros((height, width), dtype=np.int32)

    running = True
    ct = 0
   
    while running:
        # 100, 400, 1200
        compute_mandelbrot(mandelbrot_prob_red, 2, width, height,100, rmin, rmax, imin, imax)
        compute_mandelbrot(mandelbrot_prob_green, 2, width, height, 200, rmin, rmax, imin, imax)
        compute_mandelbrot(mandelbrot_prob_blue, 2, width, height, 500, rmin, rmax, imin, imax)

        prob_map_red = create_image_map(mandelbrot_prob_red)
        prob_map_green = create_image_map(mandelbrot_prob_green)
        prob_map_blue = create_image_map(mandelbrot_prob_blue)

        prob_surface = pygame.surfarray.make_surface(np.stack([prob_map_blue, prob_map_green, prob_map_red], axis=-1).transpose(1, 0, 2))
        screen.blit(prob_surface, (0, 0))
        pygame.display.update()

        if ct%100 == 0:
            pygame.image.save(screen, f"mandelbrot_prob_{ct//100:03d}.png")

        ct += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
    
    pygame.quit()

if __name__=="__main__":
    main(1920, 1080)
#    main(700, 700)