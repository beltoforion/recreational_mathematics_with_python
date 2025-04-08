import pygame
import numpy as np
import uuid
import colorsys
from numba import jit
from typing import Tuple
from enum import Enum


class OrbitTrap(Enum):
    NoTrap = 0
    Lines = 1
    Circle = 2
    LinesAndCircle = 3


class OutsideColorScheme(Enum):
    IterationCount = 0
    SmoothIterationCount = 1
    PointDistance = 2
    Black = 3


class InsideColorScheme(Enum):
    PointDistance = 0
    Black = 1


def complex_to_screen(z, rmin, rmax, imin, imax, width, height):
    re = z.real
    im = z.imag
    x = int((re - rmin) / (rmax - rmin) * width)
    y = int((im - imin) / (imax - imin) * height)
    return x, y


@jit
def compute_mandelbrot(
    dimx : int, 
    dimy : int, 
    maxiter : int, 
    rmin : float, 
    rmax : float, 
    imin : float, 
    imax : float, 
    inside_color_scheme : InsideColorScheme,
    outside_color_scheme : OutsideColorScheme,
    orbit_trap : OrbitTrap):

    istep = (imax - imin) / dimy
    rstep = (rmax - rmin) / dimx

    mandelbrot_set = np.zeros((dimy, dimx), dtype=np.float64)

    for row in range(dimy):
        for col in range(dimx):
            c = (rmin + col * rstep) + 1j * (imin + row * istep)
            z = 0

            escape_radius = 100
            dist = 1e10

            # Points to check for orbit trap
            points = [complex(0, 0), complex(-1, 0), complex(0, 1), complex(0, -1)]

            trapped = False
            trap_size = 0.01
            trap_dist = 0
            trap_radius = 0.5
            for k in range(maxiter):
                z = z*z + c
                #z = z*z + 0.1 * z*z*z + 0.05 * z*z*z*z + 1.1*c
                #z = z*z + 2*z +c
                #z = z*z + 0.19 * z*z*z + c

                if abs(z) > escape_radius:
                    break

                # Compute distance to the closest point
                if outside_color_scheme == OutsideColorScheme.PointDistance or inside_color_scheme == InsideColorScheme.PointDistance:
                    for point in points:
                        dist = min(dist, abs(z-point))

                if orbit_trap == OrbitTrap.Circle or orbit_trap == OrbitTrap.LinesAndCircle:
                    if (trap_radius-trap_size) < abs(z) < (trap_radius + trap_size):
                        trap_dist = trap_radius - abs(z)
                        trapped = True

                if orbit_trap == OrbitTrap.Lines or orbit_trap == OrbitTrap.LinesAndCircle:
                    if abs(z.real) < trap_size:
                        trap_dist = abs(z.real)
                        trapped = True
                    elif abs(z.imag) < trap_size:
                        trap_dist = abs(z.imag)
                        trapped = True

                if trapped:
                    dist = 1 - trap_dist/trap_size
                    break

            if abs(z) > escape_radius:
                if outside_color_scheme == OutsideColorScheme.IterationCount:
                    value = np.log(k)
                elif outside_color_scheme == OutsideColorScheme.SmoothIterationCount:
                    value = k + 1 - np.log(np.log(np.abs(z))) / np.log(2)
                elif outside_color_scheme == OutsideColorScheme.PointDistance:
                    value = dist
                elif outside_color_scheme == OutsideColorScheme.Black:
                    value = 0
                else:
                    raise ValueError("Invalid outside color scheme")
            else:
                if inside_color_scheme == InsideColorScheme.PointDistance:
                    value = dist
                else:
                    if trapped:
                        value = dist
                    else:
                        value = 0
#                value = dist if color_members else 0

            mandelbrot_set[row, col] = value
                
    return mandelbrot_set


def compute_orbit(c, max_iter):
    orbit = []
    z = 0
    for _ in range(max_iter):
        z = z * z + c
        orbit.append(z)
        if abs(z) > 2:
            break
    return orbit


def draw_axes(screen, font, width, height, rmin, rmax, imin, imax):
    x_axis, y_axis = complex_to_screen(complex(0,0), rmin, rmax, imin, imax, width, height)

    pygame.draw.line(screen, (255, 0, 0), (0, y_axis), (width, y_axis), 1)
    for r in np.arange(rmin, rmax + 0.01, 0.5):
        x = int((r - rmin) / (rmax - rmin) * width)
        draw_label_with_outline(screen, f"{r:.1f}", font, (x + 2, y_axis + 2))

    pygame.draw.line(screen, (255, 0, 0), (x_axis, 0), (x_axis, height), 1)
    for im in np.arange(imin, imax + 0.01, 0.5):
        if abs(im) < 1e-6:
            continue

        y = int((im - imin) / (imax - imin) * height)
        draw_label_with_outline(screen, f"{-im:.1f}i", font, (x_axis + 2, y + 2))


def draw_label_with_outline(screen, text, font, pos, text_color=(255, 255, 255), outline_color=(0, 0, 0)):
    x, y = pos
    label_surface = font.render(text, True, text_color)
    outline_surface = font.render(text, True, outline_color)

    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx != 0 or dy != 0:
                screen.blit(outline_surface, (x + dx, y + dy))

    screen.blit(label_surface, (x, y))


def draw_state(screen, font, width : int, height : int, rmin : float, rmax : float, imin : float, imax : float, max_iter : int):
    x, y = pygame.mouse.get_pos()
    c = screen_to_complex(x, y, rmin, rmax, imin, imax, width, height)
    
    text = f"pos: {c.real:1.7f}, {c.imag:1.7f}; max_iter= {max_iter}"
    label_surface = font.render(text, True, (255, 255, 255))

    pos_x = 5
    pos_y = screen.get_height() - label_surface.get_height() - 5

    draw_label_with_outline(screen, text, font, (pos_x + 2, pos_y + 2))


def draw_orbit(screen, orbit, color, rmin, rmax, imin, imax, width, height):
    points = [
        complex_to_screen(z, rmin, rmax, imin, imax, width, height)
        for z in orbit
    ]
    points = [p for p in points if 0 <= p[0] < width and 0 <= p[1] < height]
    if len(points) >= 2:
        pygame.draw.lines(screen, color, False, points, 1)
    for x, y in points:
        pygame.draw.circle(screen, color, (x, y), 2)


def render_mandelbrot(
        width : int, 
        height : int, 
        max_iter : int, 
        rmin : float, 
        rmax : float, 
        imin : float, 
        imax : float, 
        inside_color_scheme : InsideColorScheme, 
        outside_color_scheme : OutsideColorScheme, 
        orbit_trap : OrbitTrap,
        color_scheme):
    mandelbrot_array = compute_mandelbrot(width, height, max_iter, rmin, rmax, imin, imax, inside_color_scheme, outside_color_scheme, orbit_trap)
    mandelbrot_array = np.log(mandelbrot_array + 1)
    mandelbrot_array = mandelbrot_array / np.max(mandelbrot_array)

    if color_scheme==1:
        start = 2.5
        freq = 20
        r = (0.5 + 0.5*np.cos(start + mandelbrot_array * freq + 0))
        g = (0.5 + 0.5*np.cos(start + mandelbrot_array * freq + 0.6))
        b = (0.5 + 0.5*np.cos(start + mandelbrot_array * freq + 1.0))
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
    elif color_scheme==3:
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
                    r[i, j] = 1
                    g[i, j] = 1
                    b[i, j] = 1
    else:
        r = mandelbrot_array
        g = mandelbrot_array
        b = mandelbrot_array

    r = (r / np.max(r) * 255).astype(np.uint8)
    g = (g / np.max(g) * 255).astype(np.uint8)
    b = (b / np.max(b) * 255).astype(np.uint8)

    surface = pygame.surfarray.make_surface(np.stack([r.T,g.T,b.T], axis=-1))
    return surface
 

def screen_to_complex(x, y, rmin, rmax, imin, imax, width, height):
    re = rmin + x / width * (rmax - rmin)
    im = imin + y / height * (imax - imin)
    return complex(re, im)


def set_position(center : complex, aoi : complex, width : int, height : int) -> Tuple[float, float, float, float]:
    rmin = center.real - aoi.real / 2
    rmax = center.real + aoi.real / 2
    imin = center.imag - aoi.imag / 2
    imax = center.imag + aoi.imag / 2

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

    return rmin, rmax, imin, imax


def main(width, height, max_iter):
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Mandelbrot mit Orbit-Vorschau")

    saved_orbits = []
    preview_orbit = []

    timer_set = False
    running = True
    save_img_count = 0
    click_count = 0

    font = pygame.font.SysFont("Arial", 20)

    #
    # Render Settings
    #

    show_axis = True
    show_orbits = False
    show_state = False
    color_scheme = 1
    color_members = True
    use_orbit_trap = False
    inside_color_scheme = InsideColorScheme.PointDistance
    outside_color_scheme = OutsideColorScheme.SmoothIterationCount
    orbit_trap = OrbitTrap.NoTrap
    aoi = complex(3, 3)
    pos = complex(-0.5, 0)
    recompute = True

    EVENT_DOUBLE_CLICK : int  = pygame.USEREVENT + 1

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_o:
                    show_orbits = not show_orbits

                    if not show_orbits:
                        saved_orbits.clear()
                        preview_orbit.clear()
                elif event.key == pygame.K_a:
                    show_axis = not show_axis

                elif event.key == pygame.K_s:
                    if pygame.key.get_mods() & pygame.KMOD_CTRL:
                        save_img_count += 1
                        # use guid to create a unique filename
                        filename = f"mandelbrot_r={pos.real:1.8g},i={pos.imag:1.8g}_{str(uuid.uuid4())[:8]}.png"
                        pygame.image.save(screen, filename) #f"mandelbrot_{save_img_count}.png")
                    else:
                        show_state = not show_state

                elif event.key == pygame.K_F1:
                    inside_color_scheme = InsideColorScheme.Black
                    recompute = True

                elif event.key == pygame.K_F2:
                    inside_color_scheme = InsideColorScheme.PointDistance
                    recompute = True

                elif event.key == pygame.K_F3:
                    outside_color_scheme = OutsideColorScheme.SmoothIterationCount
                    recompute = True

                elif event.key == pygame.K_F4:
                    outside_color_scheme = OutsideColorScheme.IterationCount
                    recompute = True

                elif event.key == pygame.K_F5:
                    outside_color_scheme = OutsideColorScheme.PointDistance
                    recompute = True

                elif event.key == pygame.K_F6:
                    outside_color_scheme = OutsideColorScheme.Black
                    recompute = True

                elif event.key == pygame.K_F7:
                    orbit_trap = OrbitTrap.NoTrap
                    recompute = True

                elif event.key == pygame.K_F8:
                    orbit_trap = OrbitTrap.Lines
                    recompute = True

                elif event.key == pygame.K_F9:
                    orbit_trap = OrbitTrap.Circle
                    recompute = True

                elif event.key == pygame.K_F10:
                    orbit_trap = OrbitTrap.LinesAndCircle
                    recompute = True

                elif event.key == pygame.K_1:   
                    color_scheme = 0
                    recompute = True
                
                elif event.key == pygame.K_2:
                    color_scheme = 1
                    recompute = True
                
                elif event.key == pygame.K_3:
                    color_scheme = 2
                    recompute = True

                elif event.key == pygame.K_4:
                    color_scheme = 3
                    recompute = True

                elif event.key == pygame.K_i:
                    max_iter = int(max_iter * (0.5 if pygame.key.get_mods() & pygame.KMOD_SHIFT else  1.5))
                    recompute = True

                elif event.key == pygame.K_t:
                    use_orbit_trap = not use_orbit_trap
                    recompute = True

                elif event.key == pygame.K_m:
                    color_members = not color_members
                    recompute = True

                elif event.key == pygame.K_r:
                    aoi = complex(3, 3)
                    pos = complex(-0.5, 0)
                    max_iter = 200
                    recompute = True
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if not timer_set:
                    pygame.time.set_timer(EVENT_DOUBLE_CLICK, 200)
                    click_count += 1

            elif event.type == EVENT_DOUBLE_CLICK:
                if click_count == 1:
                    if preview_orbit:
                        saved_orbits.append(preview_orbit.copy())
                elif click_count == 2:
                    mx, my = pygame.mouse.get_pos()
                    pos = screen_to_complex(mx, my, rmin, rmax, imin, imax, width, height)
                    aoi *= .5
                    recompute = True
                        
                pygame.time.set_timer(EVENT_DOUBLE_CLICK, 0)
                timer_set = False
                click_count = 0

        #
        # Do the rendering
        #

        if recompute:
            rmin,rmax,imin,imax = set_position(pos, aoi, width, height)
            surface = render_mandelbrot(width, height, max_iter, rmin, rmax, imin, imax, inside_color_scheme, outside_color_scheme, orbit_trap, color_scheme)
            recompute = False

        screen.blit(surface, (0, 0))

        if show_axis:
            draw_axes(screen, font, width, height, rmin, rmax, imin, imax)

        if show_orbits:
            mx, my = pygame.mouse.get_pos()
            c = screen_to_complex(mx, my, rmin, rmax, imin, imax, width, height)
            preview_orbit = compute_orbit(c, 200)

            for orbit in saved_orbits:
                draw_orbit(screen, orbit, (0, 200, 0), rmin, rmax, imin, imax, width, height)

            if preview_orbit:
                draw_orbit(screen, preview_orbit, (255, 255, 0), rmin, rmax, imin, imax, width, height)
        
        if show_state:
            draw_state(screen, font, width, height, rmin, rmax, imin, imax, max_iter)

        pygame.display.update()

    pygame.quit()


if __name__ == "__main__":
    main(800, 800, 500)