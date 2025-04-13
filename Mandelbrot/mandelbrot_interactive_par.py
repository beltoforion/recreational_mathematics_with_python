import pygame
import numpy as np
import uuid
import colorsys
import os
from numba import njit, prange
from typing import Tuple
from enum import Enum


class GeometricOrbitTrap(Enum):
    NoTrap = 0
    Lines = 1
    Circle = 2
    LinesAndCircle = 3


class OutsideColorScheme(Enum):
    IterationCount = 0
    IterationCountPlusTrapFunction = 1
    SmoothIterationCount = 2
    SmoothIterationCountPlusTrapFunction = 3
    TrapFunction = 4
    PointDistance = 5
    Black = 6


class InsideColorScheme(Enum):
    Black = 0
    PointDistance = 1
    TrapFunction = 2


log = []

EVENT_DOUBLE_CLICK : int  = pygame.USEREVENT + 1
EVENT_REMOVE_LOG : int = pygame.USEREVENT + 2

script_path = os.path.dirname(__file__)

show_axis = True
show_orbits = False
show_state = False
show_log = True
show_help = True


@njit(fastmath=True)
def trap_function(z, petals=5):
    return abs(np.sin(petals * np.angle(z)) * np.abs(z))


@njit(fastmath=True)
def trap_function2(z, petals=5):
    return np.log(abs(z) * abs(np.log(abs(z))))


@njit(fastmath=True)
def compute_mandelbrot(
    dimx : int, 
    dimy : int, 
    maxiter : int, 
    rmin : float, 
    rmax : float, 
    imin : float, 
    imax : float, 
    inside_color_scheme : int,
    outside_color_scheme : int,
    orbit_trap : int) -> np.ndarray:

    istep = (imax - imin) / dimy
    rstep = (rmax - rmin) / dimx

    mandelbrot_set = np.zeros((dimy, dimx), dtype=np.float64)

    escape_radius = 50
    petals = 9

    for row in range(dimy):
        for col in range(dimx):
            c = (rmin + col * rstep) + 1j * (imin + row * istep)
            z = 0

            dist = 1e10
            trap_fun_dist = 1e10 #trap_function(c, petals)     

            # Points to check for orbit trap
            points = [complex(0, 0), complex(-1, 0), complex(0, 1), complex(0, -1)]

            trapped = False
            trap_size = 0.01
            trap_dist = 0
            trap_radius = 0.5
            for k in range(maxiter):
                # Mandelbrot Set
                z = z*z + c

                # Sine Mandelbrot Set
                #z = np.sin(z) + c

                # Exponential Mandelbrot
                #z = np.exp(z) + c

                #z = z*z + 0.1 * z*z*z + 0.05 * z*z*z*z + 1.1*c
                #z = z*z + 2*z +c
                #z = z*z + 0.19 * z*z*z + c
                
                # Burning Ship Fractal:
                #z = (abs(z.real) + 1j * abs(z.imag))**2 + c

                if abs(z) > escape_radius:
                    break

                trap_fun_dist = min(trap_fun_dist, trap_function(z, petals))

                # Compute distance to the closest point
                if outside_color_scheme == 4 or inside_color_scheme == 1:
                    for point in points:
                        dist = min(dist, abs(z-point))

                if orbit_trap == 2 or orbit_trap == 3:
                    if (trap_radius-trap_size) < abs(z) < (trap_radius + trap_size):
                        trap_dist = trap_radius - abs(z)
                        trapped = True

                if orbit_trap == 1 or orbit_trap == 3:
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
                if outside_color_scheme == 0:   # OutsideColorScheme.IterationCount
                    value = k

                elif outside_color_scheme == 1: # OutsideColorScheme.IterationCountPlusTrapFunction
                    value = k
                    value += .5*trap_fun_dist

                elif outside_color_scheme == 2: # OutsideColorScheme.SmoothIterationCount
                    value  = k + 1 - np.log(np.log(np.abs(z))) / np.log(2)

                elif outside_color_scheme == 3: # OutsideColorScheme.SmoothIterationCountPlusTrapFunction
                    value  = k + 1 - np.log(np.log(np.abs(z))) / np.log(2)
                    value += .5*trap_fun_dist

                elif outside_color_scheme == 4: # OutsideColorScheme.TrapFunction
                    value = 60*trap_fun_dist

                elif outside_color_scheme == 5: # OutsideColorScheme.PointDistance
                    value = dist

                elif outside_color_scheme == 6: # OutsideColorScheme.Black
                    value = 0

                else:
                    raise ValueError("Invalid outside color scheme")
            else:
                if inside_color_scheme == 1:
                    value  = dist
                elif inside_color_scheme == 2:
                    value = 60*trap_fun_dist
                else:
                    if trapped:
                        value = dist
                    else:
                        value = 0

            mandelbrot_set[row, col] = value

    mandelbrot_set = np.log(mandelbrot_set + 1)
    return mandelbrot_set


def complex_to_screen(z, rmin, rmax, imin, imax, width, height):
    re = z.real
    im = z.imag
    x = int((re - rmin) / (rmax - rmin) * width)
    y = int((im - imin) / (imax - imin) * height)
    return x, y


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


def draw_label_with_outline(screen, text, font, pos, text_color=(255, 255, 255), outline_color=(0, 0, 0), alpha=1.0):
    x, y = pos
    a = max(0, min(255, int(alpha * 255)))  # Clamp to [0, 255]

    # Render text and outline
    text_surface = font.render(text, True, text_color)
    outline_surface = font.render(text, True, outline_color)

    # Create transparent surfaces for alpha blending
    text_surf_alpha = pygame.Surface(text_surface.get_size(), pygame.SRCALPHA)
    outline_surf_alpha = pygame.Surface(outline_surface.get_size(), pygame.SRCALPHA)

    # Apply alpha manually
    text_surf_alpha.blit(text_surface, (0, 0))
    outline_surf_alpha.blit(outline_surface, (0, 0))
    text_surf_alpha.set_alpha(a)
    outline_surf_alpha.set_alpha(a)

    # Blit outlines
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx != 0 or dy != 0:
                screen.blit(outline_surf_alpha, (x + dx, y + dy))

    # Blit main text
    screen.blit(text_surf_alpha, (x, y))


def draw_state(screen, font, width : int, height : int, rmin : float, rmax : float, imin : float, imax : float, max_iter : int):
    x, y = pygame.mouse.get_pos()
    c = screen_to_complex(x, y, rmin, rmax, imin, imax, width, height)

    dr = rmax - rmin

    text = f"pos: {c.real:4.14g}, {c.imag:4.14g}; max_iter= {max_iter}; dr= {dr:4.4g}; dr/iter={max_iter/dr:1.2f};"
    label_surface = font.render(text, True, (255, 255, 255))

    pos_x = 5
    pos_y = screen.get_height() - label_surface.get_height() - 5

    draw_label_with_outline(screen, text, font, (pos_x + 2, pos_y + 2))


def draw_log(screen, font, width: int, height: int, rmin: float, rmax: float, imin: float, imax: float, max_iter: int):
    global log

    while len(log) > 10:
        log.pop(0)

    pos_x = 5
    pos_y = height - ((len(log)+1) * font.get_linesize()) - 10

    for i in range(len(log)):
        line = log[i]
        alpha = float(i) / len(log)  # 0 for oldest, close to 1 for newest
        draw_label_with_outline(screen, line, font, (pos_x + 2, pos_y + 2), (255, 255, 255), (0, 0, 0, 0), alpha)
        pos_y += font.get_linesize()



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

def draw_help(screen, font, width, height):
    help_text = [
        "Mandelbrot Explorer - Key Bindings",
        "F1       – Toggle this help",
        "F2       – Toggle orbit preview",
        "A        – Toggle axes",
        "I        – Cycle inside color scheme (Shift = reverse)",
        "O        – Cycle outside color scheme (Shift = reverse)",
        "S        – Toggle position display",
        "T        – Cycle orbit trap mode (Shift = reverse)",
        "+ / -    – Increase / decrease max iterations",
        "Ctrl+S   – Save screenshot",
        "Double click – Zoom in",
        "Single click – Retain orbit (if orbit preview visible)",
        "Mouse Wheel – Color Cycle",
    ]

    x, y = 20, 20
    padding = 5

    for i, line in enumerate(help_text):
        draw_label_with_outline(screen, line, font, (x, y), (255, 255, 255), (0, 0, 0), 0.9)
        y += font.get_linesize() + padding


@njit(parallel=True, fastmath=True, cache=True)
def compute_mandelbrot_parallel(
        width : int, 
        height : int, 
        max_iter : int, 
        rmin : float, 
        rmax : float, 
        imin : float, 
        imax : float, 
        inside_color_scheme : InsideColorScheme, 
        outside_color_scheme : OutsideColorScheme, 
        orbit_trap : GeometricOrbitTrap):
    
    # Split computation domain into n*n chunks and compute mandelbrot set in parallel
    n = 8
    chunk_width = width // n
    chunk_height = height // n

    rmin_chunk, rmax_chunk = [], []
    imin_chunk, imax_chunk = [], []

    dr = (rmax-rmin) / n
    di = (imax-imin) / n

    for r in range(n):
        for c in range(n):
            rmin_chunk.append(rmin + c * dr)
            rmax_chunk.append(rmin + (c+1) * dr)
            imin_chunk.append(imin + r * di)
            imax_chunk.append(imin + (r+1) * di)

    mandelbrot_array = np.zeros((height, width), dtype=np.float64)
    for k in prange(n*n):
        mandelbrot_chunk = compute_mandelbrot(
            chunk_width,
            chunk_height,
            max_iter,
            rmin_chunk[k],
            rmax_chunk[k],
            imin_chunk[k],
            imax_chunk[k],
            inside_color_scheme.value,
            outside_color_scheme.value, 
            orbit_trap.value)
        
        i = k // n
        j = k % n
        mandelbrot_array[i*chunk_height:(i+1)*chunk_height, j*chunk_width:(j+1)*chunk_width] = mandelbrot_chunk
        
    return mandelbrot_array


def color_mandelbrot(mandelbrot_array, mask, color_scheme, color_start):
    if color_scheme==1:
        start, freq = color_start/20, 9
        r = mask * (0.5 + 0.5*np.cos(start + mandelbrot_array * freq + 0))
        g = mask * (0.5 + 0.5*np.cos(start + mandelbrot_array * freq + 0.6))
        b = mask * (0.5 + 0.5*np.cos(start + mandelbrot_array * freq + 1.0))
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
                    h = (color_start + (mandelbrot_array[i, j] * 360)) % 360
                    s = 1
                    v = .5
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


def append_log(text):
    global log

    log.append(text)
    pygame.time.set_timer(EVENT_REMOVE_LOG, 5000)


def main(width, height, max_iter):
    global show_log
    global show_state
    global show_axis
    global show_orbits
    global show_help

    global EVENT_DOUBLE_CLICK
    global EVENT_REMOVE_LOG

    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Mandelbrot Explorer")

    saved_orbits = []
    preview_orbit = []

    timer_set = False
    running = True
    click_count = 0

    font = pygame.font.SysFont("Arial", 20)
    small_font = pygame.font.SysFont("Arial", 14)
    help_font = pygame.font.SysFont("Courier New", 18)

    #
    # Render Settings
    #

    color_scheme = 1
    color_start = 1
    inside_color_scheme = InsideColorScheme.TrapFunction
    outside_color_scheme = OutsideColorScheme.SmoothIterationCountPlusTrapFunction
    orbit_trap = GeometricOrbitTrap.NoTrap
    aoi = complex(3, 3)
    pos = complex(-0.5, 0)
    recompute = True

    rmin : float = 0
    rmax : float = 0
    imin : float = 0
    imax : float = 0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F1:
                    show_help = not show_help
                    append_log(f"Showing help" if show_help else "Hiding help")

                elif event.key == pygame.K_F2:
                    show_orbits = not show_orbits
                    append_log(f"Showing orbits" if show_orbits else "Hiding orbits")
                    
                    if not show_orbits:
                        saved_orbits.clear()
                        preview_orbit.clear()

                elif event.key == pygame.K_a:
                    show_axis = not show_axis
                    append_log(f"Showing axis" if show_orbits else "Hiding axis")

                elif event.key == pygame.K_c:
                    if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                        color_scheme = (color_scheme - 1) % 4
                    else:
                        color_scheme = (color_scheme + 1) % 4

                    append_log(f"Color scheme set to {color_scheme}")
                    recompute = True

                elif event.key == pygame.K_i:
                    enum_values = list(InsideColorScheme)
                    dir = 1 if pygame.key.get_mods() & pygame.KMOD_SHIFT else (len(enum_values) - 1)
                    inside_color_scheme = enum_values[(inside_color_scheme.value + dir) % len(enum_values)]
                    append_log(f"Inside color set to {inside_color_scheme.name}")
                    recompute = True

                elif event.key == pygame.K_o:
                    enum_values = list(OutsideColorScheme)                   
                    dir = 1 if pygame.key.get_mods() & pygame.KMOD_SHIFT else (len(enum_values) - 1)
                    outside_color_scheme = enum_values[(outside_color_scheme.value + dir) % len(enum_values)]
                    append_log(f"Outside color set to {outside_color_scheme.name}")
                    recompute = True

                elif event.key == pygame.K_x:
                    max_iter = 1
                    recompute = True

                elif event.key == pygame.K_n:
                    if pygame.key.get_mods() & pygame.KMOD_CTRL:
                        max_iter += 1
                    elif pygame.key.get_mods() & pygame.KMOD_SHIFT:
                        max_iter = int(max_iter * 0.75)
                    else:
                        max_iter = int(max_iter * 1.25)

                    append_log(f"Increasing max iterations to {max_iter}")
                    recompute = True

                elif event.key == pygame.K_r:
                    aoi = complex(3, 3)
                    pos = complex(-0.5, 0)
                    max_iter = 200
                    append_log(f"Resetting View")
                    recompute = True

                elif event.key == pygame.K_s:
                    if pygame.key.get_mods() & pygame.KMOD_CTRL:
                        filename = f"{script_path}/mandelbrot_r={pos.real:2.12g},i={pos.imag:2.12g},it={max_iter},color_start={color_start}_{str(uuid.uuid4())[:8]}.png"
                        pygame.image.save(screen, filename)
                        append_log(f"Saved {filename}")
                    else:
                        show_state = not show_state

                elif event.key == pygame.K_t:
                    enum_values = list(GeometricOrbitTrap)                   
                    dir = 1 if pygame.key.get_mods() & pygame.KMOD_SHIFT else (len(enum_values) - 1)
                    orbit_trap = enum_values[(orbit_trap.value + dir) % len(enum_values)]
                    append_log(f"Orbit Trap set to {orbit_trap.name}")
                    recompute = True

                elif event.key == pygame.K_PLUS:
                    max_iter = int(max_iter * 1.24)
                    append_log(f"Increasing max iterations to {max_iter}")
                    recompute = True

                elif event.key == pygame.K_MINUS:
                    max_iter = int(max_iter * 0.75)
                    append_log(f"Reducing max Iterations to {max_iter}")
                    recompute = True

                elif event.key == pygame.K_SPACE:
                    dr = rmax - rmin
                    str_data = f"{dr:4.10g}, {max_iter}\r\n"
                    # append str to file "iter.txt"
                    with open("iter.txt", "a") as f:
                        f.write(str_data)
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:
                    color_start += 1
                elif event.button == 5:
                    color_start -= 1
                elif not timer_set:
                    pygame.time.set_timer(EVENT_DOUBLE_CLICK, 250)
                    click_count += 1

            elif event.type == EVENT_DOUBLE_CLICK:
                if click_count == 1:
                    if preview_orbit:
                        saved_orbits.append(preview_orbit.copy())
                        append_log(f"Saving orbit")
                elif click_count == 2:
                    mx, my = pygame.mouse.get_pos()
                    pos = screen_to_complex(mx, my, rmin, rmax, imin, imax, width, height)
                    aoi *= .5
                    append_log(f"Zooming in to {pos.real:1.7f}, {pos.imag:1.7f}")
                    recompute = True

                pygame.time.set_timer(EVENT_DOUBLE_CLICK, 0)
                timer_set = False
                click_count = 0

            elif event.type == EVENT_REMOVE_LOG:                       
                show_log = False

        #
        # Do the rendering
        #

        if recompute:
            rmin,rmax,imin,imax = set_position(pos, aoi, width, height)
            
            start_ticks = pygame.time.get_ticks()
            mandelbrot_array = compute_mandelbrot_parallel(width, height, max_iter, rmin, rmax, imin, imax, inside_color_scheme, outside_color_scheme, orbit_trap)
            mandelbrot_array = mandelbrot_array / np.max(mandelbrot_array)
            mask = mandelbrot_array != 0
            elapsed = pygame.time.get_ticks() - start_ticks
            print(f"Rendering took {elapsed} ms")
            recompute = False
        
        surface = color_mandelbrot(mandelbrot_array, mask, color_scheme, color_start)
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
            draw_state(screen, help_font, width, height, rmin, rmax, imin, imax, max_iter)

        if show_log:
            draw_log(screen, small_font, width, height, rmin, rmax, imin, imax, max_iter)

        if show_help:
            draw_help(screen, help_font, width, height)

        pygame.display.update()

    pygame.quit()


if __name__ == "__main__":
    scale = 1.5
    main(int(scale*860), int(scale*540), 300)