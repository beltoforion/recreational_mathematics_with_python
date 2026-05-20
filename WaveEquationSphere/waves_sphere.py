"""Wave equation on the surface of a sphere, solved on a spherical Voronoi mesh.

The Laplace-Beltrami operator is discretized with the finite-volume two-point
flux approximation on the Voronoi tessellation:

    (Δ_S u)_i  ≈  (1 / A_i) * Σ_j (L_ij / d_ij) * (u_j - u_i)

where A_i is the spherical area of cell i, L_ij is the great-circle arc length
of the edge shared with neighbor j, and d_ij is the great-circle distance
between the two generator points.

Time integration is the standard leapfrog scheme

    u^{n+1} = 2 u^n - u^{n-1} + (c Δt)^2 (Δ_S u^n)

Since a sphere has no boundary, there are no boundary conditions to worry
about: a single Gaussian pulse spreads out, refocuses at the antipodal point,
returns to the source, and so on - a clean demonstration of energy
conservation on a closed manifold.
"""

import math
import time

import numpy as np
import pygame
from scipy.spatial import SphericalVoronoi


# --- Simulation parameters ----------------------------------------------------

N_POINTS    = 12000    # number of Voronoi cells on the sphere
RADIUS      = 1.0      # sphere radius
WAVE_SPEED  = 0.5      # propagation speed c
CFL         = 0.35     # safety factor for the time step
SCREEN_SIZE = 800      # window edge length in pixels
AUTO_ROTATE = 0.003    # radians per frame around the vertical axis (paused while dragging)
MOUSE_SENS  = 0.008    # radians per pixel of mouse drag
SUBSTEPS    = 2        # physics steps per rendered frame
RAIN_RATE   = 0.04     # probability per frame of a new "raindrop"
RAIN_SIGMA  = 0.035    # angular width of a raindrop
RAIN_AMPL   = 0.8      # amplitude of a raindrop


# --- Mesh generation ----------------------------------------------------------

def fibonacci_sphere(n: int) -> np.ndarray:
    """Return n approximately uniformly distributed points on the unit sphere."""
    indices = np.arange(n, dtype=np.float64) + 0.5
    phi   = np.arccos(1.0 - 2.0 * indices / n)
    theta = math.pi * (1.0 + math.sqrt(5.0)) * indices
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    return np.stack([x, y, z], axis=1)


def build_mesh(n_points: int, radius: float):
    """Build the spherical Voronoi mesh and precompute finite-volume geometry.

    Returns
    -------
    points   : (N, 3)   generator points on the sphere
    sv       : SphericalVoronoi
    areas    : (N,)     spherical cell areas
    src, dst : (E,)     directed edge endpoints (each undirected edge twice)
    weights  : (E,)     L_ij / d_ij for each directed edge
    min_d    : float    smallest generator distance (drives the CFL condition)
    """
    points = fibonacci_sphere(n_points) * radius
    sv = SphericalVoronoi(points, radius=radius, center=np.zeros(3))
    sv.sort_vertices_of_regions()
    areas = sv.calculate_areas()

    # Map each undirected polygon edge (pair of Voronoi vertex indices) to the
    # two generator cells that share it.
    edge_to_cells: dict[tuple[int, int], list[int]] = {}
    for i, region in enumerate(sv.regions):
        m = len(region)
        for k in range(m):
            v0 = region[k]
            v1 = region[(k + 1) % m]
            key = (v0, v1) if v0 < v1 else (v1, v0)
            edge_to_cells.setdefault(key, []).append(i)

    src_list:    list[int]   = []
    dst_list:    list[int]   = []
    weight_list: list[float] = []
    d_min = float("inf")

    inv_r2 = 1.0 / (radius * radius)
    for (va, vb), cells in edge_to_cells.items():
        if len(cells) != 2:
            # Should not happen on a closed sphere with a clean Voronoi diagram.
            continue
        i, j = cells

        cos_ab = np.clip(sv.vertices[va] @ sv.vertices[vb] * inv_r2, -1.0, 1.0)
        L = math.acos(cos_ab) * radius

        cos_ij = np.clip(points[i] @ points[j] * inv_r2, -1.0, 1.0)
        d = math.acos(cos_ij) * radius

        w = L / d
        src_list.append(i); dst_list.append(j); weight_list.append(w)
        src_list.append(j); dst_list.append(i); weight_list.append(w)
        if d < d_min:
            d_min = d

    src     = np.asarray(src_list,    dtype=np.int32)
    dst     = np.asarray(dst_list,    dtype=np.int32)
    weights = np.asarray(weight_list, dtype=np.float64)
    return points, sv, areas, src, dst, weights, d_min


# --- Initial condition --------------------------------------------------------

def add_gaussian(u: np.ndarray, points: np.ndarray, radius: float,
                 direction: np.ndarray, amplitude: float, sigma: float) -> None:
    """Add a Gaussian bump centered at the surface point along `direction`.

    `sigma` is the angular width in radians (not the geodesic distance).
    """
    d = direction / np.linalg.norm(direction)
    cos_a = np.clip(points @ d / radius, -1.0, 1.0)
    angle = np.arccos(cos_a)
    bump = amplitude * np.exp(-(angle / sigma) ** 2)
    u[1] += bump
    u[2] += bump


# --- Rendering ----------------------------------------------------------------

def colormap(values: np.ndarray, scale: float) -> list:
    """Map signed displacement values to a list of (R, G, B) tuples."""
    v   = np.clip(values / scale, -1.0, 1.0)
    pos = np.clip( v, 0.0, 1.0)
    neg = np.clip(-v, 0.0, 1.0)
    rgb = np.empty((len(values), 3), dtype=np.int16)
    rgb[:, 0] = 255 - (255 * neg).astype(np.int16)
    rgb[:, 1] = 255 - (255 * (pos + neg)).astype(np.int16)
    rgb[:, 2] = 255 - (255 * pos).astype(np.int16)
    return rgb.tolist()


def draw_sphere(screen, vx, vy, points_z, cell_regions, colors):
    """Draw all front-facing Voronoi cells as filled polygons."""
    for i, region in enumerate(cell_regions):
        if points_z[i] <= 0.0:
            continue
        poly = [(int(vx[k]), int(vy[k])) for k in region]
        pygame.draw.polygon(screen, colors[i], poly)


# --- Main loop ----------------------------------------------------------------

def main() -> None:
    pygame.init()
    pygame.font.init()
    font = pygame.font.SysFont("Consolas", 15)
    screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
    pygame.display.set_caption("Wave Equation on a Sphere (Spherical Voronoi)")

    print(f"Building spherical Voronoi mesh with {N_POINTS} cells ...")
    t0 = time.time()
    points, sv, areas, src, dst, weights, min_d = build_mesh(N_POINTS, RADIUS)
    print(f"  done in {time.time() - t0:.2f} s")

    n_edges = len(src) // 2
    dt = CFL * min_d / WAVE_SPEED
    c2dt2 = (WAVE_SPEED * dt) ** 2
    print(f"  cells = {len(points)}, edges = {n_edges}")
    print(f"  min generator distance = {min_d:.4f}")
    print(f"  dt = {dt:.5f}  (CFL = {CFL})")

    N = len(points)
    u = np.zeros((3, N), dtype=np.float64)
    rng = np.random.default_rng()

    def random_raindrop() -> None:
        d = rng.normal(size=3)
        add_gaussian(u, points, RADIUS,
                     direction=d, amplitude=RAIN_AMPL, sigma=RAIN_SIGMA)

    random_raindrop()

    vertices_local = sv.vertices.copy()

    # Rotation state
    angle_y = 0.0
    angle_x = 0.35
    dragging = False
    last_mouse = (0, 0)

    clock = pygame.time.Clock()
    tick = 0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    random_raindrop()
                elif event.key == pygame.K_r:
                    u[:] = 0.0
                    random_raindrop()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                dragging = True
                last_mouse = event.pos
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                dragging = False
            elif event.type == pygame.MOUSEMOTION and dragging:
                dx = event.pos[0] - last_mouse[0]
                dy = event.pos[1] - last_mouse[1]
                angle_y += dx * MOUSE_SENS
                angle_x += dy * MOUSE_SENS
                # clamp vertical tilt so the poles don't flip past the camera
                angle_x = max(-math.pi / 2, min(math.pi / 2, angle_x))
                last_mouse = event.pos

        # ---- random raindrop sources -----------------------------------------
        if rng.random() < RAIN_RATE:
            random_raindrop()

        # ---- physics: leapfrog update on the Voronoi mesh --------------------
        for _ in range(SUBSTEPS):
            flux = weights * (u[1, dst] - u[1, src])
            lap = np.zeros(N, dtype=np.float64)
            np.add.at(lap, src, flux)
            lap /= areas

            u_next = 2.0 * u[1] - u[2] + c2dt2 * lap
            u[2] = u[1]
            u[1] = u_next
            tick += 1

        # ---- rotation & projection ------------------------------------------
        if not dragging:
            angle_y += AUTO_ROTATE
        cy, sy = math.cos(angle_y), math.sin(angle_y)
        cx, sx = math.cos(angle_x), math.sin(angle_x)
        Ry = np.array([[ cy, 0.0,  sy],
                       [0.0, 1.0, 0.0],
                       [-sy, 0.0,  cy]])
        Rx = np.array([[1.0, 0.0, 0.0],
                       [0.0,  cx, -sx],
                       [0.0,  sx,  cx]])
        R = Rx @ Ry

        points_rot   = points          @ R.T
        vertices_rot = vertices_local  @ R.T

        scale = SCREEN_SIZE * 0.42
        cx_s  = SCREEN_SIZE // 2
        cy_s  = SCREEN_SIZE // 2
        vx = ( vertices_rot[:, 0] * scale + cx_s)
        vy = (-vertices_rot[:, 1] * scale + cy_s)

        # ---- coloring --------------------------------------------------------
        u_curr  = u[1]
        max_abs = max(0.4, float(np.abs(u_curr).max()))
        colors  = colormap(u_curr, max_abs)

        # ---- draw ------------------------------------------------------------
        screen.fill((18, 18, 28))
        draw_sphere(screen, vx, vy, points_rot[:, 2], sv.regions, colors)

        info = (f"Spherical Voronoi  |  N = {N} cells, {n_edges} edges  |  "
                f"t = {tick * dt:5.2f} s  |  FPS = {clock.get_fps():4.1f}")
        screen.blit(font.render(info, True, (230, 230, 230)), (8, 6))
        screen.blit(font.render("LMB drag: rotate   SPACE: drop   R: reset   ESC: quit",
                                True, (170, 170, 180)),
                    (8, SCREEN_SIZE - 22))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
