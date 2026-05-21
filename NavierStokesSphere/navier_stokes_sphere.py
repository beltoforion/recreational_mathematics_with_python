"""
Navier-Stokes (Shallow-Water) Simulation auf einer rotierenden Kugel.

Loest die Flachwassergleichungen auf einem sphaerischen Voronoi-Gitter,
um zu zeigen, wie die Corioliskraft auf einem rotierenden Planeten zu
charakteristischen Wirbelstrukturen fuehrt (analog zu Hoch- und Tiefdruck-
gebieten in der Erdatmosphaere).

Bedienung:
  - Maus ziehen:  Kugel drehen
  - Mausrad:      Hineinzoomen / herauszoomen
  - Taste 'r':    Simulation zuruecksetzen
  - Taste ' ':    Neue zufaellige Stoerung hinzufuegen
  - Taste 'c':    Farbgebung wechseln (Vortizitaet / Hoehe)
"""

import numpy as np
from scipy.spatial import SphericalVoronoi
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm


# ----------------------------- Parameter ----------------------------------- #
N_POINTS         = 800      # Anzahl Voronoi-Zellen
RADIUS           = 1.0      # Kugelradius (Planet)
OMEGA            = 6.0      # Eigenrotationsrate (Staerke der Corioliskraft)
G                = 1.0      # reduzierte Schwerkraft
H0               = 1.0      # mittlere Fluid-Hoehe
DT               = 0.0015   # Zeitschrittweite
NU               = 5e-4     # kuenstliche Viskositaet (Stabilitaet)
DRAG             = 5e-3     # leichte Reibung gegen Energie-Akkumulation
STEPS_PER_FRAME  = 6
COLORMAP_VORT    = cm.RdBu_r
COLORMAP_H       = cm.viridis


# -------------------------- Gittererzeugung -------------------------------- #
def fibonacci_sphere(n):
    """Quasi-uniforme Punktverteilung auf der Einheitskugel (Fibonacci-Spirale)."""
    indices = np.arange(0, n, dtype=float) + 0.5
    phi = np.arccos(1.0 - 2.0 * indices / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * indices
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.column_stack([x, y, z])


def build_mesh(n_points):
    """Erzeugt sphaerisches Voronoi-Gitter und alle geometrischen Hilfsgroessen."""
    points = fibonacci_sphere(n_points)
    sv = SphericalVoronoi(points, RADIUS, np.zeros(3))
    sv.sort_vertices_of_regions()  # CCW-Reihenfolge (von aussen betrachtet)
    areas = sv.calculate_areas()

    n = len(points)

    # Kanten-Dictionary: (v_a, v_b) -> [Zelle1, Zelle2]
    edge_dict = {}
    for i, region in enumerate(sv.regions):
        nv = len(region)
        for k in range(nv):
            a, b = region[k], region[(k + 1) % nv]
            key = (min(a, b), max(a, b))
            edge_dict.setdefault(key, []).append(i)

    # Nachbarn (CCW), pro Zelle in der Reihenfolge der Regionen-Vertices
    max_nbr = max(len(r) for r in sv.regions)
    nbr_arr      = -np.ones((n, max_nbr), dtype=int)
    edge_len_arr = np.zeros((n, max_nbr))
    dist_arr     = np.ones((n, max_nbr))      # >0 (Division)
    n_out_arr    = np.zeros((n, max_nbr, 3))  # Aussennormale (tangential zur Kugel)
    e_tan_arr    = np.zeros((n, max_nbr, 3))  # Tangente entlang Kante in CCW-Richtung
    nbr_count    = np.zeros(n, dtype=int)

    for i, region in enumerate(sv.regions):
        nv = len(region)
        pi = points[i]
        for k in range(nv):
            a, b = region[k], region[(k + 1) % nv]
            key = (min(a, b), max(a, b))
            cells = edge_dict[key]
            j = cells[0] if cells[1] == i else cells[1] if len(cells) > 1 else -1
            if j < 0:
                continue

            v1 = sv.vertices[a]
            v2 = sv.vertices[b]

            # Bogenlaenge der Kante
            dot = np.clip(np.dot(v1, v2) / RADIUS ** 2, -1.0, 1.0)
            L = RADIUS * np.arccos(dot)

            # Abstand der Zellzentren (Grosskreis)
            pj = points[j]
            dot2 = np.clip(np.dot(pi, pj) / RADIUS ** 2, -1.0, 1.0)
            d = RADIUS * np.arccos(dot2)

            # Aussennormale tangential zur Kugel am Mittelpunkt der Verbindung i->j
            n_out = pj - np.dot(pj, pi) * pi
            n_out /= np.linalg.norm(n_out) + 1e-14

            # Tangente entlang der Kante in CCW-Richtung (v1 -> v2)
            t = v2 - v1
            mid = 0.5 * (v1 + v2)
            mid /= np.linalg.norm(mid) + 1e-14
            t = t - np.dot(t, mid) * mid
            t /= np.linalg.norm(t) + 1e-14

            nbr_arr[i, k]      = j
            edge_len_arr[i, k] = L
            dist_arr[i, k]     = max(d, 1e-12)
            n_out_arr[i, k]    = n_out
            e_tan_arr[i, k]    = t
            nbr_count[i] += 1

    mask = (nbr_arr >= 0).astype(float)
    return dict(
        points=points, sv=sv, areas=areas,
        nbr=nbr_arr, mask=mask,
        edge_len=edge_len_arr, dist=dist_arr,
        n_out=n_out_arr, e_tan=e_tan_arr,
        nbr_count=nbr_count, max_nbr=max_nbr,
    )


# ------------------------- Anfangsbedingungen ------------------------------ #
def gaussian_bump(points, center, amp=0.25, width=0.35):
    """Gauss-Hoegel ueber Geodaeten-Distanz auf der Kugel."""
    c = center / (np.linalg.norm(center) + 1e-14)
    cos_d = np.clip(points @ c, -1.0, 1.0)
    return amp * np.exp(-((1.0 - cos_d) / width ** 2))


def initial_conditions(points):
    n = len(points)
    h = H0 * np.ones(n)
    # mehrere Stoerungen verteilt auf der Kugel
    centers = [
        np.array([ 1.0,  0.0,  0.3]),
        np.array([-0.7,  0.8,  0.1]),
        np.array([ 0.2, -0.9,  0.5]),
        np.array([-0.4, -0.4, -0.8]),
    ]
    amps = [0.30, -0.25, 0.22, -0.20]
    for c, a in zip(centers, amps):
        h += gaussian_bump(points, c, amp=a, width=0.35)
    u = np.zeros((n, 3))
    return h, u


# --------------------------- Operatoren ------------------------------------ #
def project_tangent(vec, points):
    """Projiziert ein Vektorfeld auf den Tangentialraum der Kugel."""
    radial = np.sum(vec * points, axis=1, keepdims=True)
    return vec - radial * points


def green_gauss_gradient(field, mesh):
    """Skalar-Gradient pro Zelle (Green-Gauss), tangential zur Kugel."""
    nbr = mesh['nbr']
    nbr_safe = np.where(nbr >= 0, nbr, 0)
    f_j = field[nbr_safe]
    face_val = 0.5 * (field[:, None] + f_j)                      # (N, K)
    contrib = (face_val * mesh['edge_len'] * mesh['mask'])[:, :, None] * mesh['n_out']
    grad = np.sum(contrib, axis=1) / mesh['areas'][:, None]
    return project_tangent(grad, mesh['points'])


def divergence(vec_field, mesh, h_for_upwind=None):
    """Divergenz von h*u (mit Upwind), bzw. divergenz wenn h_for_upwind=None."""
    nbr = mesh['nbr']
    nbr_safe = np.where(nbr >= 0, nbr, 0)
    u_j = vec_field[nbr_safe]                                    # (N, K, 3)
    u_i = vec_field[:, None, :]
    n   = mesh['n_out']
    u_n = 0.5 * (np.sum(u_i * n, axis=-1) + np.sum(u_j * n, axis=-1))

    if h_for_upwind is None:
        flux = u_n
    else:
        h_j = h_for_upwind[nbr_safe]
        h_up = np.where(u_n > 0, h_for_upwind[:, None], h_j)
        flux = h_up * u_n

    return np.sum(flux * mesh['edge_len'] * mesh['mask'], axis=1) / mesh['areas']


def scalar_laplacian_vec(vec_field, mesh):
    """Komponentenweiser Laplace-Operator fuer ein Tangentialvektorfeld."""
    nbr = mesh['nbr']
    nbr_safe = np.where(nbr >= 0, nbr, 0)
    v_j = vec_field[nbr_safe]                                    # (N, K, 3)
    grad_face = (v_j - vec_field[:, None, :]) / mesh['dist'][:, :, None]
    contrib = grad_face * (mesh['edge_len'] * mesh['mask'])[:, :, None]
    lap = np.sum(contrib, axis=1) / mesh['areas'][:, None]
    return lap


def vorticity(u, mesh):
    """Radiale Komponente der Vortizitaet via Stokes'scher Zirkulation."""
    nbr = mesh['nbr']
    nbr_safe = np.where(nbr >= 0, nbr, 0)
    u_j = u[nbr_safe]
    u_edge = 0.5 * (u[:, None, :] + u_j)
    circ = np.sum(u_edge * mesh['e_tan'], axis=-1) * mesh['edge_len'] * mesh['mask']
    return np.sum(circ, axis=1) / mesh['areas']


# ----------------------------- Zeitschritt --------------------------------- #
def step(h, u, mesh, omega_vec):
    grad_h = green_gauss_gradient(h, mesh)
    div_hu = divergence(u, mesh, h_for_upwind=h)
    coriolis = -2.0 * np.cross(omega_vec[None, :], u)
    coriolis = project_tangent(coriolis, mesh['points'])
    lap_u = scalar_laplacian_vec(u, mesh)

    dh = -div_hu
    du = -G * grad_h + coriolis + NU * lap_u - DRAG * u

    h_new = h + DT * dh
    u_new = u + DT * du
    u_new = project_tangent(u_new, mesh['points'])
    return h_new, u_new


# --------------------------- Visualisierung -------------------------------- #
class Simulation:
    def __init__(self):
        print(f"Erzeuge sphaerisches Voronoi-Gitter mit {N_POINTS} Zellen ...")
        self.mesh = build_mesh(N_POINTS)
        self.omega_vec = np.array([0.0, 0.0, OMEGA])
        self.h, self.u = initial_conditions(self.mesh['points'])
        self.color_mode = 'vorticity'

        polys = [self.mesh['sv'].vertices[r] for r in self.mesh['sv'].regions]
        self.fig = plt.figure(figsize=(10, 9))
        self.ax = self.fig.add_subplot(111, projection='3d')
        try:
            self.ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass
        self.collection = Poly3DCollection(
            polys, edgecolors=(0, 0, 0, 0.08), linewidths=0.25
        )
        self.ax.add_collection3d(self.collection)
        L = 1.05 * RADIUS
        self.ax.set_xlim(-L, L); self.ax.set_ylim(-L, L); self.ax.set_zlim(-L, L)
        self.ax.set_axis_off()
        self.title = self.ax.set_title(
            "Navier-Stokes auf rotierender Kugel  (Coriolis-induzierte Wirbel)\n"
            "Maus: drehen   |   SPACE: Stoerung   |   r: Reset   |   c: Farbe wechseln",
            fontsize=10,
        )
        self._refresh_colors()

        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.ani = FuncAnimation(
            self.fig, self._animate, interval=25, blit=False, cache_frame_data=False
        )

    # --- Farb-Update ------------------------------------------------------- #
    def _refresh_colors(self):
        if self.color_mode == 'vorticity':
            field = vorticity(self.u, self.mesh)
            vmax = max(2.5, 0.9 * np.max(np.abs(field)) + 1e-9)
            vals = np.clip(field / vmax, -1.0, 1.0) * 0.5 + 0.5
            colors = COLORMAP_VORT(vals)
        else:
            field = self.h - H0
            vmax = max(0.15, 0.9 * np.max(np.abs(field)) + 1e-9)
            vals = np.clip(field / vmax, -1.0, 1.0) * 0.5 + 0.5
            colors = COLORMAP_H(vals)
        self.collection.set_facecolors(colors)

    # --- Tastatur ---------------------------------------------------------- #
    def _on_key(self, event):
        if event.key == 'r':
            self.h, self.u = initial_conditions(self.mesh['points'])
        elif event.key == ' ':
            # zufaellige Stoerung
            c = np.random.randn(3)
            c /= np.linalg.norm(c)
            amp = np.random.choice([-1, 1]) * np.random.uniform(0.15, 0.3)
            self.h = self.h + gaussian_bump(self.mesh['points'], c, amp=amp, width=0.3)
        elif event.key == 'c':
            self.color_mode = 'height' if self.color_mode == 'vorticity' else 'vorticity'

    # --- Animation --------------------------------------------------------- #
    def _animate(self, _frame):
        for _ in range(STEPS_PER_FRAME):
            self.h, self.u = step(self.h, self.u, self.mesh, self.omega_vec)
        self._refresh_colors()
        return (self.collection,)

    def run(self):
        plt.show()


def main():
    np.random.seed(1)
    Simulation().run()


if __name__ == '__main__':
    main()
