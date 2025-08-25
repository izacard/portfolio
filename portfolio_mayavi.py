"""
Gyrointegral 3D Visualization using Mayavi
-----------------------------------------

This script generates high-resolution 3D scientific visualizations of
particle dynamics, gyrointegrals, and magnetic field geometry using
Mayavi. It demonstrates the use of advanced scientific Python tooling
to produce publication-quality visuals for research and communication.
The gyrointegrated kinetic theory is a novel approach different from
the well-known gyrokinetic theory reduced from the 6D kinetic theory
to statistically describe the dyanmics of particles in plasma physics.

Highlights:
- Publication-quality 3D scientific visualization
- Modular functions for particle, gyrointegral, and magnetic field plotting
- Clear structure, reproducible configuration, and portfolio-ready code
"""

import os
import glob
import datetime
import logging
import numpy as np
from mayavi import mlab  # python3 -m np.pip install mayavi

from mayavi.core.scene import Scene  # mlab.Figure
from mayavi.modules.surface import Surface  # mlab.Surface


# ======================
# Logging configuration
# ======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ======================
# Configuration
# ======================
OUTPUT_DIR = "Images/mayavi"
OUTPUT_PREFIX = "mayavi_GI_3D_gyrointegral_"
FIG_SIZE = (1500, 1500)  # recommended for faster run (pixels)
FIG_SIZE = (6000, 6000)  # recommended for papers (pixels)
POINT_RESOLUTION = 80
TUBE_RADIUS = 0.01
TUBE_AXIS = 0.002
BACKGROUND_COLOR = (0.85, 0.85, 0.85)  # Light grey background

# Color palette
COLORS = {
    "basis": (0, 0, 0),  # black
    "Bfield": (0, 0, 0),  # black
    "particle": (0.5, 0.5, 0.5),  # grey
    "selected": (1, 0, 1),  # magenta
    "gyrointegral": (0, 1, 0),  # green
    "inst_gyrocenter": (0, 1, 1),  # cyan
    "vz": (1, 0.55, 0.05),  # orange
    "x": (0, 0, 0),  # black
    "gc": (1, 0, 0),  # red
    "rho": (0.2, 0.2, 1),  # slightly lighter blue
}


# ======================
# Utility Functions
# ======================
def generate_demo_vectors(
    n_vec: int = 10, scale: float = 0.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic particle vectors for demonstration.

    Args:
        n_vec: Number of vectors to generate.
        scale: Scaling factor for vector magnitude.

    Returns:
        Tuple of (u, v, w) vector components.
    """
    vperp = scale * np.ones((n_vec))
    # linear: f(alpha) = const
    theta0 = np.linspace(0, 2 * np.pi * (n_vec - 1) / (n_vec + 0.0), n_vec)
    # nonlinear: f(alpha) not = const
    theta0 = -1.0 * np.pi / 2.0
    tt = np.linspace(-1, 1 - 2 / (n_vec + 0.0), n_vec)
    cc = 0.3
    theta = np.pi * (1 + cc * tt + (1 - cc) * tt**3) - theta0

    u = vperp * np.cos(theta)
    v = vperp * np.sin(theta)
    w = 3 * scale * np.ones((n_vec))
    return u, v, w


def plot_particle_vectors(
    fig: object,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    indices: list[int],
    b: float = 0.9,
) -> object:
    """Plot particle vectors.

    Args:
        fig: Mayavi figure object.
        u, v, w: Vector components (arrays of equal length).
        indices: Indices of vectors to plot.
        b: Blending factor for head/tail positioning.

    Returns:
        Mayavi quiver3d object.
    """
    obj = mlab.quiver3d(
        b * v[indices],
        b * w[indices],
        b * u[indices],
        (1 - b) * v[indices],
        (1 - b) * w[indices],
        (1 - b) * u[indices],
        line_width=0.75,
        scale_factor=1.0,
        color=COLORS["particle"],
        resolution=POINT_RESOLUTION,
        mode="cone",
        figure=fig,
    )
    for i, j, k in zip(v[indices], w[indices], u[indices]):
        mlab.plot3d(
            [0, b * i],
            [0, b * j],
            [0, b * k],
            color=COLORS["particle"],
            tube_radius=0.5 * TUBE_RADIUS,
            figure=fig,
        )
    return obj


def plot_selected_particle_vectors(
    fig: object, u: np.ndarray, v: np.ndarray, w: np.ndarray, k0: int, b: float = 0.9
) -> object:
    """Plot selected particle vector.

    Args:
        fig: Mayavi figure object.
        u, v, w: Vector components (arrays of equal length).
        k0: Index of the vector to highlight.
        b: Blending factor for head/tail positioning.

    Returns:
        Mayavi quiver3d object.
    """
    obj = mlab.quiver3d(
        b * v[k0],
        b * w[k0],
        b * u[k0],
        (1 - b) * v[k0],
        (1 - b) * w[k0],
        (1 - b) * u[k0],
        line_width=0.75,
        scale_factor=1.0,
        color=COLORS["selected"],
        resolution=POINT_RESOLUTION,
        mode="cone",
        figure=fig,
    )
    mlab.plot3d(
        [0, b * v[k0]],
        [0, b * w[k0]],
        [0, b * u[k0]],
        color=COLORS["selected"],
        tube_radius=0.5 * TUBE_RADIUS,
        figure=fig,
    )
    return obj


def plot_selected_particle_vectors_projection(
    fig: object, u: np.ndarray, v: np.ndarray, w: np.ndarray, k0: float, wgi: float
) -> None:
    """Plot selected particle vector projection.

    Args:
        fig: Mayavi figure object.
        u, v, w: Vector components (arrays of equal length).
        k0: Index of the vector to highlight.
        wgi: Vector component of the gyrointegrated (mean) velocity.
    """
    Nt = 101
    t = np.linspace(0, wgi, Nt)
    Nseg = 33.0
    dpts = int((Nt - 1) / Nseg)
    for ii in range(int((Nt - 1) / (2.0 * dpts))):
        jj = 2 * dpts * ii
        mlab.plot3d(
            [v[k0], v[k0]],
            [t[jj], t[jj + dpts]],
            [u[k0], u[k0]],
            color=COLORS["selected"],
            tube_radius=TUBE_RADIUS / 2.0,
            opacity=0.3,
            figure=fig,
        )


def plot_magnetic_field(fig: object, c0: float = 1.5, scale: float = 0.5) -> None:
    """Plot magnetic field lines

    Args:
        fig: Mayavi figure object.
        c0: Scaling factor for the field line extension.
        scale: Scaling factor for vector magnitude.
    """
    # Straigth magnetic field line   OR   AXIS
    mlab.plot3d(
        [0, 0],
        [0, 6 * scale],
        [0, 0],
        color=COLORS["basis"],
        tube_radius=TUBE_AXIS,
        figure=fig,
    )
    mlab.plot3d(
        [0, 0],
        [0, -3 * scale],
        [0, 0],
        color=COLORS["basis"],
        tube_radius=TUBE_AXIS,
        figure=fig,
    )
    mlab.plot3d(
        [0, -c0 * scale],
        [0, 0],
        [0, 0],
        color=COLORS["basis"],
        tube_radius=TUBE_AXIS,
        figure=fig,
    )
    mlab.plot3d(
        [0, 0],
        [0, 0],
        [0, c0 * scale],
        color=COLORS["basis"],
        tube_radius=TUBE_AXIS,
        figure=fig,
    )

    mlab.plot3d(
        [0, -c0 * scale],
        [3 * scale, 3 * scale],
        [0, 0],
        color=COLORS["basis"],
        tube_radius=TUBE_AXIS,
        figure=fig,
    )
    mlab.plot3d(
        [0, 0],
        [3 * scale, 3 * scale],
        [0, c0 * scale],
        color=COLORS["basis"],
        tube_radius=TUBE_AXIS,
        figure=fig,
    )

    # Curved magnetic field line
    Nt = 801
    t = np.linspace(-3 * scale, 6 * scale, Nt)
    x = 1.5 * scale / (6.0 * scale) ** 2 * t**2
    y = t
    z = np.zeros((Nt))
    mlab.plot3d(x, y, z, color=COLORS["Bfield"], tube_radius=TUBE_RADIUS, figure=fig)


def plot_gyrointegral_circles(fig: object, scale: float = 0.5) -> None:
    """Plot gyrointegral circles at particle position.

    Args:
        fig: Mayavi figure object.
        scale: Scaling factor for vector magnitude.
    """
    # Plot gyrointegral circle at particle position
    Nt = 361
    t = np.linspace(0, 2 * np.pi, Nt)
    cir_x, cir_y, cir_z = scale * np.sin(t), np.zeros((Nt)), scale * np.cos(t)
    mlab.plot3d(
        cir_x,
        cir_y,
        cir_z,
        color=COLORS["gyrointegral"],
        tube_radius=TUBE_RADIUS,
        figure=fig,
    )

    # Plot gyrointegral circle at particles position
    Nt = 361
    t = np.linspace(0, 2 * np.pi, Nt)
    cir_x, cir_y, cir_z = scale * np.sin(t), np.zeros((Nt)), scale * np.cos(t)
    Nseg = 60.0
    dpts = int((Nt - 1) / Nseg)
    for ii in range(int((Nt - 1) / (2.0 * dpts))):
        jj = 2 * dpts * ii
        mlab.plot3d(
            cir_x[jj : jj + dpts],
            cir_y[jj : jj + dpts],
            cir_z[jj : jj + dpts],
            color=COLORS["gyrointegral"],
            tube_radius=TUBE_RADIUS / 2.0,
            opacity=0.3,
            figure=fig,
        )

    # Plot gyrointegral circle at end of velocity vectors
    Nt = 361
    t = np.linspace(0, 2 * np.pi, Nt)
    cir_x, cir_y, cir_z = (
        scale * np.sin(t),
        3 * scale * np.ones((Nt)),
        scale * np.cos(t),
    )
    Nseg = 60.0
    dpts = int((Nt - 1) / Nseg)
    for ii in range(int((Nt - 1) / (2.0 * dpts))):
        jj = 2 * dpts * ii
        mlab.plot3d(
            cir_x[jj : jj + dpts],
            cir_y[jj : jj + dpts],
            cir_z[jj : jj + dpts],
            color=COLORS["gyrointegral"],
            tube_radius=TUBE_RADIUS / 2.0,
            opacity=0.3,
            figure=fig,
        )


def plot_gyrointegrated_vectors(
    fig: object, vgi: float, wgi: float, ugi: float, b: float = 0.9
) -> object:
    """Plot gyrointegrated vectors.

    Args:
        fig: Mayavi figure object.
        ugi, vgi, wgi: Vector components of the gyrointegrated (mean) velocity.
        b: Blending factor for head/tail positioning.

    Returns:
        Mayavi quiver3d object.
    """
    # Plot gyrointegrated vector
    obj = mlab.quiver3d(
        b * vgi,
        b * wgi,
        b * ugi,
        (1 - b) * vgi,
        (1 - b) * wgi,
        (1 - b) * ugi,
        line_width=1.25,
        scale_factor=1.0,
        color=COLORS["gyrointegral"],
        resolution=POINT_RESOLUTION,
        mode="cone",
        figure=fig,
    )
    Nt = 101
    t = np.linspace(0, 1, Nt)
    Nseg = 33.0
    dpts = int((Nt - 1) / Nseg)
    for ii in range(int((Nt - 1) / (2.0 * dpts))):
        jj = 2 * dpts * ii
        t1, t2 = t[jj], t[jj + dpts]
        #  mlab.plot3d([0, b*vgi], [0, b*wgi], [0, b*ugi], color=COLORS["gyrointegral"], tube_radius=TUBE_RADIUS)
        #  mlab.plot3d( cir_x[jj:jj+dpts], cir_y[jj:jj+dpts], cir_z[jj:jj+dpts], color=COLORS["gyrointegral"], tube_radius=TUBE_RADIUS/2., opacity=0.3 )
        mlab.plot3d(
            [t1 * vgi, t2 * vgi],
            [t1 * wgi, t2 * wgi],
            [t1 * ugi, t2 * ugi],
            color=COLORS["gyrointegral"],
            tube_radius=TUBE_RADIUS,
            figure=fig,
        )

    # Plot gyrointegrated vector projection
    Nt = 101
    t = np.linspace(0, wgi, Nt)
    Nseg = 33.0
    dpts = int((Nt - 1) / Nseg)
    for ii in range(int((Nt - 1) / (2.0 * dpts))):
        jj = 2 * dpts * ii
        mlab.plot3d(
            [vgi, vgi],
            [t[jj], t[jj + dpts]],
            [ugi, ugi],
            color=COLORS["gyrointegral"],
            tube_radius=TUBE_RADIUS / 2.0,
            opacity=0.3,
            figure=fig,
        )

    return obj


def plot_vparallel_vector(fig: object, wgi: float, b: float = 0.9) -> object:
    """Plot v// vector.

    Args:
        fig: Mayavi figure object.
        wgi: Vector component of the gyrointegrated (mean) velocity.
        b: Blending factor for head/tail positioning.

    Returns:
        Mayavi quiver3d object.
    """
    obj = mlab.quiver3d(
        0,
        b * wgi,
        0,
        0,
        (1 - b) * wgi,
        0,
        line_width=1.25,
        scale_factor=1.0,
        color=COLORS["vz"],
        resolution=POINT_RESOLUTION,
        mode="cone",
        figure=fig,
    )
    mlab.plot3d(
        [0, 0],
        [0, b * wgi],
        [0, 0],
        color=COLORS["vz"],
        tube_radius=TUBE_RADIUS,
        figure=fig,
    )
    return obj


def plot_disk_perpvect_points(
    fig: object, ugi: float, vgi: float, scale: float = 0.5
) -> object:
    """Plot 2D surface (disk), perpendicular vectors and points.

    Args:
        fig: Mayavi figure object.
        ugi, vgi: Vector components of the gyrointegrated (mean) velocity.
        scale: Scaling factor for vector magnitude.

    Returns:
        Mayavi quiver3d object.
    """
    # Plot 2D surface (disk) filling gyrointegrated positions (it helps to see in 3D) at particle position
    n = 361
    t = np.linspace(0, 2 * np.pi, n)
    c = 1.0
    x = c * scale * np.sin(t)
    # y = 3*scale*np.ones((n))
    y = np.zeros((n))
    z = c * scale * np.cos(t)
    triangles = [(0, i, i + 1) for i in range(1, n)]
    x = np.r_[0.0, x]
    y = np.r_[y[0], y]
    z = np.r_[0.0, z]
    t = np.r_[0.0, t]
    mlab.triangular_mesh(
        x, y, z, triangles, color=(0.5, 0.5, 0.5), opacity=0.65, figure=fig
    )

    # Plot points
    mlab.points3d(
        0,
        0,
        0,
        color=(0, 0, 0),
        scale_factor=sf,
        resolution=POINT_RESOLUTION,
        figure=fig,
    )
    # mlab.points3d(0,3*s,0, color=(0,0,0), scale_factor=sf, resolution=POINT_RESOLUTION )
    # mlab.points3d(vgi,wgi,ugi, color=COLORS["gyrointegral"], scale_factor=sf, resolution=POINT_RESOLUTION )

    # Plot gyrointegrated perpenticular velocity vector on the disk
    b2 = 0.75
    # vperp_gi_x,vperp_gi_y,vperp_gi_z,vperp_gi_vx,vperp_gi_vy,vperp_gi_vz = b2*vgi, 3*scale, b2*ugi, (1-b2)*vgi, 0, (1-b2)*ugi
    vperp_gi_x, vperp_gi_y, vperp_gi_z, vperp_gi_dx, vperp_gi_dy, vperp_gi_dz = (
        b2 * vgi,
        0,
        b2 * ugi,
        (1 - b2) * vgi,
        0,
        (1 - b2) * ugi,
    )
    obj = mlab.quiver3d(
        vperp_gi_x,
        vperp_gi_y,
        vperp_gi_z,
        vperp_gi_dx,
        vperp_gi_dy,
        vperp_gi_dz,
        line_width=2.5,
        scale_factor=1.0,
        color=COLORS["gyrointegral"],
        resolution=60,
        mode="cone",
        figure=fig,
    )
    mlab.plot3d(
        [0, vperp_gi_x],
        [vperp_gi_y, vperp_gi_y],
        [0, vperp_gi_z],
        color=COLORS["gyrointegral"],
        tube_radius=0.5 * TUBE_RADIUS,
        figure=fig,
    )
    return obj


def plot_gyrocenter_features(
    fig: object,
    u: np.ndarray,
    v: np.ndarray,
    k0: int,
    c0: float = 1.5,
    sf: float = 0.06,
    scale: float = 0.5,
) -> None:
    """Plot instantaneous gyrocenter-related features.

    Args:
        fig: Mayavi figure object.
        u, v: Vector components (arrays of equal length).
        k0: Index of the vector to highlight.
        c0: Scaling factor for the field line extension.
        sf: Scaling factor for points.
        scale: Scaling factor for vector magnitude.
    """
    Dx, Dy, Dz = 1.75 * scale, 0, -1.25 * scale
    rho = np.sqrt(Dx**2 + Dy**2 + Dz**2)

    # Straigth magnetic field line   OR   AXIS
    mlab.plot3d(
        [Dx, Dx],
        [Dy, Dy + 6 * scale],
        [Dz, Dz],
        color=COLORS["basis"],
        tube_radius=TUBE_AXIS,
        figure=fig,
    )
    mlab.plot3d(
        [Dx, Dx],
        [Dy, Dy - 3 * scale],
        [Dz, Dz],
        color=COLORS["basis"],
        tube_radius=TUBE_AXIS,
        figure=fig,
    )
    mlab.plot3d(
        [Dx, Dx - c0 * scale],
        [Dy, Dy],
        [Dz, Dz],
        color=COLORS["basis"],
        tube_radius=TUBE_AXIS,
        figure=fig,
    )
    mlab.plot3d(
        [Dx, Dx],
        [Dy, Dy],
        [Dz, Dz + c0 * scale],
        color=COLORS["basis"],
        tube_radius=TUBE_AXIS,
        figure=fig,
    )

    # Curved magnetic field line
    Nt = 801
    t = np.linspace(-3 * scale, 6 * scale, Nt)
    x = 1.5 * scale / (6.0 * scale) ** 2 * t**2
    y = t
    z = np.zeros((Nt))
    mlab.plot3d(
        Dx + x,
        Dy + y,
        Dz + z,
        color=COLORS["Bfield"],
        tube_radius=TUBE_RADIUS,
        figure=fig,
    )

    # Plot points
    mlab.points3d(
        Dx,
        Dy,
        Dz,
        color=COLORS["gc"],
        scale_factor=sf,
        resolution=POINT_RESOLUTION,
        figure=fig,
    )

    ox, oy, oz = 5 * scale, 0, -8.5 * scale
    dX, dY, dZ = Dx - ox, Dy - oy, Dz - oz
    b0 = 0.95
    b3 = 0.001
    # Plot spatial position of GC (red) and particles (black)
    c = 0.6
    obj = mlab.quiver3d(
        Dx - (1 - b0) * dX,
        Dy - (1 - b0) * dY,
        Dz - (1 - b0) * dZ,
        (1 - b0 - b3) * dX,
        (1 - b0 - b3) * dY,
        (1 - b0 - b3) * dZ,
        line_width=1.25,
        scale_factor=1.0,
        color=COLORS["gc"],
        resolution=POINT_RESOLUTION,
        mode="cone",
        figure=fig,
    )
    mlab.plot3d(
        [ox + c * dX, ox + b0 * dX],
        [oy + c * dY, oy + b0 * dY],
        [oz + c * dZ, oz + b0 * dZ],
        color=COLORS["gc"],
        tube_radius=TUBE_RADIUS,
        figure=fig,
    )
    c = 0.5
    obj = mlab.quiver3d(
        (1 - b0) * ox,
        (1 - b0) * oy,
        (1 - b0) * oz,
        -(1 - b0 - b3) * ox,
        -(1 - b0 - b3) * oy,
        -(1 - b0 - b3) * oz,
        line_width=1.25,
        scale_factor=1.0,
        color=COLORS["x"],
        resolution=POINT_RESOLUTION,
        mode="cone",
        figure=fig,
    )
    mlab.plot3d(
        [(1 - c) * ox, (1 - b0) * ox],
        [(1 - c) * oy, (1 - b0) * oy],
        [(1 - c) * oz, (1 - b0) * oz],
        color=COLORS["x"],
        tube_radius=TUBE_RADIUS,
        figure=fig,
    )

    b0 = 0.85

    # Plot gyroaveraged Larmor radius rho (blue)
    obj = mlab.quiver3d(
        (1 - b0) * Dx,
        (1 - b0) * Dy,
        (1 - b0) * Dz,
        -(1 - b0 - b3) * Dx,
        -(1 - b0 - b3) * Dy,
        -(1 - b0 - b3) * Dz,
        line_width=1.25,
        scale_factor=1.0,
        color=COLORS["rho"],
        resolution=POINT_RESOLUTION,
        mode="cone",
        figure=fig,
    )
    mlab.plot3d(
        [Dx, (1 - b0) * Dx],
        [Dy, (1 - b0) * Dy],
        [Dz, (1 - b0) * Dz],
        color=COLORS["rho"],
        tube_radius=TUBE_RADIUS,
        figure=fig,
    )

    # Plot instantaneous gyrocenter circle (cyan)
    ratio = rho / scale
    igc_x, igc_y, igc_z = ratio * u[k0], 0, ratio * v[k0]
    Nt = 361
    t = np.linspace(0, 2 * np.pi, Nt)
    cir_x, cir_y, cir_z = rho * np.sin(t), np.zeros((Nt)), rho * np.cos(t)
    mlab.plot3d(
        cir_x,
        cir_y,
        cir_z,
        color=COLORS["inst_gyrocenter"],
        tube_radius=TUBE_RADIUS,
        figure=fig,
    )

    # Plot instantaneous gyrocenter vector
    obj = mlab.quiver3d(
        b0 * igc_x,
        b0 * igc_y,
        b0 * igc_z,
        (1 - b0 - b3) * igc_x,
        (1 - b0 - b3) * igc_y,
        (1 - b0 - b3) * igc_z,
        line_width=1.25,
        scale_factor=1.0,
        color=COLORS["inst_gyrocenter"],
        resolution=POINT_RESOLUTION,
        mode="cone",
        figure=fig,
    )
    mlab.plot3d(
        [0, b0 * igc_x],
        [0, b0 * igc_y],
        [0, b0 * igc_z],
        color=COLORS["inst_gyrocenter"],
        tube_radius=TUBE_RADIUS,
        figure=fig,
    )

    # Plot instantaneous gyrocenter position
    mlab.points3d(
        igc_x,
        igc_y,
        igc_z,
        color=COLORS["inst_gyrocenter"],
        scale_factor=sf,
        resolution=POINT_RESOLUTION,
        figure=fig,
    )


# ======================
# Main Execution
# ======================
if __name__ == "__main__":
    start_time = datetime.datetime.utcnow()
    logger.info("<> START <>")

    #    mlab.init_notebook('x3d')
    #    mlab.options.offscreen = True
    # mlab.options.backend = "wx" # 'qt' or 'wx'

    b = 0.9  # Blending factor for head/tail positioning.
    c0 = 1.5  # Scaling factor for the field line extension.
    sf = 0.06  # Scaling factor for points.

    n_vec = 10  # Number of vectors to generate.
    scale = 0.5  # Scaling factor for vector magnitude.
    # Vector components (arrays of equal length):
    u, v, w = generate_demo_vectors(n_vec=n_vec, scale=scale)

    # Compute vector components of the gyrointegrated (mean) velocity:
    ugi, vgi, wgi = np.mean(u), np.mean(v), np.mean(w)

    k0 = len(u) // 2  # int(Nvec / 2.0) # Index of the vector to highlight.
    indices = [k for k in range(len(u)) if k != k0]  # Indices of vectors to plot.

    fig = mlab.figure(bgcolor=BACKGROUND_COLOR, size=FIG_SIZE)

    # Plot particle vectors
    obj = plot_particle_vectors(fig=fig, u=u, v=v, w=w, indices=indices, b=b)

    #     # Plot selected particle vector
    obj = plot_selected_particle_vectors(fig=fig, u=u, v=v, w=w, k0=k0, b=b)

    #     # Plot selected particle vector projection
    obj = plot_selected_particle_vectors_projection(
        fig=fig, u=u, v=v, w=w, k0=k0, wgi=wgi
    )

    # Straigth magnetic field line   OR   AXIS
    plot_magnetic_field(fig=fig, c0=c0, scale=scale)

    # Gyrointegral circles at particle position
    plot_gyrointegral_circles(fig=fig, scale=scale)

    # Plot gyrointegrated vectors
    obj = plot_gyrointegrated_vectors(fig=fig, vgi=vgi, wgi=wgi, ugi=ugi, b=b)

    # Plot v// vector
    obj = plot_vparallel_vector(fig=fig, wgi=wgi, b=b)

    # Plot 2D surface (disk), perpendicular vectors and points
    plot_disk_perpvect_points(fig=fig, ugi=ugi, vgi=vgi, scale=scale)

    ####################################################

    ifshow_gc = True
    if ifshow_gc:
        # Plot instantaneous gyrocenter-related features
        plot_gyrocenter_features(fig=fig, u=u, v=v, k0=k0, c0=c0, sf=sf, scale=scale)

    mlab.draw()

    vv = mlab.view()
    # mlab.view(vv[0],vv[1]+45)
    mlab.view(distance=5)
    mlab.view(azimuth=vv[0] - 10)

    mlab.draw()

    # SAVE FIGURE
    fsize = (200, 150)
    # Ensure directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Count existing figures
    existing = sorted(glob.glob(os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}*.png")))
    n_figures = len(existing)
    # Build filename
    filename = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}{str(n_figures).zfill(3)}.png")
    # Save figure
    mlab.savefig(filename, size=fsize, figure=fig)
    logger.info(f"   Figure {filename} saved.")
    mlab.show()
    # Clear figure instead of closing engine
    mlab.clf(fig)  # clears content but keeps window alive
    # If you really want to close, just do: mlab.close(fig)

    logger.info("<> script MAYAVI done <>")
