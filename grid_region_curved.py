import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splev, splprep
from scipy.ndimage import gaussian_filter
from skimage import measure

# ==========================================================
# 1. Generate Synthetic 10-Class Grid with Curved Boundaries
# ==========================================================
np.random.seed(101)

# Grid setup: 55 x 55 = 3,025 points
nx, ny = 55, 55
x = np.linspace(0, 10, nx)
y = np.linspace(0, 10, ny)
X, Y = np.meshgrid(x, y)

num_classes = 10

# Generate smooth, organic potential fields using multi-channel Gaussian noise
potentials = np.zeros((num_classes, ny, nx))
for i in range(num_classes):
    # Random sparse peaks
    field = np.zeros((ny, nx))
    num_peaks = np.random.randint(2, 5)
    px = np.random.randint(5, nx - 5, size=num_peaks)
    py = np.random.randint(5, ny - 5, size=num_peaks)
    field[py, px] = np.random.uniform(5.0, 15.0, size=num_peaks)

    # Smooth with Gaussian filter to induce continuous non-linear curved basins
    potentials[i] = gaussian_filter(field, sigma=7.0)

# Class assignment via argmax of continuous potentials (guarantees continuous curved boundaries)
grid_labels = np.argmax(potentials, axis=0)

# ==========================================================
# 2. Side-by-Side Visualization Comparison
# ==========================================
fig, axs = plt.subplots(2, 2, figsize=(14, 12))
cmap = mpl.colormaps["tab10"]
dx = 10.0 / (nx - 1)
dy = 10.0 / (ny - 1)

# --- Panel 1: Raw Categorical Grid ---
ax1 = axs[0, 0]
ax1.imshow(grid_labels, origin="lower", extent=[0, 10, 0, 10], cmap=cmap, alpha=0.75)
ax1.scatter(X, Y, c="black", s=2, alpha=0.25, label="Measured Points (3,025)")
ax1.set_title("1. Raw Sampled Points & Curved Regions", fontsize=12, fontweight="bold")
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.legend(loc="upper right", markerscale=3)

# --- Panel 2: One-Hot Marching Squares ---
ax2 = axs[0, 1]
ax2.imshow(grid_labels, origin="lower", extent=[0, 10, 0, 10], cmap=cmap, alpha=0.3)
for cat in range(num_classes):
    binary_mask = (grid_labels == cat).astype(float)
    contours = measure.find_contours(binary_mask, level=0.5)
    for c in contours:
        cx = c[:, 1] * dx
        cy = c[:, 0] * dy
        ax2.plot(cx, cy, color="black", linewidth=1.8)
ax2.set_title("2. One-Hot Marching Squares Contours", fontsize=12, fontweight="bold")
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)

# --- Panel 3: Exact Orthogonal Cell Boundaries ---
ax3 = axs[1, 0]
ax3.imshow(grid_labels, origin="lower", extent=[0, 10, 0, 10], cmap=cmap, alpha=0.3)

# Horizontal interface segments
for i in range(ny - 1):
    for j in range(nx):
        if grid_labels[i, j] != grid_labels[i + 1, j]:
            y_edge = (y[i] + y[i + 1]) / 2.0
            ax3.plot(
                [x[j] - dx / 2.0, x[j] + dx / 2.0],
                [y_edge, y_edge],
                color="crimson",
                linewidth=1.2,
            )

# Vertical interface segments
for i in range(ny):
    for j in range(nx - 1):
        if grid_labels[i, j] != grid_labels[i, j + 1]:
            x_edge = (x[j] + x[j + 1]) / 2.0
            ax3.plot(
                [x_edge, x_edge],
                [y[i] - dy / 2.0, y[i] + dy / 2.0],
                color="crimson",
                linewidth=1.2,
            )

ax3.set_title(
    "3. Exact Discrete Step Edges (No Interpolation)", fontsize=12, fontweight="bold"
)
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 10)

# --- Panel 4: Spline-Smoothed Vector Curves ---
ax4 = axs[1, 1]
ax4.imshow(grid_labels, origin="lower", extent=[0, 10, 0, 10], cmap=cmap, alpha=0.3)
for cat in range(num_classes):
    binary_mask = (grid_labels == cat).astype(float)
    contours = measure.find_contours(binary_mask, level=0.5)
    for c in contours:
        cx = c[:, 1] * dx
        cy = c[:, 0] * dy
        # Fit cubic B-spline if enough vertices exist
        if len(c) > 6:
            try:
                # Remove duplicate closing point for periodic spline fitting if closed
                is_closed = np.allclose(c[0], c[-1])
                tck, u = splprep([cx, cy], s=2.0, per=is_closed, k=3)
                u_new = np.linspace(0, 1, 300)
                smooth_x, smooth_y = splev(u_new, tck)
                ax4.plot(smooth_x, smooth_y, color="navy", linewidth=2.0)
            except Exception:
                ax4.plot(cx, cy, color="navy", linewidth=1.8)
        else:
            ax4.plot(cx, cy, color="navy", linewidth=1.8)

ax4.set_title(
    "4. Spline-Smoothed Vector Contours (C2 Smooth)", fontsize=12, fontweight="bold"
)
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 10)

plt.tight_layout()
plt.savefig("grid_comparison_curved.png", dpi=300, bbox_inches="tight")
plt.show()
