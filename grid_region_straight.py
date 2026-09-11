import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree, Voronoi, voronoi_plot_2d
from skimage import measure

# ==========================================
# 1. Generate Synthetic 10-Class Grid Data
# ==========================================
np.random.seed(42)

# Define grid: 55 x 55 = 3,025 points
nx, ny = 55, 55
x = np.linspace(0, 10, nx)
y = np.linspace(0, 10, ny)
X, Y = np.meshgrid(x, y)
points = np.column_stack([X.ravel(), Y.ravel()])

# Seed 10 distinct region centers to ensure continuous categories
num_classes = 10
centers = np.random.uniform(1.0, 9.0, size=(num_classes, 2))
tree = KDTree(centers)

# Assign each grid point to nearest center (generates 10 continuous regions)
_, labels_flat = tree.query(points)
grid_labels = labels_flat.reshape((ny, nx))

# ==========================================
# 2. Plotting Comparisons Side-by-Side
# ==========================================
fig, axs = plt.subplots(2, 2, figsize=(14, 12))
cmap = mpl.colormaps["tab10"]

# Subplot 1: Raw Categorical Grid
ax1 = axs[0, 0]
im1 = ax1.imshow(
    grid_labels, origin="lower", extent=[0, 10, 0, 10], cmap=cmap, alpha=0.75
)
ax1.scatter(X, Y, c="black", s=3, alpha=0.3, label="Measured Points (~3k)")
ax1.set_title("1. Raw Grid Points & Regions", fontsize=12, fontweight="bold")
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.legend(loc="upper right", markerscale=3)

# Subplot 2: One-Hot Marching Squares Contours
ax2 = axs[0, 1]
ax2.imshow(grid_labels, origin="lower", extent=[0, 10, 0, 10], cmap=cmap, alpha=0.25)
for cat in range(num_classes):
    binary_mask = (grid_labels == cat).astype(float)
    # skimage find_contours works in array coordinate space (0..ny-1, 0..nx-1)
    contours = measure.find_contours(binary_mask, level=0.5)
    for c in contours:
        # Scale back to spatial coordinate domain [0, 10]
        cx = c[:, 1] * (10 / (nx - 1))
        cy = c[:, 0] * (10 / (ny - 1))
        ax2.plot(cx, cy, color="black", linewidth=1.8)
ax2.set_title("2. One-Hot Marching Squares Contours", fontsize=12, fontweight="bold")
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)

# Subplot 3: Exact Orthogonal Grid Step Boundaries
ax3 = axs[1, 0]
ax3.imshow(grid_labels, origin="lower", extent=[0, 10, 0, 10], cmap=cmap, alpha=0.3)
dx = 10 / (nx - 1)
dy = 10 / (ny - 1)

# Horizontal interface segments
for i in range(ny - 1):
    for j in range(nx):
        if grid_labels[i, j] != grid_labels[i + 1, j]:
            y_edge = (y[i] + y[i + 1]) / 2
            ax3.plot(
                [x[j] - dx / 2, x[j] + dx / 2],
                [y_edge, y_edge],
                color="crimson",
                linewidth=1.2,
            )

# Vertical interface segments
for i in range(ny):
    for j in range(nx - 1):
        if grid_labels[i, j] != grid_labels[i, j + 1]:
            x_edge = (x[j] + x[j + 1]) / 2
            ax3.plot(
                [x_edge, x_edge],
                [y[i] - dy / 2, y[i] + dy / 2],
                color="crimson",
                linewidth=1.2,
            )

ax3.set_title(
    "3. Exact Cell Step Boundaries (Zero Interpolation)", fontsize=12, fontweight="bold"
)
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 10)

# Subplot 4: 1-NN Voronoi Decision Boundary
ax4 = axs[1, 1]
ax4.scatter(points[:, 0], points[:, 1], c=labels_flat, cmap=cmap, s=12, alpha=0.6)
# Seed centers show the true partition lines
vor = Voronoi(centers)
voronoi_plot_2d(
    vor,
    ax=ax4,
    show_points=False,
    show_vertices=False,
    line_colors="navy",
    line_width=1.8,
    line_style="solid",
)
ax4.set_title(
    "4. Fitted Boundary (1-NN / Seed Partition)", fontsize=12, fontweight="bold"
)
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 10)

plt.tight_layout()
plt.savefig("grid_comparison_straight.png", dpi=300, bbox_inches="tight")
plt.show()
