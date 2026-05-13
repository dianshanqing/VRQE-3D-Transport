"""Generate a publication-style geometry figure."""

import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rcParams
from matplotlib.patches import Patch

OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
rcParams['mathtext.fontset'] = 'stixsans'
rcParams['font.size'] = 11

Nx, Ny, Nz = 16, 8, 8
slab_x = [8, 9]
duct_y = [3, 4]
duct_z = [3, 4]
roi_x = 12
source = (1, 3, 3)

fig = plt.figure(figsize=(6, 5), dpi=300)
ax = fig.add_subplot(111, projection='3d')

for x in slab_x:
    for y in range(Ny):
        for z in range(Nz):
            if (y in duct_y) and (z in duct_z):
                continue
            ax.bar3d(x, y, z, 1, 1, 1,
                     color='#D3D3D3',
                     edgecolor='black',
                     linewidth=0.4,
                     alpha=0.3)

for x in slab_x:
    for y in duct_y:
        for z in duct_z:

            ax.bar3d(x, y, z, 1, 1, 1,
                     color=(0, 0, 0, 0),
                     edgecolor='#004488',
                     linewidth=0.5,
                     shade=False)

for y in duct_y:
    for z in duct_z:
        ax.bar3d(roi_x, y, z, 1, 1, 1,
                 color='#B22222',
                 edgecolor='black',
                 linewidth=0.5,
                 alpha=0.8)

ax.bar3d(*source, 1, 1, 1,
         color='#000000',
         edgecolor='black',
         linewidth=0.5)

ax.set_xlim(0, Nx)
ax.set_ylim(0, Ny)
ax.set_zlim(0, Nz)

ax.set_xlabel('$x$', labelpad=5, fontsize=14)
ax.set_ylabel('$y$', labelpad=5, fontsize=14)
ax.set_zlabel('$z$', labelpad=5, fontsize=14)

ax.grid(False)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('black')
ax.yaxis.pane.set_edgecolor('black')
ax.zaxis.pane.set_edgecolor('black')

ax.plot([0, 0], [0, 0], [0, Nz], color='black', linewidth=0.8, zorder=0)

ax.view_init(elev=25, azim=-60)

legend_elements = [
    Patch(facecolor='#000000', edgecolor='black', label='Source'),
    Patch(facecolor='#D3D3D3', edgecolor='black', alpha=0.5, label='Dense slab'),
    Patch(facecolor='none', edgecolor='#004488', linewidth=1.5, label='Air duct'),
    Patch(facecolor='#B22222', edgecolor='black', alpha=0.8, label='ROI (detect)')
]

ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.05, 1.05), frameon=False, fontsize=11)

plt.tight_layout()

plt.savefig(os.path.join(OUT_DIR, "Fig2_1_geometry_PRL.pdf"), bbox_inches='tight')
plt.savefig(os.path.join(OUT_DIR, "Fig2_1_geometry_PRL.png"), dpi=400, bbox_inches='tight')
plt.close(fig)
print(f"Fig 2.1 successfully saved to {OUT_DIR}/")
