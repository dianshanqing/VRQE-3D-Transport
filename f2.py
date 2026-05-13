"""Generate the predicate-layer schematic figure."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
from matplotlib import rcParams
OUT_DIR = 'figures'
os.makedirs(OUT_DIR, exist_ok=True)
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
rcParams['mathtext.fontset'] = 'stixsans'
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
BG = '#ffffff'
TEXT = '#000000'
EDGE = '#000000'
MUTED = '#404040'
HEAD = '#000000'
PANEL_EDGE = '#707070'
BOX_FC = '#ffffff'
BOX_FC_SHADE = '#f4f4f4'
fig, ax = plt.subplots(figsize=(15.8, 6.6), dpi=400)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

def add_panel(x, y, w, h, title):
    ax.add_patch(Rectangle((x, y), w, h, facecolor='none', edgecolor=PANEL_EDGE, linewidth=1.2, linestyle='--'))
    ax.add_patch(Rectangle((x + w / 2 - 0.08, y + h - 0.005), 0.16, 0.04, facecolor=BG, edgecolor='none', zorder=2))
    ax.text(x + w / 2, y + h + 0.015, title, ha='center', va='center', fontsize=14, fontweight='bold', color=HEAD, zorder=3)

def add_box(x, y, w, h, fc, title, sub=None, fs=12.5, subfs=10.5, lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.01,rounding_size=0.005', facecolor=fc, edgecolor=EDGE, linewidth=lw, zorder=3))
    if sub:
        ax.text(x + w / 2, y + h * 0.65, title, ha='center', va='center', fontsize=fs, fontweight='bold', color=TEXT, zorder=4)
        ax.text(x + w / 2, y + h * 0.28, sub, ha='center', va='center', fontsize=subfs, color=MUTED, zorder=4)
    else:
        ax.text(x + w / 2, y + h * 0.5, title, ha='center', va='center', fontsize=fs, fontweight='bold', color=TEXT, zorder=4)

def arrow(x1, y1, x2, y2, ms=14, lw=1.2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=ms, linewidth=lw, color=EDGE, zorder=2))
add_panel(0.035, 0.1, 0.195, 0.78, 'INPUTS')
add_panel(0.265, 0.1, 0.255, 0.78, 'DOMAIN-PREDICATE LAYER')
add_panel(0.555, 0.1, 0.255, 0.78, 'SINGLE-STEP KERNEL')
add_panel(0.845, 0.1, 0.12, 0.78, 'OUTPUTS')
add_box(0.055, 0.695, 0.155, 0.115, BOX_FC, 'Main registers', '$|x, y, z\\rangle \\otimes |g, \\mathbf{d}\\rangle$')
add_box(0.055, 0.525, 0.155, 0.115, BOX_FC, 'Geometry constants', 'diagnostic message')
add_box(0.055, 0.355, 0.155, 0.115, BOX_FC, 'Step context', 'alive-at-step-start')
add_box(0.055, 0.165, 0.155, 0.11, BOX_FC, 'Interface rule', 'local predicates only')
add_box(0.295, 0.7, 0.195, 0.1, BOX_FC_SHADE, 'Boundary predicate', '$p_{\\mathrm{bd}}(x, y, z)$')
add_box(0.295, 0.565, 0.195, 0.1, BOX_FC_SHADE, 'ROI predicate', '$p_{\\mathrm{roi}}(x, y, z)$')
add_box(0.295, 0.43, 0.195, 0.1, BOX_FC_SHADE, 'Slab predicate', '$p_{\\mathrm{slab}}(x)$')
add_box(0.295, 0.295, 0.195, 0.1, BOX_FC_SHADE, 'Duct predicate', '$p_{\\mathrm{duct}}(y, z)$')
add_box(0.295, 0.145, 0.195, 0.105, BOX_FC, 'Material derivation', 'diagnostic message')
arrow(0.392, 0.43, 0.392, 0.395, ms=10, lw=1.0)
arrow(0.392, 0.295, 0.392, 0.25, ms=10, lw=1.0)
add_box(0.585, 0.695, 0.195, 0.11, BOX_FC, 'Step-start terminal write', 'boundary $\\rightarrow$ status=11\nROI $\\rightarrow$ status=01')
add_box(0.585, 0.55, 0.195, 0.11, BOX_FC, 'Guard / ledger build', 'alive condition guard')
add_box(0.585, 0.405, 0.195, 0.11, BOX_FC, 'Interaction module', '$R_y(\\theta_{\\mathrm{int}})$ conditional')
add_box(0.585, 0.26, 0.195, 0.11, BOX_FC, 'Absorb module', 'absorb $\\rightarrow$ status=10')
add_box(0.585, 0.115, 0.195, 0.11, BOX_FC, 'Scatter & Move', '$U_{\\mathrm{scat}} \\cdot U_{\\mathrm{move}} \\cdot U_{\\mathrm{pred}}^\\dagger$')
for y1, y2 in [(0.695, 0.66), (0.55, 0.515), (0.405, 0.37), (0.26, 0.225)]:
    arrow(0.6825, y1, 0.6825, y2, ms=12, lw=1.2)
add_box(0.865, 0.615, 0.08, 0.13, BOX_FC, 'Registers', '$|x,y,z\\rangle, |g, \\mathbf{d}\\rangle$')
add_box(0.865, 0.405, 0.08, 0.13, BOX_FC_SHADE, 'Status', '$|\\sigma\\rangle \\in \\{00,01,10,11\\}$')
add_box(0.865, 0.195, 0.08, 0.13, BOX_FC, 'Ancilla', 'Uncomputed')
arrow(0.23, 0.49, 0.265, 0.49, ms=16, lw=1.5)
arrow(0.52, 0.49, 0.555, 0.49, ms=16, lw=1.5)
arrow(0.81, 0.49, 0.845, 0.49, ms=16, lw=1.5)
plt.tight_layout(pad=0.15)
for ext in ['png', 'pdf']:
    plt.savefig(os.path.join(OUT_DIR, f'Fig2_2_PRL_style.{ext}'), dpi=400, bbox_inches='tight', facecolor=BG)
plt.close(fig)
print(f'Fig 2.2 successfully saved to {OUT_DIR}/')
