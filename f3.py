"""Generate the transport-semantics schematic figure."""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib import rcParams

OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
rcParams['mathtext.fontset'] = 'stixsans'
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

BG = '#ffffff'
TEXT = '#000000'
EDGE = '#000000'
GUIDE = '#A0A0A0'
HEAD = '#000000'

WFC = '#FFFFFF'
BFC = '#EBF4FA'
GFC = '#F4F4F4'

fig, ax = plt.subplots(figsize=(16.0, 6.2), dpi=400)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

ys = {'xyz':0.83, 'pred':0.70, 'status':0.57, 'ledger':0.44, 'coin':0.31, 'gdir':0.18}
labels = {
    'xyz': r'$|x,y,z\rangle$',
    'pred': r'$|\mathrm{pred}\rangle_{\mathrm{anc}}$',
    'status': r'$|\sigma\rangle_{\mathrm{status}}$',
    'ledger': r'$|\mathrm{ledger}\rangle$',
    'coin': r'$|\mathrm{coin}\rangle$',
    'gdir': r'$|g, \mathbf{d}\rangle$'
}

def draw_lanes(ax, ys, labels, x_left=0.14, x_right=0.965, label_x=0.13):
    for k, y in ys.items():
        ax.plot([x_left, x_right], [y, y], color=EDGE, lw=1.2, zorder=1)
        ax.text(label_x, y, labels[k], ha='right', va='center', fontsize=17, color=TEXT)

def draw_stage_titles(ax, stages, y=0.94, sep_y0=0.08, sep_y1=0.90):
    for x0, x1, title in stages:
        ax.text((x0+x1)/2, y, title, ha='center', va='center', fontsize=14, fontweight='bold', color=HEAD)
    for x in [stages[i][1]+0.01 for i in range(len(stages)-1)]:
        ax.plot([x, x], [sep_y0, sep_y1], color=GUIDE, lw=1.0, ls='--', zorder=0)

def span_box(ax, xc, y_top, y_bot, w, label, fc, fs=13.0, lw=1.2):
    ax.add_patch(FancyBboxPatch((xc-w/2, y_bot), w, y_top-y_bot,
                                boxstyle='square,pad=0.0',
                                facecolor=fc, edgecolor=EDGE, linewidth=lw, zorder=4))
    ax.text(xc, (y_top+y_bot)/2, label, ha='center', va='center', fontsize=fs, color=TEXT, zorder=5)

def gate_box(ax, xc, yc, w, h, label, fc, fs=13.0, lw=1.2):
    ax.add_patch(FancyBboxPatch((xc-w/2, yc-h/2), w, h,
                                boxstyle='square,pad=0.0',
                                facecolor=fc, edgecolor=EDGE, linewidth=lw, zorder=4))
    ax.text(xc, yc, label, ha='center', va='center', fontsize=fs, color=TEXT, zorder=5)

def control_column(ax, xc, ys, controls, target, lw=1.2):
    yvals = [ys[k] for k in controls] + [ys[target]]
    ax.plot([xc, xc], [min(yvals), max(yvals)], color=EDGE, lw=lw, zorder=3)
    for k in controls:
        ax.scatter([xc], [ys[k]], s=45, c=EDGE, marker='o', zorder=6, edgecolors='none')

draw_lanes(ax, ys, labels)
stages = [
    (0.16,0.40,'STEP-START'),
    (0.44,0.67,'INTERACTION / ABSORB'),
    (0.71,0.80,'SCATTER / MOVE'),
    (0.84,0.96,'UNCOMPUTE')
]
draw_stage_titles(ax, stages)

span_box(ax, 0.17, ys['xyz']+0.038, ys['pred']-0.038, 0.040, r'$U_{\mathrm{pred}}^{\mathrm{bd}}$', GFC, fs=13)
span_box(ax, 0.23, ys['xyz']+0.038, ys['pred']-0.038, 0.040, r'$U_{\mathrm{pred}}^{\mathrm{roi}}$', GFC, fs=13)
control_column(ax, 0.27, ys, ['pred'], 'status')
gate_box(ax, 0.27, ys['status'], 0.060, 0.070, r'$W_{\mathrm{term}}$', WFC, fs=13)
span_box(ax, 0.335, ys['status']+0.038, ys['ledger']-0.038, 0.040, 'Alive\nGuard', GFC, fs=12)
span_box(ax, 0.38, ys['xyz']+0.038, ys['pred']-0.038, 0.040, r'$U_{\mathrm{pred}}^{\mathrm{mat}}$', GFC, fs=13)

control_column(ax, 0.45, ys, ['pred'], 'coin')
gate_box(ax, 0.45, ys['coin'], 0.065, 0.072, r'$R_y(\theta_{\mathrm{int}})$', BFC, fs=14)
span_box(ax, 0.52, ys['status']+0.038, ys['ledger']-0.038, 0.080, 'Interaction\nLedger', GFC, fs=12)

control_column(ax, 0.59, ys, ['pred'], 'coin')
gate_box(ax, 0.59, ys['coin'], 0.065, 0.072, r'$R_y(\theta_{\mathrm{abs}})$', BFC, fs=14)
control_column(ax, 0.64, ys, ['coin'], 'status')
gate_box(ax, 0.64, ys['status'], 0.050, 0.068, r'$W_{\mathrm{abs}}$', WFC, fs=13)

control_column(ax, 0.73, ys, ['coin'], 'gdir')
gate_box(ax, 0.73, ys['gdir'], 0.065, 0.100, r'$U_{\mathrm{scat}}$', BFC, fs=14)
gate_box(ax, 0.76, ys['xyz'], 0.060, 0.100, r'$U_{\mathrm{move}}$', BFC, fs=14)

span_box(ax, 0.87, ys['xyz']+0.038, ys['pred']-0.038, 0.080, r'$U_{\mathrm{pred}}^{\dagger}$', GFC, fs=14)
span_box(ax, 0.91, ys['ledger']+0.038, ys['ledger']-0.038, 0.060, r'$U_{\mathrm{guard}}^{\dagger}$', GFC, fs=14)

leg_y = 0.045
def legend_chip(x, color, text):
    ax.add_patch(Rectangle((x, leg_y - 0.008), 0.020, 0.016,
                           facecolor=color, edgecolor=EDGE, linewidth=1.0,
                           transform=ax.transAxes, clip_on=False, zorder=8))
    ax.text(x + 0.028, leg_y, text, ha='left', va='center', fontsize=12, color=TEXT,
            transform=ax.transAxes, clip_on=False, zorder=8)

legend_chip(0.20, GFC, 'Query / Guard / Uncompute')
legend_chip(0.45, BFC, 'Unitary Evolution (Coin/Scatter/Move)')
legend_chip(0.75, WFC, 'Status Terminal Write')

plt.tight_layout(pad=0.18)
for ext in ['png', 'pdf']:
    fig.savefig(os.path.join(OUT_DIR, f'Fig2_4_PRL_style.{ext}'), dpi=400, bbox_inches='tight', facecolor=BG)
plt.close(fig)
print(f"Fig 2.4 successfully saved to {OUT_DIR}/")
