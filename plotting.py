"""Plotting helpers for benchmark outputs."""


from __future__ import annotations

import os
from typing import List

import numpy as np

from geometry import material_at

def _mpl_import():

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt

def save_mc_plots(cfg, res, outdir: str = "outputs") -> List[str]:

    os.makedirs(outdir, exist_ok=True)
    plt = _mpl_import()

    saved: List[str] = []

    labels = ["Detect", "Absorb", "Boundary", "Survive"]
    vals = [res.p_detect, res.p_absorb, res.p_boundary, res.p_survive]
    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(111)
    ax.bar(labels, vals)
    ax.set_ylim(0.0, max(vals) * 1.15 + 1e-9)
    ax.set_ylabel("Probability")
    ax.set_title("Phase 1: Termination probabilities")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    p1 = os.path.join(outdir, "mc_termination_probs.png")
    fig.savefig(p1, dpi=160)
    plt.close(fig)
    saved.append(p1)

    G = cfg.groups.n_groups

    b = cfg.groups.boundaries

    xlabels = [f"g={g}\n[{b[g+1]:.2f},{b[g]:.2f}]" for g in range(G)]
    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(111)
    ax.bar(list(range(G)), res.energy_hist_detect)
    ax.set_xticks(list(range(G)))
    ax.set_xticklabels(xlabels)
    ax.set_ylim(0.0, max(0.05, float(res.energy_hist_detect.max()) * 1.15))
    ax.set_ylabel("Fraction (conditional on detect)")
    ax.set_title("Phase 1: Detector energy composition")
    for i, v in enumerate(res.energy_hist_detect.tolist()):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    p2 = os.path.join(outdir, "mc_detector_energy_hist.png")
    fig.savefig(p2, dpi=160)
    plt.close(fig)
    saved.append(p2)

    mat_code = {"concrete": 0, "tungsten": 1, "air": 2}

    x_slice = (cfg.geom.slab_x0 + cfg.geom.slab_x1) // 2
    yz = np.zeros((cfg.geom.ny, cfg.geom.nz), dtype=int)
    for y in range(cfg.geom.ny):
        for z in range(cfg.geom.nz):
            yz[y, z] = mat_code.get(material_at(cfg, x_slice, y, z), 0)

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    im = ax.imshow(yz.T, origin="lower", aspect="auto")
    ax.set_title(f"Geometry slice (Y-Z) at x={x_slice}")
    ax.set_xlabel("y")
    ax.set_ylabel("z")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    p3 = os.path.join(outdir, "geometry_yz_slice.png")
    fig.savefig(p3, dpi=160)
    plt.close(fig)
    saved.append(p3)

    z_slice = cfg.source.z0
    xy = np.zeros((cfg.geom.nx, cfg.geom.ny), dtype=int)
    for x in range(cfg.geom.nx):
        for y in range(cfg.geom.ny):
            xy[x, y] = mat_code.get(material_at(cfg, x, y, z_slice), 0)

    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(111)
    im = ax.imshow(xy.T, origin="lower", aspect="auto")
    ax.set_title(f"Geometry slice (X-Y) at z={z_slice}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    p4 = os.path.join(outdir, "geometry_xy_slice.png")
    fig.savefig(p4, dpi=160)
    plt.close(fig)
    saved.append(p4)

    return saved
