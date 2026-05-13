"""Geometry and material lookup helpers shared by MC and circuit code."""


from __future__ import annotations

from typing import Tuple

from config import TransportConfig

def is_boundary(cfg: TransportConfig, x: int, y: int, z: int) -> bool:

    g = cfg.geom
    return (x == 0 or x == g.nx - 1 or y == 0 or y == g.ny - 1 or z == 0 or z == g.nz - 1)

def is_in_slab(cfg: TransportConfig, x: int) -> bool:
    g = cfg.geom
    return g.slab_x0 <= x <= g.slab_x1

def is_in_duct(cfg: TransportConfig, y: int, z: int) -> bool:
    g = cfg.geom
    return (g.duct_y0 <= y <= g.duct_y1) and (g.duct_z0 <= z <= g.duct_z1)

def is_in_roi(cfg: TransportConfig, x: int, y: int, z: int) -> bool:
    g = cfg.geom
    return (x == g.detector_x) and (g.roi_y0 <= y <= g.roi_y1) and (g.roi_z0 <= z <= g.roi_z1)

def material_at(cfg: TransportConfig, x: int, y: int, z: int) -> str:

    if is_in_slab(cfg, x):
        if is_in_duct(cfg, y, z):
            return "air"
        return "tungsten"
    return "concrete"

def clamp_pos(cfg: TransportConfig, x: int, y: int, z: int) -> Tuple[int, int, int]:
    g = cfg.geom
    x = max(0, min(g.nx - 1, x))
    y = max(0, min(g.ny - 1, y))
    z = max(0, min(g.nz - 1, z))
    return x, y, z
