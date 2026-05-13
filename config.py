"""Configuration objects and default transport benchmark parameters."""


from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class Geometry3D:
    nx: int
    ny: int
    nz: int

    slab_x0: int
    slab_x1: int

    duct_y0: int
    duct_y1: int
    duct_z0: int
    duct_z1: int

    detector_x: int
    roi_y0: int
    roi_y1: int
    roi_z0: int
    roi_z1: int

@dataclass(frozen=True)
class EnergyGroups:

    n_groups: int
    boundaries: Tuple[float, ...]

@dataclass(frozen=True)
class MaterialParams:

    p_int: Tuple[float, ...]
    p_abs: Tuple[float, ...]
    q_down: Tuple[float, ...]

@dataclass(frozen=True)
class Materials:
    concrete: MaterialParams
    tungsten: MaterialParams
    air: MaterialParams

@dataclass(frozen=True)
class Source:
    x0: int
    y0: int
    z0: int
    g0: int

    dir0: int = 0

@dataclass(frozen=True)
class TransportConfig:
    geom: Geometry3D
    groups: EnergyGroups
    mats: Materials
    source: Source

def paper1_config() -> TransportConfig:

    geom = Geometry3D(
        nx=16, ny=8, nz=8,

        slab_x0=8, slab_x1=9,

        duct_y0=3, duct_y1=4,
        duct_z0=3, duct_z1=4,

        detector_x=12,
        roi_y0=3, roi_y1=4,
        roi_z0=3, roi_z1=4,
    )

    groups = EnergyGroups(
        n_groups=2,

        boundaries=(0.70, 0.30, 0.05),
    )

    mats = Materials(
        concrete=MaterialParams(
            p_int=(0.10, 0.12),
            p_abs=(0.10, 0.15),
            q_down=(0.0, 0.0),
        ),
        tungsten=MaterialParams(
            p_int=(0.28, 0.32),
            p_abs=(0.45, 0.55),
            q_down=(0.0, 0.0),
        ),
        air=MaterialParams(
            p_int=(0.02, 0.02),
            p_abs=(0.02, 0.03),
            q_down=(0.0, 0.0),
        ),
    )

    source = Source(
        x0=1,
        y0=3,
        z0=3,
        g0=0,
        dir0=0,
    )

    cfg = TransportConfig(geom=geom, groups=groups, mats=mats, source=source)
    validate_config(cfg)
    return cfg

def paper_scale_config() -> TransportConfig:

    geom = Geometry3D(
        nx=32, ny=16, nz=16,
        slab_x0=16, slab_x1=23,
        duct_y0=8, duct_y1=11,
        duct_z0=8, duct_z1=11,
        detector_x=30,
        roi_y0=8, roi_y1=11,
        roi_z0=8, roi_z1=11,
    )

    groups = EnergyGroups(
        n_groups=4,
        boundaries=(1.00, 0.70, 0.30, 0.10, 0.01),
    )

    mats = Materials(
        concrete=MaterialParams(
            p_int=(0.12, 0.10, 0.08, 0.06),
            p_abs=(0.10, 0.09, 0.08, 0.07),
            q_down=(0.20, 0.20, 0.15, 0.0),
        ),
        tungsten=MaterialParams(
            p_int=(0.35, 0.30, 0.25, 0.22),
            p_abs=(0.50, 0.45, 0.40, 0.35),
            q_down=(0.25, 0.20, 0.15, 0.0),
        ),
        air=MaterialParams(
            p_int=(0.02, 0.02, 0.015, 0.01),
            p_abs=(0.02, 0.02, 0.015, 0.01),
            q_down=(0.10, 0.08, 0.05, 0.0),
        ),
    )

    source = Source(x0=1, y0=9, z0=9, g0=0, dir0=0)
    cfg = TransportConfig(geom=geom, groups=groups, mats=mats, source=source)
    validate_config(cfg)
    return cfg

def validate_config(cfg: TransportConfig) -> None:
    g = cfg.geom
    if not (g.nx > 2 and g.ny > 2 and g.nz > 2):
        raise ValueError("grid must be >2 in each dimension")

    if not (0 <= g.slab_x0 <= g.slab_x1 < g.nx):
        raise ValueError("invalid slab x-range")
    if not (0 <= g.duct_y0 <= g.duct_y1 < g.ny):
        raise ValueError("invalid duct y-range")
    if not (0 <= g.duct_z0 <= g.duct_z1 < g.nz):
        raise ValueError("invalid duct z-range")

    if not (0 <= g.detector_x < g.nx):
        raise ValueError("invalid detector_x")
    if not (0 <= g.roi_y0 <= g.roi_y1 < g.ny):
        raise ValueError("invalid ROI y-range")
    if not (0 <= g.roi_z0 <= g.roi_z1 < g.nz):
        raise ValueError("invalid ROI z-range")

    if cfg.groups.n_groups < 1:
        raise ValueError("n_groups must be >= 1")
    if len(cfg.groups.boundaries) != cfg.groups.n_groups + 1:
        raise ValueError("energy boundaries length must be n_groups+1")

    for mat in (cfg.mats.concrete, cfg.mats.tungsten, cfg.mats.air):
        if len(mat.p_int) != cfg.groups.n_groups:
            raise ValueError("p_int length mismatch")
        if len(mat.p_abs) != cfg.groups.n_groups:
            raise ValueError("p_abs length mismatch")
        if len(mat.q_down) != cfg.groups.n_groups:
            raise ValueError("q_down length mismatch")
        for i in range(cfg.groups.n_groups):
            for name, v in (("p_int", mat.p_int[i]), ("p_abs", mat.p_abs[i]), ("q_down", mat.q_down[i])):
                if not (0.0 <= float(v) <= 1.0):
                    raise ValueError(f"{name}[{i}] must be in [0,1]")

    s = cfg.source
    if not (0 <= s.x0 < g.nx and 0 <= s.y0 < g.ny and 0 <= s.z0 < g.nz):
        raise ValueError("invalid source position")
    if not (0 <= s.g0 < cfg.groups.n_groups):
        raise ValueError("invalid source g0")
    if not (0 <= s.dir0 <= 5):
        raise ValueError("invalid source dir0 (0..5)")

def default_config() -> TransportConfig:

    return paper1_config()
