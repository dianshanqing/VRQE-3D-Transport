"""Monte Carlo diagnostic utilities for spatial and state distributions."""

from __future__ import annotations
import argparse, json, math, os
from typing import Dict
import numpy as np
from config import paper1_config
from geometry import is_boundary, is_in_roi, material_at, clamp_pos

_DIRS = [
    (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
]

def _binom_se(p: float, n: int) -> float:
    if n <= 0:
        return 0.0
    p = max(0.0, min(1.0, float(p)))
    return math.sqrt(p * (1.0 - p) / float(n))

def run_mc_diag(steps: int, histories: int, seed: int = 1234) -> Dict:
    cfg = paper1_config()
    rng = np.random.default_rng(seed)
    c_detect = c_absorb = c_boundary = c_survive = 0
    x_total: Dict[int, int] = {}
    x_alive: Dict[int, int] = {}
    x_detect: Dict[int, int] = {}
    x_absorb: Dict[int, int] = {}
    x_boundary: Dict[int, int] = {}
    for _ in range(histories):
        x, y, z = cfg.source.x0, cfg.source.y0, cfg.source.z0
        g = cfg.source.g0
        dir_idx = int(getattr(cfg.source, 'dir0', 0))
        status = 'alive'
        for _t in range(steps):
            if is_boundary(cfg, x, y, z):
                status = 'boundary'; c_boundary += 1; break
            if is_in_roi(cfg, x, y, z):
                status = 'detect'; c_detect += 1; break
            mp = getattr(cfg.mats, material_at(cfg, x, y, z))
            if rng.random() < float(mp.p_int[g]):
                if rng.random() < float(mp.p_abs[g]):
                    status = 'absorb'; c_absorb += 1; break
                dir_idx = int(rng.integers(0, 6))
            dx, dy, dz = _DIRS[dir_idx]
            x, y, z = clamp_pos(cfg, x + dx, y + dy, z + dz)
        else:
            status = 'survive'; c_survive += 1
        x_total[x] = x_total.get(x, 0) + 1
        if status in ('alive', 'survive'):
            x_alive[x] = x_alive.get(x, 0) + 1
        elif status == 'detect':
            x_detect[x] = x_detect.get(x, 0) + 1
        elif status == 'absorb':
            x_absorb[x] = x_absorb.get(x, 0) + 1
        elif status == 'boundary':
            x_boundary[x] = x_boundary.get(x, 0) + 1
    p_detect = c_detect / histories
    p_absorb = c_absorb / histories
    p_boundary = c_boundary / histories
    p_survive = c_survive / histories
    return {
        'phase': 'mc_diag',
        'steps': int(steps),
        'histories': int(histories),
        'seed': int(seed),
        'n_detect': int(c_detect),
        'n_absorb': int(c_absorb),
        'n_boundary': int(c_boundary),
        'n_survive': int(c_survive),
        'p_detect': p_detect,
        'p_absorb': p_absorb,
        'p_boundary': p_boundary,
        'p_survive': p_survive,
        'se_detect': _binom_se(p_detect, histories),
        'se_absorb': _binom_se(p_absorb, histories),
        'se_boundary': _binom_se(p_boundary, histories),
        'se_survive': _binom_se(p_survive, histories),
        'x_count_total': {str(k): int(v) for k, v in sorted(x_total.items())},
        'x_count_alive': {str(k): int(v) for k, v in sorted(x_alive.items())},
        'x_count_detect': {str(k): int(v) for k, v in sorted(x_detect.items())},
        'x_count_absorb': {str(k): int(v) for k, v in sorted(x_absorb.items())},
        'x_count_boundary': {str(k): int(v) for k, v in sorted(x_boundary.items())},
        'diag': {
            'alive_at_x0': int(x_alive.get(0, 0)),
            'alive_at_xmax': int(x_alive.get(cfg.geom.nx - 1, 0)),
            'alive_on_x_boundary': bool(x_alive.get(0, 0) + x_alive.get(cfg.geom.nx - 1, 0) > 0),
        },
    }

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=3)
    ap.add_argument('--histories', type=int, default=1000000)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--out', type=str, default='')
    args = ap.parse_args()
    out = run_mc_diag(args.steps, args.histories, args.seed)
    txt = json.dumps(out, ensure_ascii=False, indent=2)
    print(txt)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            f.write(txt)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
