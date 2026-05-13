"""Report sigma-level agreement between merged TEST output and an MC reference."""

from __future__ import annotations
import argparse, json, math
from typing import Dict, Any

def _read_json(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def _sigma(test_p: float, test_se: float, mc_p: float, mc_se: float) -> float:
    den = math.sqrt(float(test_se) ** 2 + float(mc_se) ** 2)
    if den <= 0:
        return 0.0
    return (float(test_p) - float(mc_p)) / den

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--test-merged', required=True)
    ap.add_argument('--mc-json', default='')
    ap.add_argument('--mc-p-absorb', type=float, default=None)
    ap.add_argument('--mc-p-boundary', type=float, default=None)
    ap.add_argument('--mc-p-survive', type=float, default=None)
    ap.add_argument('--mc-se-absorb', type=float, default=0.0)
    ap.add_argument('--mc-se-boundary', type=float, default=0.0)
    ap.add_argument('--mc-se-survive', type=float, default=0.0)
    args = ap.parse_args()
    test = _read_json(args.test_merged)
    if args.mc_json:
        mc = _read_json(args.mc_json)
        mc_p_absorb = float(mc['p_absorb']); mc_se_absorb = float(mc.get('se_absorb', 0.0))
        mc_p_boundary = float(mc['p_boundary']); mc_se_boundary = float(mc.get('se_boundary', 0.0))
        mc_p_survive = float(mc['p_survive']); mc_se_survive = float(mc.get('se_survive', 0.0))
    else:
        mc_p_absorb = float(args.mc_p_absorb); mc_se_absorb = float(args.mc_se_absorb)
        mc_p_boundary = float(args.mc_p_boundary); mc_se_boundary = float(args.mc_se_boundary)
        mc_p_survive = float(args.mc_p_survive); mc_se_survive = float(args.mc_se_survive)
    rows = []
    for key, mp, mse in [('absorb', mc_p_absorb, mc_se_absorb), ('boundary', mc_p_boundary, mc_se_boundary), ('survive', mc_p_survive, mc_se_survive)]:
        tp = float(test['p_' + key]); tse = float(test['se_' + key])
        z = _sigma(tp, tse, mp, mse)
        rows.append({'state': key, 'test_p': tp, 'test_se': tse, 'mc_p': mp, 'mc_se': mse, 'diff': tp - mp, 'combined_sigma': z, 'within_3sigma': abs(z) <= 3.0})
    print(json.dumps({'steps': test.get('steps'), 'shots': test.get('shots'), 'rows': rows}, ensure_ascii=False, indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
