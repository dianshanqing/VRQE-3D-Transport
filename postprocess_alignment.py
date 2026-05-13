"""Postprocess alignment metrics across MC and TEST outputs."""
from __future__ import annotations
'\nPostprocess alignment / mechanism analysis for compressed outputs.\n\nDesign goals\n------------\n- Read the existing non-fullhist output layout without changing any existing code.\n- Prefer merged/results_merged.json for TEST outputs.\n- Read MC outputs from main.py --phase mc outdir/results.json.\n- Optionally read mc_diag JSON for x-profile mechanism analysis.\n- Produce compact CSV/JSON artifacts for paper tables/figures.\n\nSupported root layouts\n----------------------\nTEST root:\n  outputs\\test_s{step}_shards_total{shots}_compressed_seed1234_chunk{chunk}_{bond}_{trunc}\n  -> merged/results_merged.json\n\nMC root:\n  outputs\\mc_s{step}_h4e6_seed1234\n  -> results.json\n\nMC-DIAG json:\n  outputs\\mc_diag_s{step}_h2e6_seed1234.json\n  or any user-supplied json path.\n'
import argparse
import csv
import json
import math
import os
import statistics
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
STATES = ['detect', 'absorb', 'boundary', 'survive']
TERMINAL_STATES = ['detect', 'absorb', 'boundary']

def _read_json(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)

def _write_json(path: str, payload: Dict[str, Any]) -> None:
    _ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

def _write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    _ensure_dir(os.path.dirname(os.path.abspath(path)))
    base_fieldnames = list(fieldnames) if fieldnames else []
    seen = set(base_fieldnames)
    for row in rows:
        for k in row.keys():
            if k not in seen:
                base_fieldnames.append(k)
                seen.add(k)
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=base_fieldnames, extrasaction='ignore')
        w.writeheader()
        for row in rows:
            w.writerow(row)

def _fmt(x: Any) -> str:
    if isinstance(x, float):
        return f'{x:.6g}'
    return str(x)

def _safe_float(x: Any, default: float=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)

def _load_test_payload(root_or_json: str) -> Dict[str, Any]:
    p = root_or_json
    if os.path.isdir(p):
        p = os.path.join(p, 'merged', 'results_merged.json')
    if not os.path.exists(p):
        raise FileNotFoundError(f'TEST result message{p}')
    js = _read_json(p)
    js['_src_path'] = os.path.abspath(p)
    js['_src_root'] = os.path.abspath(os.path.dirname(os.path.dirname(p))) if p.endswith(os.path.join('merged', 'results_merged.json')) else os.path.abspath(os.path.dirname(p))
    return js

def _load_mc_payload(root_or_json: str) -> Dict[str, Any]:
    p = root_or_json
    if os.path.isdir(p):
        p = os.path.join(p, 'results.json')
    if not os.path.exists(p):
        raise FileNotFoundError(f'MC result message{p}')
    js = _read_json(p)
    js['_src_path'] = os.path.abspath(p)
    js['_src_root'] = os.path.abspath(os.path.dirname(p))
    return js

def _load_mcdiag_payload(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f'MC_DIAG result message{path}')
    js = _read_json(path)
    js['_src_path'] = os.path.abspath(path)
    return js

def _get_prob_vec(payload: Dict[str, Any], states: Sequence[str]=STATES) -> List[float]:
    return [_safe_float(payload.get(f'p_{s}', 0.0)) for s in states]

def _get_se_vec(payload: Dict[str, Any], states: Sequence[str]=STATES) -> List[float]:
    return [_safe_float(payload.get(f'se_{s}', 0.0)) for s in states]

def _combine_se(se_a: float, se_b: float) -> float:
    return math.sqrt(float(se_a) ** 2 + float(se_b) ** 2)

def _state_rows(test: Dict[str, Any], mc: Dict[str, Any], states: Sequence[str]=STATES) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for s in states:
        qp = _safe_float(test.get(f'p_{s}', 0.0))
        mp = _safe_float(mc.get(f'p_{s}', 0.0))
        qse = _safe_float(test.get(f'se_{s}', 0.0))
        mse = _safe_float(mc.get(f'se_{s}', 0.0))
        cse = _combine_se(qse, mse)
        z = (qp - mp) / cse if cse > 0 else 0.0
        rows.append({'state': s, 'test_p': qp, 'mc_p': mp, 'diff': qp - mp, 'abs_diff': abs(qp - mp), 'test_se': qse, 'mc_se': mse, 'combine_se': cse, 'z': z, 'abs_z': abs(z), 'within_3sigma': abs(z) <= 3.0})
    return rows

def _tvd(p: Sequence[float], q: Sequence[float]) -> float:
    return 0.5 * sum((abs(float(a) - float(b)) for a, b in zip(p, q)))

def _kl(p: Sequence[float], q: Sequence[float], log_base: float=2.0) -> float:
    if log_base <= 0:
        raise ValueError('log_base must be positive')
    denom = math.log(log_base)
    out = 0.0
    for pi, qi in zip(p, q):
        pi = float(pi)
        qi = float(qi)
        if pi <= 0.0:
            continue
        if qi <= 0.0:
            return float('inf')
        out += pi * (math.log(pi / qi) / denom)
    return out

def _jsd(p: Sequence[float], q: Sequence[float], log_base: float=2.0) -> float:
    m = [(float(a) + float(b)) / 2.0 for a, b in zip(p, q)]
    return 0.5 * _kl(p, m, log_base=log_base) + 0.5 * _kl(q, m, log_base=log_base)

def _nested_get(payload: Dict[str, Any], *path: str, default: Any=None) -> Any:
    cur: Any = payload
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def _xprob_map(payload: Dict[str, Any], key: str) -> Dict[int, float]:
    raw = payload.get(key, None)
    if raw is None and key.startswith('x_prob_'):
        raw = _nested_get(payload, key.replace('x_prob_', 'x_count_'), default=None)
    if not isinstance(raw, dict):
        return {}
    out: Dict[int, float] = {}
    for k, v in raw.items():
        try:
            out[int(k)] = float(v)
        except Exception:
            continue
    return out

def _normalize_hist(hist: Dict[int, float]) -> Dict[int, float]:
    s = sum((float(v) for v in hist.values()))
    if s <= 0:
        return {}
    return {int(k): float(v) / s for k, v in hist.items()}

def _xprob_from_count(payload: Dict[str, Any], count_key: str, denom_count_key: str) -> Dict[int, float]:
    raw = payload.get(count_key, None)
    if not isinstance(raw, dict):
        return {}
    denom = _safe_float(payload.get(denom_count_key, 0.0), 0.0)
    if denom <= 0:
        return {}
    out: Dict[int, float] = {}
    for k, v in raw.items():
        try:
            out[int(k)] = float(v) / denom
        except Exception:
            continue
    return out

def _get_xprob(payload: Dict[str, Any], label: str) -> Dict[int, float]:
    k = f'x_prob_{label}'
    raw = payload.get(k, None)
    if isinstance(raw, dict):
        return _xprob_map(payload, k)
    if label == 'total':
        return _xprob_from_count(payload, 'x_count_total', 'shots')
    denom_key = {'alive': 'n_alive', 'detect': 'n_detect', 'absorb': 'n_absorb', 'boundary': 'n_boundary'}.get(label, '')
    if denom_key:
        return _xprob_from_count(payload, f'x_count_{label}', denom_key)
    return {}

def _hist_union(a: Dict[int, float], b: Dict[int, float]) -> List[int]:
    return sorted(set(a.keys()) | set(b.keys()))

def _hist_tvd(a: Dict[int, float], b: Dict[int, float]) -> float:
    xs = _hist_union(a, b)
    return 0.5 * sum((abs(float(a.get(x, 0.0)) - float(b.get(x, 0.0))) for x in xs))

def _hist_jsd(a: Dict[int, float], b: Dict[int, float], log_base: float=2.0) -> float:
    xs = _hist_union(a, b)
    p = [float(a.get(x, 0.0)) for x in xs]
    q = [float(b.get(x, 0.0)) for x in xs]
    return _jsd(p, q, log_base=log_base)

def _front_mass(hist: Dict[int, float], x_min: int) -> float:
    return sum((float(v) for x, v in hist.items() if int(x) >= int(x_min)))

def _coin_stats(payload: Dict[str, Any]) -> Dict[str, Any]:
    cs = payload.get('coin_stats', None)
    if isinstance(cs, dict):
        return cs
    return {}

def _dir_stats(payload: Dict[str, Any]) -> Dict[str, Any]:
    ds = payload.get('dir_stats', None)
    if isinstance(ds, dict):
        return ds
    return {}

def _extract_witnesses(test: Dict[str, Any]) -> Dict[str, Any]:
    cs = _coin_stats(test)
    ds = _dir_stats(test)
    return {'p_ca1': cs.get('p_ca1', None), 'p_ci1': cs.get('p_ci1', None), 'ca1_ci0_total': int(_safe_float(cs.get('ca1_ci0_total', 0), 0)), 'nonabsorb_with_ca_total': int(_safe_float(cs.get('nonabsorb_with_ca_total', 0), 0)), 'absorb_without_ca_total': int(_safe_float(cs.get('absorb_without_ca_total', 0), 0)), 'ca1_ci0_rate': _safe_float(cs.get('ca1_ci0_rate', 0.0), 0.0), 'nonabsorb_with_ca_rate': _safe_float(cs.get('nonabsorb_with_ca_rate', 0.0), 0.0), 'absorb_without_ca_rate': _safe_float(cs.get('absorb_without_ca_rate', 0.0), 0.0), 'invalid_dir_rate': _safe_float(ds.get('invalid_dir_rate', 0.0), 0.0)}

def _source_decomposition_from_rows(rows: Sequence[Dict[str, Any]], target_positive_state: str='detect') -> Dict[str, float]:
    diff_by_state = {str(r['state']): float(r['diff']) for r in rows}
    pos = max(0.0, diff_by_state.get(target_positive_state, 0.0))
    if pos <= 0:
        return {}
    donors = [s for s in STATES if s != target_positive_state]
    deficits = {s: max(0.0, -diff_by_state.get(s, 0.0)) for s in donors}
    denom = sum(deficits.values())
    if denom <= 0:
        return {f'omega_{s}': 0.0 for s in donors}
    return {f'omega_{s}': deficits[s] / denom for s in donors}

def _compute_single(test: Dict[str, Any], mc: Dict[str, Any], mcdiag: Optional[Dict[str, Any]]=None, front_x_min: Optional[int]=None) -> Dict[str, Any]:
    test_vec = _get_prob_vec(test)
    mc_vec = _get_prob_vec(mc)
    rows = _state_rows(test, mc, STATES)
    max_abs_z = max((float(r['abs_z']) for r in rows)) if rows else 0.0
    rms_z = math.sqrt(sum((float(r['z']) ** 2 for r in rows)) / len(rows)) if rows else 0.0
    out: Dict[str, Any] = {'steps': int(_safe_float(test.get('steps', 0), 0)), 'test_src': test.get('_src_path', ''), 'mc_src': mc.get('_src_path', ''), 'states': rows, 'summary': {'max_abs_z': max_abs_z, 'rms_z': rms_z, 'tvd': _tvd(test_vec, mc_vec), 'jsd_bits': _jsd(test_vec, mc_vec, log_base=2.0), 'all_within_3sigma': all((bool(r['within_3sigma']) for r in rows))}, 'witness': _extract_witnesses(test), 'detect_source_decomposition': _source_decomposition_from_rows(rows, target_positive_state='detect'), 'absorb_source_decomposition': _source_decomposition_from_rows(rows, target_positive_state='absorb')}
    if front_x_min is not None:
        test_alive = _get_xprob(test, 'alive')
        out.setdefault('mechanism', {})['alive_front_mass_test'] = _front_mass(test_alive, front_x_min)
        out['mechanism']['front_x_min'] = int(front_x_min)
        if mcdiag is not None:
            mc_alive = _get_xprob(mcdiag, 'alive')
            out['mechanism']['alive_front_mass_mcdiag'] = _front_mass(mc_alive, front_x_min)
            out['mechanism']['alive_front_mass_diff'] = out['mechanism']['alive_front_mass_test'] - out['mechanism']['alive_front_mass_mcdiag']
    if mcdiag is not None:
        out['mcdiag_src'] = mcdiag.get('_src_path', '')
        xcmp: Dict[str, Any] = {}
        for label in ['alive', 'detect', 'absorb', 'boundary', 'total']:
            a = _get_xprob(test, label)
            b = _get_xprob(mcdiag, label)
            if a and b:
                xcmp[f'x_tvd_{label}'] = _hist_tvd(a, b)
                xcmp[f'x_jsd_bits_{label}'] = _hist_jsd(a, b, log_base=2.0)
        out['x_compare'] = xcmp
    return out

def _parse_steps(spec: str) -> List[int]:
    spec = str(spec).strip()
    if not spec:
        return []
    out: List[int] = []
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            a = int(a.strip())
            b = int(b.strip())
            if a <= b:
                out.extend(range(a, b + 1))
            else:
                out.extend(range(a, b - 1, -1))
        else:
            out.append(int(part))
    seen = set()
    dedup = []
    for x in out:
        if x not in seen:
            dedup.append(x)
            seen.add(x)
    return dedup

def _parse_ints(spec: str) -> List[int]:
    return [int(x.strip()) for x in str(spec).split(',') if x.strip()]

def _series_monotonicity(rows: Sequence[Dict[str, Any]], state: str) -> float:
    vals = [float(r[f'test_p_{state}']) for r in rows if f'test_p_{state}' in r]
    if len(vals) < 2:
        return 0.0
    if state in TERMINAL_STATES:
        return sum((max(0.0, vals[i - 1] - vals[i]) for i in range(1, len(vals))))
    if state == 'survive':
        return sum((max(0.0, vals[i] - vals[i - 1]) for i in range(1, len(vals))))
    return 0.0

def _series_increments(rows: Sequence[Dict[str, Any]], state: str, prefix: str) -> List[float]:
    key = f'{prefix}_p_{state}'
    vals = [float(r[key]) for r in rows if key in r]
    if not vals:
        return []
    out = [vals[0]]
    for i in range(1, len(vals)):
        out.append(vals[i] - vals[i - 1])
    return out

def _cmd_single(args: argparse.Namespace) -> int:
    test = _load_test_payload(args.test_root)
    mc = _load_mc_payload(args.mc_root)
    mcdiag = _load_mcdiag_payload(args.mcdiag_json) if args.mcdiag_json else None
    res = _compute_single(test, mc, mcdiag=mcdiag, front_x_min=args.front_x_min)
    out_json = args.out_json or os.path.join(args.outdir, f"single_s{res['steps']}.json")
    _write_json(out_json, res)
    print('=' * 72)
    print('Postprocess Alignment: single')
    print('=' * 72)
    print(f"TEST : {test.get('_src_path', '')}")
    print(f"MC   : {mc.get('_src_path', '')}")
    if mcdiag:
        print(f"MCDIAG: {mcdiag.get('_src_path', '')}")
    print(f"steps: {res['steps']}")
    print('-')
    for row in res['states']:
        print(f"{row['state']:8s}  test={row['test_p']:.6f}  mc={row['mc_p']:.6f}  diff={row['diff']:+.6f}  z={row['z']:+.3f}")
    print('-')
    sm = res['summary']
    print(f"max-|z| : {sm['max_abs_z']:.6f}")
    print(f"RMS-z   : {sm['rms_z']:.6f}")
    print(f"TVD     : {sm['tvd']:.6f}")
    print(f"JSD(bits): {sm['jsd_bits']:.6e}")
    if 'mechanism' in res:
        mech = res['mechanism']
        if 'alive_front_mass_test' in mech:
            print(f"alive_front_mass(test, x>={mech['front_x_min']}): {mech['alive_front_mass_test']:.6f}")
        if 'alive_front_mass_mcdiag' in mech:
            print(f"alive_front_mass(mcdiag): {mech['alive_front_mass_mcdiag']:.6f}")
            print(f"alive_front_mass(diff)  : {mech['alive_front_mass_diff']:+.6f}")
    print(f'saved json: {out_json}')
    return 0

def _cmd_series(args: argparse.Namespace) -> int:
    steps = _parse_steps(args.steps)
    if not steps:
        raise SystemExit('--steps cannot be empty')
    rows: List[Dict[str, Any]] = []
    for step in steps:
        test_root = args.test_template.format(step=step)
        mc_root = args.mc_template.format(step=step)
        test = _load_test_payload(test_root)
        mc = _load_mc_payload(mc_root)
        mcdiag = None
        if args.mcdiag_template:
            p = args.mcdiag_template.format(step=step)
            if os.path.exists(p):
                mcdiag = _load_mcdiag_payload(p)
        res = _compute_single(test, mc, mcdiag=mcdiag, front_x_min=args.front_x_min)
        sm = res['summary']
        row: Dict[str, Any] = {'step': step, 'test_shots': int(_safe_float(test.get('shots', 0), 0)), 'mc_histories': int(_safe_float(mc.get('histories', 0), 0)), 'max_abs_z': float(sm['max_abs_z']), 'rms_z': float(sm['rms_z']), 'tvd': float(sm['tvd']), 'jsd_bits': float(sm['jsd_bits']), 'all_within_3sigma': bool(sm['all_within_3sigma'])}
        for st in STATES:
            st_row = next((r for r in res['states'] if r['state'] == st))
            row[f'test_p_{st}'] = float(st_row['test_p'])
            row[f'mc_p_{st}'] = float(st_row['mc_p'])
            row[f'diff_{st}'] = float(st_row['diff'])
            row[f'z_{st}'] = float(st_row['z'])
        wit = res.get('witness', {})
        for k in ['ca1_ci0_rate', 'nonabsorb_with_ca_rate', 'absorb_without_ca_rate', 'invalid_dir_rate']:
            row[k] = float(_safe_float(wit.get(k, 0.0), 0.0))
        if args.front_x_min is not None and 'mechanism' in res:
            mech = res['mechanism']
            row['alive_front_mass_test'] = float(_safe_float(mech.get('alive_front_mass_test', 0.0), 0.0))
            if 'alive_front_mass_mcdiag' in mech:
                row['alive_front_mass_mcdiag'] = float(_safe_float(mech.get('alive_front_mass_mcdiag', 0.0), 0.0))
                row['alive_front_mass_diff'] = float(_safe_float(mech.get('alive_front_mass_diff', 0.0), 0.0))
        rows.append(row)
    for st in TERMINAL_STATES:
        h_test = _series_increments(rows, st, 'test')
        h_mc = _series_increments(rows, st, 'mc')
        for i, row in enumerate(rows):
            row[f'h_test_{st}'] = float(h_test[i])
            row[f'h_mc_{st}'] = float(h_mc[i])
            row[f'delta_h_{st}'] = float(h_test[i] - h_mc[i])
    h_test_survive = _series_increments(rows, 'survive', 'test')
    h_mc_survive = _series_increments(rows, 'survive', 'mc')
    for i, row in enumerate(rows):
        row['h_test_survive'] = float(h_test_survive[i])
        row['h_mc_survive'] = float(h_mc_survive[i])
        row['delta_h_survive'] = float(h_test_survive[i] - h_mc_survive[i])
    summary = {'steps': steps, 'test_template': args.test_template, 'mc_template': args.mc_template, 'mcdiag_template': args.mcdiag_template, 'front_x_min': args.front_x_min, 'series_metrics': {'pass_rate_3sigma': sum((1 for r in rows if r['all_within_3sigma'])) / float(len(rows)), 'max_over_steps_max_abs_z': max((float(r['max_abs_z']) for r in rows)), 'mean_tvd': statistics.mean((float(r['tvd']) for r in rows)), 'mean_jsd_bits': statistics.mean((float(r['jsd_bits']) for r in rows)), 'V_detect': _series_monotonicity(rows, 'detect'), 'V_absorb': _series_monotonicity(rows, 'absorb'), 'V_boundary': _series_monotonicity(rows, 'boundary'), 'V_survive': _series_monotonicity(rows, 'survive')}}
    out_json = args.out_json or os.path.join(args.outdir, 'series_metrics.json')
    out_csv = args.out_csv or os.path.join(args.outdir, 'series_metrics.csv')
    fieldnames = list(rows[0].keys()) if rows else []
    _write_json(out_json, {'summary': summary, 'rows': rows})
    _write_csv(out_csv, rows, fieldnames=fieldnames)
    print('=' * 72)
    print('Postprocess Alignment: series')
    print('=' * 72)
    print(f'steps           : {steps}')
    print(f'test_template   : {args.test_template}')
    print(f'mc_template     : {args.mc_template}')
    if args.mcdiag_template:
        print(f'mcdiag_template : {args.mcdiag_template}')
    print('-')
    for k, v in summary['series_metrics'].items():
        print(f'{k:24s}: {_fmt(v)}')
    print(f'saved json: {out_json}')
    print(f'saved csv : {out_csv}')
    return 0

def _cmd_frontier(args: argparse.Namespace) -> int:
    bonds = _parse_ints(args.bonds)
    if not bonds:
        raise SystemExit('--bonds cannot be empty')
    mc = _load_mc_payload(args.mc_root)
    mcdiag = _load_mcdiag_payload(args.mcdiag_json) if args.mcdiag_json else None
    rows: List[Dict[str, Any]] = []
    for bond in bonds:
        test_root = args.test_template.format(step=args.step, bond=bond)
        if not os.path.exists(test_root) and (not os.path.exists(os.path.join(test_root, 'merged', 'results_merged.json'))):
            print(f'[warn] skip bond={bond}: missing TEST root {test_root}')
            continue
        test = _load_test_payload(test_root)
        res = _compute_single(test, mc, mcdiag=mcdiag, front_x_min=args.front_x_min)
        sm = res['summary']
        row: Dict[str, Any] = {'step': int(args.step), 'bond': int(bond), 'test_shots': int(_safe_float(test.get('shots', 0), 0)), 'max_abs_z': float(sm['max_abs_z']), 'rms_z': float(sm['rms_z']), 'tvd': float(sm['tvd']), 'jsd_bits': float(sm['jsd_bits']), 'all_within_3sigma': bool(sm['all_within_3sigma'])}
        for st in STATES:
            st_row = next((r for r in res['states'] if r['state'] == st))
            row[f'test_p_{st}'] = float(st_row['test_p'])
            row[f'diff_{st}'] = float(st_row['diff'])
            row[f'z_{st}'] = float(st_row['z'])
        wit = res.get('witness', {})
        for k in ['ca1_ci0_rate', 'nonabsorb_with_ca_rate', 'absorb_without_ca_rate', 'invalid_dir_rate']:
            row[k] = float(_safe_float(wit.get(k, 0.0), 0.0))
        if args.front_x_min is not None and 'mechanism' in res:
            mech = res['mechanism']
            row['alive_front_mass_test'] = float(_safe_float(mech.get('alive_front_mass_test', 0.0), 0.0))
            if 'alive_front_mass_mcdiag' in mech:
                row['alive_front_mass_mcdiag'] = float(_safe_float(mech.get('alive_front_mass_mcdiag', 0.0), 0.0))
                row['alive_front_mass_diff'] = float(_safe_float(mech.get('alive_front_mass_diff', 0.0), 0.0))
        rows.append(row)
    out_json = args.out_json or os.path.join(args.outdir, f'frontier_s{args.step}.json')
    out_csv = args.out_csv or os.path.join(args.outdir, f'frontier_s{args.step}.csv')
    _write_json(out_json, {'step': int(args.step), 'mc_root': args.mc_root, 'rows': rows})
    _write_csv(out_csv, rows, fieldnames=list(rows[0].keys()) if rows else [])
    print('=' * 72)
    print('Postprocess Alignment: frontier')
    print('=' * 72)
    print(f'step : {args.step}')
    print(f'bonds: {bonds}')
    print('-')
    for row in rows:
        print(f"bond={row['bond']:4d}  max-|z|={row['max_abs_z']:.6f}  TVD={row['tvd']:.6f}  JSD(bits)={row['jsd_bits']:.6e}  absorb_wo_ca={row['absorb_without_ca_rate']:.6f}")
    print(f'saved json: {out_json}')
    print(f'saved csv : {out_csv}')
    return 0

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description='compressed: postprocess alignment / mechanism analysis')
    sub = ap.add_subparsers(dest='cmd', required=True)
    ap_single = sub.add_parser('single', help='diagnostic message')
    ap_single.add_argument('--test-root', required=True, help='TEST result message')
    ap_single.add_argument('--mc-root', required=True, help='Missing results.json')
    ap_single.add_argument('--mcdiag-json', default='', help='diagnostic message')
    ap_single.add_argument('--front-x-min', type=int, default=None, help='diagnostic message')
    ap_single.add_argument('--outdir', default=os.path.join('outputs', 'postprocess'))
    ap_single.add_argument('--out-json', default='')
    ap_single.set_defaults(func=_cmd_single)
    ap_series = sub.add_parser('series', help='diagnostic message')
    ap_series.add_argument('--steps', required=True, help='diagnostic message')
    ap_series.add_argument('--test-template', required=True, help='diagnostic message')
    ap_series.add_argument('--mc-template', required=True, help='diagnostic message')
    ap_series.add_argument('--mcdiag-template', default='', help='diagnostic message')
    ap_series.add_argument('--front-x-min', type=int, default=None, help='diagnostic message')
    ap_series.add_argument('--outdir', default=os.path.join('outputs', 'postprocess'))
    ap_series.add_argument('--out-json', default='')
    ap_series.add_argument('--out-csv', default='')
    ap_series.set_defaults(func=_cmd_series)
    ap_frontier = sub.add_parser('frontier', help='diagnostic message')
    ap_frontier.add_argument('--step', type=int, required=True)
    ap_frontier.add_argument('--bonds', required=True, help='diagnostic message')
    ap_frontier.add_argument('--test-template', required=True, help='diagnostic message')
    ap_frontier.add_argument('--mc-root', required=True, help='diagnostic message')
    ap_frontier.add_argument('--mcdiag-json', default='', help='diagnostic message')
    ap_frontier.add_argument('--front-x-min', type=int, default=None, help='diagnostic message')
    ap_frontier.add_argument('--outdir', default=os.path.join('outputs', 'postprocess'))
    ap_frontier.add_argument('--out-json', default='')
    ap_frontier.add_argument('--out-csv', default='')
    ap_frontier.set_defaults(func=_cmd_frontier)
    return ap

def main() -> int:
    ap = build_argparser()
    args = ap.parse_args()
    return int(args.func(args))
if __name__ == '__main__':
    raise SystemExit(main())
