"""Compare Monte Carlo and circuit-sampling result directories."""
import argparse
import json
import os
from pathlib import Path

def _load_results(outdir: str):
    p = Path(outdir) / 'results.json'
    if not p.exists():
        raise FileNotFoundError(f'Missing results.json{p}')
    with p.open('r', encoding='utf-8') as f:
        return json.load(f)

def _fmt(x):
    if x is None:
        return 'N/A'
    if isinstance(x, (int,)):
        return str(x)
    try:
        return f'{float(x):.6f}'
    except Exception:
        return str(x)

def main():
    ap = argparse.ArgumentParser(description='TEST result message')
    ap.add_argument('--mc', required=True, help='Missing results.json')
    ap.add_argument('--test', required=True, help='Missing results.json')
    ap.add_argument('--show-xhist', action='store_true', help='diagnostic message')
    args = ap.parse_args()
    mc = _load_results(args.mc)
    te = _load_results(args.test)
    keys = ['steps', 'histories', 'shots', 'p_detect', 'p_absorb', 'p_boundary', 'p_survive']
    print('========================================')
    print(' Paper-1 Alignment Summary (MC vs TEST) ')
    print('========================================')
    print(f'MC  dir: {args.mc}')
    print(f'TEST dir: {args.test}')
    print('')

    def show_block(title, d):
        print(f'[{title}]')
        for k in keys:
            if k in d:
                print(f'  {k:10s}: {_fmt(d[k])}')
        if 'num_qubits' in d:
            print(f"  num_qubits : {d['num_qubits']}")
        if 'backend' in d:
            print(f"  backend    : {d['backend']}")
        print('')
    show_block('MC', mc)
    show_block('TEST', te)

    def get(d, k):
        return d.get(k, None)
    print('[ABS DIFF |TEST - MC|]')
    for k in ['p_detect', 'p_absorb', 'p_boundary', 'p_survive']:
        a = get(mc, k)
        b = get(te, k)
        if a is None or b is None:
            continue
        try:
            diff = abs(float(b) - float(a))
            print(f'  {k:10s}: {diff:.6f}')
        except Exception:
            pass
    if args.show_xhist:
        xh = te.get('x_prob_total', None) or te.get('x_hist_total', None) or te.get('x_hist', None)
        if xh:
            print('')
            print('[TEST x histogram (total)]')
            items = sorted(((int(k), v) for k, v in xh.items()), key=lambda kv: kv[0])
            for k, v in items[:50]:
                print(f'  x={k:02d}: {v:.6f}')
        else:
            print('')
            print('TEST result message')
if __name__ == '__main__':
    main()
