"""Run full-step diagnostics for selected transport subchains."""
from __future__ import annotations
import argparse
import json
import math
import os
from dataclasses import replace
from datetime import datetime
from typing import Dict, Iterable, List, Tuple
from config import default_config, Materials, MaterialParams, TransportConfig
from unitary_circuit import build_unitary_circuit_qiskit

def _binom_se(p: float, n: int) -> float:
    if n <= 0:
        return 0.0
    p = max(0.0, min(1.0, float(p)))
    return math.sqrt(p * (1.0 - p) / float(n))

def _tv_distance(p: Dict[Tuple[int, ...], float], q: Dict[Tuple[int, ...], float]) -> float:
    keys = set(p) | set(q)
    return 0.5 * sum((abs(float(p.get(k, 0.0)) - float(q.get(k, 0.0))) for k in keys))

def _get_reg(qc, name: str):
    for reg in qc.qregs:
        if reg.name == name:
            return reg
    raise KeyError(f'register not found: {name}')

def _measure_distribution_mps(qc, reg_groups: List[Tuple[str, Iterable]], *, shots: int, seed: int, aer_safe_basis: bool, aer_threads: int, mps_max_bond: int, mps_trunc: float) -> Dict[Tuple[int, ...], float]:
    from qiskit import ClassicalRegister, QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    total_bits = sum((len(list(reg)) for _, reg in reg_groups))
    qcm = QuantumCircuit(*qc.qregs, name=qc.name + '_meas')
    qcm.compose(qc, inplace=True)
    creg = ClassicalRegister(total_bits, 'c')
    qcm.add_register(creg)
    cidx = 0
    widths: List[int] = []
    for _, reg in reg_groups:
        reg_list = list(reg)
        widths.append(len(reg_list))
        for q in reg_list:
            qcm.measure(q, creg[cidx])
            cidx += 1
    sim = AerSimulator(method='matrix_product_state')
    sim.set_options(max_parallel_threads=int(aer_threads))
    if int(mps_max_bond) > 0:
        sim.set_options(matrix_product_state_max_bond_dimension=int(mps_max_bond))
    if float(mps_trunc) > 0.0:
        sim.set_options(matrix_product_state_truncation_threshold=float(mps_trunc))
    if aer_safe_basis:
        try:
            sim.set_options(fusion_enable=False)
        except Exception:
            pass
        tqc = transpile(qcm, optimization_level=0, basis_gates=['rz', 'sx', 'x', 'cx'], num_processes=1)
    else:
        tqc = transpile(qcm, optimization_level=0, num_processes=1)
    job = sim.run(tqc, shots=int(shots), seed_simulator=int(seed))
    counts = job.result().get_counts()
    dist: Dict[Tuple[int, ...], float] = {}
    for bitstr, cnt in counts.items():
        bits_lsb = list(reversed(bitstr.strip()))
        pos = 0
        vals = []
        for width in widths:
            val = 0
            for i in range(width):
                val |= int(bits_lsb[pos + i]) << i
            pos += width
            vals.append(val)
        key = tuple(vals)
        dist[key] = dist.get(key, 0.0) + float(cnt) / float(shots)
    return dist

def _zmax_vs_target(mps: Dict[Tuple[int, ...], float], target: Dict[Tuple[int, ...], float], shots: int) -> float:
    keys = set(mps) | set(target)
    zmax = 0.0
    for k in keys:
        pt = float(target.get(k, 0.0))
        pm = float(mps.get(k, 0.0))
        se = _binom_se(max(pt, 1e-15), shots)
        if se > 0:
            zmax = max(zmax, abs((pm - pt) / se))
    return float(zmax)

def _cfg_for_scatter_only() -> TransportConfig:
    cfg = default_config()

    def _force_scatter(_: MaterialParams) -> MaterialParams:
        return MaterialParams(p_int=(1.0, 1.0), p_abs=(0.0, 0.0), q_down=(0.0, 0.0))
    mats = Materials(concrete=_force_scatter(cfg.mats.concrete), tungsten=_force_scatter(cfg.mats.tungsten), air=_force_scatter(cfg.mats.air))
    return replace(cfg, mats=mats)

def _scatter_target_one_step(cfg: TransportConfig) -> Dict[Tuple[int, int, int, int, int], float]:
    x0, y0, z0 = (int(cfg.source.x0), int(cfg.source.y0), int(cfg.source.z0))
    return {(0, 0, x0 + 1, y0, z0): 1.0 / 6.0, (0, 1, x0 - 1, y0, z0): 1.0 / 6.0, (0, 2, x0, y0 + 1, z0): 1.0 / 6.0, (0, 3, x0, y0 - 1, z0): 1.0 / 6.0, (0, 4, x0, y0, z0 + 1): 1.0 / 6.0, (0, 5, x0, y0, z0 - 1): 1.0 / 6.0}

def _status_target_default_one_step(cfg: TransportConfig) -> Dict[Tuple[int], float]:
    p_int = float(cfg.mats.concrete.p_int[cfg.source.g0])
    p_abs = float(cfg.mats.concrete.p_abs[cfg.source.g0])
    p_absorb = p_int * p_abs
    return {(0,): 1.0 - p_absorb, (1,): 0.0, (2,): p_absorb, (3,): 0.0}

def run_a1(*, shots: int, seed: int, aer_safe_basis: bool, aer_threads: int, mps_max_bond: int, mps_trunc: float) -> Dict[str, object]:
    cfg = _cfg_for_scatter_only()
    qc, info = build_unitary_circuit_qiskit(cfg, n_steps=1, use_dir_tape=True, use_absorb_tape=True, use_term_tape=True, disable_detector=True, disable_boundary=False)
    reg_groups = [('status', _get_reg(qc, 'status')), ('dir', _get_reg(qc, 'dir')), ('x', _get_reg(qc, 'x')), ('y', _get_reg(qc, 'y')), ('z', _get_reg(qc, 'z'))]
    mps = _measure_distribution_mps(qc, reg_groups, shots=shots, seed=seed, aer_safe_basis=aer_safe_basis, aer_threads=aer_threads, mps_max_bond=mps_max_bond, mps_trunc=mps_trunc)
    target = _scatter_target_one_step(cfg)
    tv = _tv_distance(mps, target)
    support = sum((mps.get(k, 0.0) for k in target))
    zmax = _zmax_vs_target(mps, target, shots)
    if zmax > 5.0 or tv > 0.02 or 1.0 - support > 0.002:
        label = 'FULLSTEP_SINGLE_STEP_MISMATCH'
        msg = 'diagnostic message'
    else:
        label = 'PASS'
        msg = 'diagnostic message'
    return {'kind': 'A1_scatter_only_step', 'target_distribution': {str(k): v for k, v in target.items()}, 'mps_distribution': {str(k): v for k, v in sorted(mps.items()) if v > 1e-12}, 'metrics': {'tv_mps_vs_target': float(tv), 'support_mass_on_target': float(support), 'bad_mass_off_target': float(max(0.0, 1.0 - support)), 'max_state_z_mps_vs_target': float(zmax)}, 'label': label, 'message': msg, 'n_qubits': int(qc.num_qubits), 'depth': int(qc.depth()), 'count_ops': {str(k): int(v) for k, v in qc.count_ops().items()}, 'resources': info}

def run_a2(*, shots: int, seed: int, aer_safe_basis: bool, aer_threads: int, mps_max_bond: int, mps_trunc: float) -> Dict[str, object]:
    cfg = default_config()
    qc, info = build_unitary_circuit_qiskit(cfg, n_steps=1, use_dir_tape=True, use_absorb_tape=True, use_term_tape=True, disable_detector=False, disable_boundary=False)
    reg_groups = [('status', _get_reg(qc, 'status'))]
    mps = _measure_distribution_mps(qc, reg_groups, shots=shots, seed=seed, aer_safe_basis=aer_safe_basis, aer_threads=aer_threads, mps_max_bond=mps_max_bond, mps_trunc=mps_trunc)
    target = _status_target_default_one_step(cfg)
    tv = _tv_distance(mps, target)
    zmax = _zmax_vs_target(mps, target, shots)
    if zmax > 5.0 or tv > 0.01:
        label = 'FULLSTEP_SINGLE_STEP_MISMATCH'
        msg = 'MC result message'
    else:
        label = 'PASS'
        msg = 'MC result message'
    return {'kind': 'A2_full_step_absorb_vs_survive', 'target_status_distribution': {str(k): v for k, v in target.items()}, 'mps_status_distribution': {str(k): v for k, v in sorted(mps.items()) if v > 1e-12}, 'metrics': {'tv_mps_vs_target': float(tv), 'max_state_z_mps_vs_target': float(zmax)}, 'label': label, 'message': msg, 'n_qubits': int(qc.num_qubits), 'depth': int(qc.depth()), 'count_ops': {str(k): int(v) for k, v in qc.count_ops().items()}, 'resources': info}

def summarize(a1: Dict[str, object], a2: Dict[str, object]) -> Dict[str, str]:
    l1 = str(a1['label'])
    l2 = str(a2['label'])
    if l1 != 'PASS':
        return {'overall_label': 'A1_SINGLE_STEP_SCATTER_MOVE_ALREADY_FAILS', 'overall_message': 'diagnostic message'}
    if l2 != 'PASS':
        return {'overall_label': 'A2_SINGLE_STEP_DEFAULT_FULLSTEP_ALREADY_FAILS', 'overall_message': 'MC result message'}
    return {'overall_label': 'A1_A2_PASS_MULTI_STEP_FLOOR_REMAINS', 'overall_message': 'diagnostic message'}

def build_argparser():
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument('--which', type=str, default='A1,A2', help='comma separated: A1,A2')
    p.add_argument('--shots', type=int, default=200000)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--aer-safe-basis', action='store_true')
    p.add_argument('--aer-threads', type=int, default=1)
    p.add_argument('--mps-max-bond', type=int, default=256)
    p.add_argument('--mps-trunc', type=float, default=1e-12)
    p.add_argument('--outdir', type=str, default='outputs/diag_fullstep_a1a2')
    return p

def main(argv=None):
    args = build_argparser().parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)
    which = [s.strip().upper() for s in str(args.which).split(',') if s.strip()]
    payload: Dict[str, object] = {'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'which': which, 'shots': int(args.shots), 'seed': int(args.seed), 'aer_safe_basis': bool(args.aer_safe_basis), 'aer_threads': int(args.aer_threads), 'mps_max_bond': int(args.mps_max_bond), 'mps_trunc': float(args.mps_trunc)}
    print('=== Final full-step diagnostic-only tests (A1 + A2) ===')
    print(f'which={which} shots={int(args.shots)} seed={int(args.seed)} safe_basis={bool(args.aer_safe_basis)} bond={int(args.mps_max_bond)} trunc={float(args.mps_trunc)}')
    a1 = a2 = None
    if 'A1' in which:
        print('\n--- Running A1: scatter_only_step ---')
        a1 = run_a1(shots=int(args.shots), seed=int(args.seed), aer_safe_basis=bool(args.aer_safe_basis), aer_threads=int(args.aer_threads), mps_max_bond=int(args.mps_max_bond), mps_trunc=float(args.mps_trunc))
        print(f"A1 label={a1['label']} :: {a1['message']}")
    if 'A2' in which:
        print('\n--- Running A2: full_step_absorb_vs_survive ---')
        a2 = run_a2(shots=int(args.shots), seed=int(args.seed), aer_safe_basis=bool(args.aer_safe_basis), aer_threads=int(args.aer_threads), mps_max_bond=int(args.mps_max_bond), mps_trunc=float(args.mps_trunc))
        print(f"A2 label={a2['label']} :: {a2['message']}")
    payload['A1'] = a1
    payload['A2'] = a2
    if a1 is not None and a2 is not None:
        payload['summary'] = summarize(a1, a2)
    elif a1 is not None:
        payload['summary'] = {'overall_label': str(a1['label']), 'overall_message': str(a1['message'])}
    elif a2 is not None:
        payload['summary'] = {'overall_label': str(a2['label']), 'overall_message': str(a2['message'])}
    else:
        raise SystemExit('No valid tests selected. Use --which A1,A2 or A1 or A2.')
    with open(os.path.join(args.outdir, 'fullstep_diagnostic_results.json'), 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    lines = []
    lines.append('Final full-step diagnostic-only summary (A1 + A2)')
    lines.append('===============================================')
    lines.append(f"overall_label   : {payload['summary']['overall_label']}")
    lines.append(f"overall_message : {payload['summary']['overall_message']}")
    lines.append('')
    if a1 is not None:
        lines.append('A1 scatter_only_step')
        lines.append(f"  label   : {a1['label']}")
        lines.append(f"  message : {a1['message']}")
        for k, v in a1['metrics'].items():
            lines.append(f'  {k:28s}: {v}')
        lines.append('')
    if a2 is not None:
        lines.append('A2 full_step_absorb_vs_survive')
        lines.append(f"  label   : {a2['label']}")
        lines.append(f"  message : {a2['message']}")
        for k, v in a2['metrics'].items():
            lines.append(f'  {k:28s}: {v}')
        lines.append('')
    lines.append('Decision rules:')
    lines.append('diagnostic message')
    lines.append('diagnostic message')
    lines.append('diagnostic message')
    lines.append('Recommended bond sweep if mismatch: 256 -> 512 -> 1024 (if feasible)')
    with open(os.path.join(args.outdir, 'fullstep_diagnostic_summary.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n' + payload['summary']['overall_label'])
    print(payload['summary']['overall_message'])
    print(f"Saved: {os.path.join(args.outdir, 'fullstep_diagnostic_results.json')}")
    print(f"Saved: {os.path.join(args.outdir, 'fullstep_diagnostic_summary.txt')}")
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
