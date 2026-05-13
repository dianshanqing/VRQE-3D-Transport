"""Run isolated diagnostic checks for transport predicates and status updates."""
from __future__ import annotations
import argparse
import json
import math
import os
from datetime import datetime
from typing import Dict, List, Tuple
from config import default_config
from unitary_circuit import _theta, _multi_controlled_ry, _mcx, _set_status_via_term

def _binom_se(p: float, n: int) -> float:
    if n <= 0:
        return 0.0
    p = max(0.0, min(1.0, float(p)))
    return math.sqrt(p * (1.0 - p) / float(n))

def _parse_csv_str(x: str, *, allowed=None) -> List[str]:
    out = [s.strip() for s in str(x).split(',') if s.strip()]
    if allowed is not None:
        bad = [s for s in out if s not in allowed]
        if bad:
            raise ValueError(f'invalid values: {bad}; allowed={sorted(allowed)}')
    return out

def _parse_csv_int(x: str) -> List[int]:
    return [int(s.strip()) for s in str(x).split(',') if s.strip()]

def _mat_probs(cfg, material: str, g: int) -> Tuple[float, float]:
    mats = {'concrete': cfg.mats.concrete, 'tungsten': cfg.mats.tungsten, 'air': cfg.mats.air}
    mat = mats[material]
    return (float(mat.p_int[g]), float(mat.p_abs[g]))

def _basis_index_prob(qc, state, qubits, bits_lsb: List[int]) -> float:
    qidx = [qc.find_bit(q).index for q in qubits]
    total = 0.0
    data = state.data
    for basis_idx, amp in enumerate(data):
        ok = True
        for idx, bit in zip(qidx, bits_lsb):
            if basis_idx >> idx & 1 != int(bit):
                ok = False
                break
        if ok:
            total += float((amp.conjugate() * amp).real)
    return total

def _run_mps_sample_prob(qc, qubits, bits_lsb: List[int], *, shots: int, seed: int, aer_safe_basis: bool, aer_threads: int, mps_max_bond: int, mps_trunc: float) -> Dict[str, float]:
    from qiskit import ClassicalRegister, QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    qcm = QuantumCircuit(*qc.qregs, name=qc.name + '_meas')
    qcm.compose(qc, inplace=True)
    creg = ClassicalRegister(len(qubits), 'c')
    qcm.add_register(creg)
    for i, q in enumerate(qubits):
        qcm.measure(q, creg[i])
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
    target_key = ''.join((str(int(b)) for b in reversed(bits_lsb)))
    hit = int(counts.get(target_key, 0))
    p = float(hit) / float(shots)
    se = _binom_se(p, int(shots))
    return {'target_key': target_key, 'hits': hit, 'shots': int(shots), 'p_mps': p, 'se_mps': se}

def _material_flag_regs(qc, material: str):
    from qiskit import QuantumRegister
    qmat = QuantumRegister(3, 'mat')
    qc.add_register(qmat)
    idx = {'concrete': 0, 'tungsten': 1, 'air': 2}[material]
    qc.x(qmat[idx])
    return qmat

def _group_flag_regs(qc, g: int):
    from qiskit import QuantumRegister
    qg = QuantumRegister(1, 'g')
    qw0 = QuantumRegister(1, 'wg0')
    qc.add_register(qg)
    qc.add_register(qw0)
    if int(g) == 0:
        qc.x(qw0[0])
    else:
        qc.x(qg[0])
    return (qg, qw0)

def _add_common_guards(qc):
    from qiskit import QuantumRegister
    qnb = QuantumRegister(1, 'nb')
    qalive = QuantumRegister(1, 'alive0')
    qc.add_register(qnb)
    qc.add_register(qalive)
    qc.x(qnb[0])
    qc.x(qalive[0])
    return (qnb, qalive)

def _apply_single_material_interaction_rotation(qc, *, material: str, g: int, mat, qg, wg0, nb, alive, target):
    cfg = default_config()
    p_int, _ = _mat_probs(cfg, material, g)
    theta = _theta(p_int)
    if material == 'concrete':
        ctrl = [mat[0], wg0[0] if int(g) == 0 else qg[0], nb[0], alive[0]]
    elif material == 'tungsten':
        ctrl = [mat[1], wg0[0] if int(g) == 0 else qg[0], nb[0], alive[0]]
    else:
        ctrl = [mat[2], wg0[0] if int(g) == 0 else qg[0], nb[0], alive[0]]
    _multi_controlled_ry(qc, theta, ctrl, target)

def _apply_single_material_absorb_rotation(qc, *, material: str, g: int, mat, qg, wg0, int_guard, target):
    cfg = default_config()
    _, p_abs = _mat_probs(cfg, material, g)
    theta = _theta(p_abs)
    if material == 'concrete':
        ctrl = [mat[0], wg0[0] if int(g) == 0 else qg[0], int_guard]
    elif material == 'tungsten':
        ctrl = [mat[1], wg0[0] if int(g) == 0 else qg[0], int_guard]
    else:
        ctrl = [mat[2], wg0[0] if int(g) == 0 else qg[0], int_guard]
    _multi_controlled_ry(qc, theta, ctrl, target)

def build_interaction_coin_diag(material: str, g: int):
    from qiskit import QuantumCircuit, QuantumRegister
    qc = QuantumCircuit(name=f'diag_int_{material}_g{g}')
    mat = _material_flag_regs(qc, material)
    qg, wg0 = _group_flag_regs(qc, g)
    nb, alive = _add_common_guards(qc)
    qci = QuantumRegister(1, 'cI')
    qc.add_register(qci)
    _apply_single_material_interaction_rotation(qc, material=material, g=g, mat=mat, qg=qg, wg0=wg0, nb=nb, alive=alive, target=qci[0])
    return (qc, [qci[0]], [1])

def build_absorb_status_write_diag(p_abs: float):
    from qiskit import QuantumCircuit, QuantumRegister
    qc = QuantumCircuit(name=f'diag_write_p{p_abs:.4f}')
    qstatus = QuantumRegister(2, 'status')
    qterm = QuantumRegister(1, 'term')
    qpred = QuantumRegister(1, 'pred')
    qc.add_register(qstatus)
    qc.add_register(qterm)
    qc.add_register(qpred)
    qc.ry(_theta(p_abs), qpred[0])
    _set_status_via_term(qc, qstatus, qterm[0], qpred[0], 2)
    return (qc, list(qstatus), [0, 1])

def build_local_absorb_chain_diag(material: str, g: int):
    from qiskit import QuantumCircuit, QuantumRegister
    qc = QuantumCircuit(name=f'diag_chain_{material}_g{g}')
    mat = _material_flag_regs(qc, material)
    qg, wg0 = _group_flag_regs(qc, g)
    nb, alive = _add_common_guards(qc)
    qstatus = QuantumRegister(2, 'status')
    qci = QuantumRegister(1, 'cI')
    qtA = QuantumRegister(1, 'tA')
    qca = QuantumRegister(1, 'cA')
    qevent = QuantumRegister(1, 'wAbsEvt')
    qterm = QuantumRegister(1, 'term')
    qc.add_register(qstatus)
    qc.add_register(qci)
    qc.add_register(qtA)
    qc.add_register(qca)
    qc.add_register(qevent)
    qc.add_register(qterm)
    _apply_single_material_interaction_rotation(qc, material=material, g=g, mat=mat, qg=qg, wg0=wg0, nb=nb, alive=alive, target=qci[0])
    _mcx(qc, [qci[0], nb[0], alive[0]], qtA[0])
    qc.cx(qtA[0], qci[0])
    _apply_single_material_absorb_rotation(qc, material=material, g=g, mat=mat, qg=qg, wg0=wg0, int_guard=qtA[0], target=qca[0])
    _mcx(qc, [qtA[0], qca[0]], qevent[0])
    _set_status_via_term(qc, qstatus, qterm[0], qevent[0], 2)
    return (qc, list(qstatus), [0, 1])

def _exact_prob(qc, qubits, bits_lsb: List[int]) -> float:
    from qiskit.quantum_info import Statevector
    sv = Statevector.from_instruction(qc)
    return _basis_index_prob(qc, sv, qubits, bits_lsb)

def _judge_case(*, target: float, p_exact: float, p_mps: float, se_mps: float) -> Dict[str, object]:
    exact_logic_ok = abs(float(p_exact) - float(target)) <= 1e-10
    z_mps_vs_exact = 0.0 if se_mps <= 0 else (float(p_mps) - float(p_exact)) / float(se_mps)
    z_mps_vs_target = 0.0 if se_mps <= 0 else (float(p_mps) - float(target)) / float(se_mps)
    if not exact_logic_ok:
        label = 'EXACT_LOGIC_MISMATCH'
    elif abs(z_mps_vs_exact) > 5.0:
        label = 'MPS_BIAS'
    else:
        label = 'PASS'
    return {'target': float(target), 'p_exact': float(p_exact), 'p_mps': float(p_mps), 'se_mps': float(se_mps), 'delta_exact_target': float(p_exact - target), 'delta_mps_target': float(p_mps - target), 'z_mps_vs_exact': float(z_mps_vs_exact), 'z_mps_vs_target': float(z_mps_vs_target), 'label': label}

def run_one_case(kind: str, *, material: str, g: int, shots: int, seed: int, aer_safe_basis: bool, aer_threads: int, mps_max_bond: int, mps_trunc: float):
    cfg = default_config()
    p_int, p_abs = _mat_probs(cfg, material, g)
    if kind == 'interaction_coin':
        qc, qubits, bits = build_interaction_coin_diag(material, g)
        target = p_int
    elif kind == 'status_write':
        qc, qubits, bits = build_absorb_status_write_diag(p_abs)
        target = p_abs
    elif kind == 'local_chain':
        qc, qubits, bits = build_local_absorb_chain_diag(material, g)
        target = p_int * p_abs
    else:
        raise ValueError(f'unknown kind={kind}')
    p_exact = _exact_prob(qc, qubits, bits)
    smp = _run_mps_sample_prob(qc, qubits, bits, shots=shots, seed=seed, aer_safe_basis=aer_safe_basis, aer_threads=aer_threads, mps_max_bond=mps_max_bond, mps_trunc=mps_trunc)
    judge = _judge_case(target=target, p_exact=p_exact, p_mps=smp['p_mps'], se_mps=smp['se_mps'])
    return {'kind': kind, 'material': material, 'g': int(g), 'target_name': {'interaction_coin': 'p_int', 'status_write': 'p_abs', 'local_chain': 'p_int*p_abs'}[kind], 'target': float(target), 'result': judge, 'sample': smp, 'n_qubits': int(qc.num_qubits), 'depth': int(qc.depth()), 'count_ops': {str(k): int(v) for k, v in qc.count_ops().items()}}

def summarize(results: List[Dict[str, object]]) -> Dict[str, object]:
    by_kind: Dict[str, List[Dict[str, object]]] = {}
    for r in results:
        by_kind.setdefault(str(r['kind']), []).append(r)

    def _labels(kind: str) -> List[str]:
        return [str(x['result']['label']) for x in by_kind.get(kind, [])]
    interaction_labels = _labels('interaction_coin')
    write_labels = _labels('status_write')
    chain_labels = _labels('local_chain')
    if any((lb == 'EXACT_LOGIC_MISMATCH' for lb in interaction_labels)):
        overall = 'ROOT_CAUSE_1A_INTERACTION_COIN_LOGIC'
        message = 'diagnostic message'
    elif any((lb == 'MPS_BIAS' for lb in interaction_labels)):
        overall = 'ROOT_CAUSE_1B_INTERACTION_COIN_MPS_BIAS'
        message = 'diagnostic message'
    elif any((lb == 'EXACT_LOGIC_MISMATCH' for lb in write_labels)):
        overall = 'ROOT_CAUSE_2A_STATUS_WRITE_LOGIC'
        message = 'diagnostic message'
    elif any((lb == 'MPS_BIAS' for lb in write_labels)):
        overall = 'ROOT_CAUSE_2B_STATUS_WRITE_MPS_BIAS'
        message = 'diagnostic message'
    elif any((lb == 'EXACT_LOGIC_MISMATCH' for lb in chain_labels)):
        overall = 'ROOT_CAUSE_3A_LOCAL_CHAIN_LOGIC'
        message = 'diagnostic message'
    elif any((lb == 'MPS_BIAS' for lb in chain_labels)):
        overall = 'ROOT_CAUSE_3B_LOCAL_CHAIN_MPS_BIAS'
        message = 'diagnostic message'
    else:
        overall = 'ROOT_CAUSE_3C_GLOBAL_LIGHTWEIGHT_FLOOR_REMAINS'
        message = 'diagnostic message'
    return {'overall_label': overall, 'overall_message': message}

def build_argparser():
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument('--materials', type=str, default='concrete,tungsten,air')
    p.add_argument('--groups', type=str, default='0,1')
    p.add_argument('--shots', type=int, default=200000)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--aer-safe-basis', action='store_true')
    p.add_argument('--aer-threads', type=int, default=1)
    p.add_argument('--mps-max-bond', type=int, default=256)
    p.add_argument('--mps-trunc', type=float, default=1e-12)
    p.add_argument('--outdir', type=str, default='outputs/diag_root_causes')
    return p

def main(argv=None):
    args = build_argparser().parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)
    materials = _parse_csv_str(args.materials, allowed={'concrete', 'tungsten', 'air'})
    groups = _parse_csv_int(args.groups)
    cfg = default_config()
    groups = [g for g in groups if 0 <= int(g) < int(cfg.groups.n_groups)]
    if not groups:
        raise SystemExit('No valid groups selected.')
    print('=== Diagnostic-only root-cause tests ===')
    print(f'materials={materials}')
    print(f'groups={groups}')
    print(f'shots={int(args.shots)} seed={int(args.seed)} safe_basis={bool(args.aer_safe_basis)} mps_max_bond={int(args.mps_max_bond)} mps_trunc={float(args.mps_trunc)}')
    results: List[Dict[str, object]] = []
    for material in materials:
        for g in groups:
            print(f'\n--- material={material} g={g} ---')
            for kind in ('interaction_coin', 'status_write', 'local_chain'):
                r = run_one_case(kind, material=material, g=int(g), shots=int(args.shots), seed=int(args.seed), aer_safe_basis=bool(args.aer_safe_basis), aer_threads=int(args.aer_threads), mps_max_bond=int(args.mps_max_bond), mps_trunc=float(args.mps_trunc))
                results.append(r)
                rr = r['result']
                print(f"[{kind}] target={rr['target']:.6f} exact={rr['p_exact']:.6f} mps={rr['p_mps']:.6f} se={rr['se_mps']:.3e} label={rr['label']}")
    summary = summarize(results)
    payload = {'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'materials': materials, 'groups': groups, 'shots': int(args.shots), 'seed': int(args.seed), 'aer_safe_basis': bool(args.aer_safe_basis), 'aer_threads': int(args.aer_threads), 'mps_max_bond': int(args.mps_max_bond), 'mps_trunc': float(args.mps_trunc), 'results': results, 'summary': summary}
    with open(os.path.join(args.outdir, 'diagnostic_results.json'), 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    txt_lines = []
    txt_lines.append('Diagnostic-only root-cause summary')
    txt_lines.append('=================================')
    txt_lines.append(f"overall_label   : {summary['overall_label']}")
    txt_lines.append(f"overall_message : {summary['overall_message']}")
    txt_lines.append('')
    for r in results:
        rr = r['result']
        txt_lines.append(f"{r['kind']:16s} material={r['material']:8s} g={r['g']} target={rr['target']:.6f} exact={rr['p_exact']:.6f} mps={rr['p_mps']:.6f} se={rr['se_mps']:.3e} label={rr['label']} z(mps-exact)={rr['z_mps_vs_exact']:.2f}")
    txt_lines.append('')
    txt_lines.append('Decision rules:')
    txt_lines.append('diagnostic message')
    txt_lines.append('diagnostic message')
    txt_lines.append('diagnostic message')
    txt_lines.append('diagnostic message')
    txt_lines.append('diagnostic message')
    txt_lines.append('  - all pass but production still low-> full-step scatter/move/history compression floor')
    with open(os.path.join(args.outdir, 'diagnostic_summary.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(txt_lines) + '\n')
    print('\n' + summary['overall_label'])
    print(summary['overall_message'])
    print(f"Saved: {os.path.join(args.outdir, 'diagnostic_results.json')}")
    print(f"Saved: {os.path.join(args.outdir, 'diagnostic_summary.txt')}")
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
