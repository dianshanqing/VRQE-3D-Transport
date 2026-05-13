"""Qiskit construction of the reversible 3D transport circuit."""


from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import RYGate
from qiskit.circuit.library import UnitaryGate

from config import TransportConfig, validate_config

def _ceil_log2(n: int) -> int:
    return max(1, int(np.ceil(np.log2(max(n, 2)))))

def _set_reg(qc: QuantumCircuit, reg: QuantumRegister, value: int) -> None:

    for i in range(len(reg)):
        if (value >> i) & 1:
            qc.x(reg[i])

def _mcx(qc: QuantumCircuit, ctrls, tgt) -> None:

    qc.mcx(ctrls, tgt, mode="noancilla")

def _theta(p: float) -> float:
    p = float(p)
    p = min(max(p, 0.0), 1.0)
    return 2.0 * float(np.arcsin(np.sqrt(p)))

def _multi_controlled_ry(qc: QuantumCircuit, theta: float, controls, target) -> None:
    if abs(theta) < 1e-15:
        return
    qc.append(RYGate(theta).control(len(controls)), list(controls) + [target])

def _const_mask(value: int, nbits: int):

    bits = [(value >> i) & 1 for i in range(nbits)]
    return [0 if b == 1 else 1 for b in bits]

def _x_mask(qc: QuantumCircuit, qubits, mask):
    for qb, m in zip(qubits, mask):
        if m:
            qc.x(qb)

def _eq_const_into(qc: QuantumCircuit, reg: QuantumRegister, const: int, out_qubit) -> None:

    mask = _const_mask(const, len(reg))
    _x_mask(qc, reg, mask)
    _mcx(qc, list(reg), out_qubit)
    _x_mask(qc, reg, mask)

def _xor_into(qc: QuantumCircuit, a, b, out) -> None:
    qc.cx(a, out)
    qc.cx(b, out)

def _and_into(qc: QuantumCircuit, a, b, out) -> None:
    _mcx(qc, [a, b], out)

def _and_not_into(qc: QuantumCircuit, a, b, out) -> None:
    qc.x(b)
    _mcx(qc, [a, b], out)
    qc.x(b)

def _or_into(qc: QuantumCircuit, a, b, out) -> None:

    qc.x(a)
    qc.x(b)
    _mcx(qc, [a, b], out)
    qc.x(a)
    qc.x(b)
    qc.x(out)

def _unor_into(qc: QuantumCircuit, a, b, out) -> None:

    qc.x(out)
    qc.x(a)
    qc.x(b)
    _mcx(qc, [a, b], out)
    qc.x(a)
    qc.x(b)

def _dft6_embedded() -> np.ndarray:

    n = 6
    w = np.exp(2j * np.pi / n)
    F = np.array([[w ** (i * j) for j in range(n)] for i in range(n)], dtype=complex) / np.sqrt(n)
    U = np.eye(8, dtype=complex)
    U[:6, :6] = F
    return U

_U6 = UnitaryGate(_dft6_embedded(), label="U6")

def _ctrl_inc_mod(qc: QuantumCircuit, reg: QuantumRegister, ctrl) -> None:

    for i in range(len(reg) - 1, 0, -1):
        _mcx(qc, [ctrl] + [reg[j] for j in range(i)], reg[i])
    qc.cx(ctrl, reg[0])

def _ctrl_dec_mod(qc: QuantumCircuit, reg: QuantumRegister, ctrl) -> None:

    for qb in reg:
        qc.x(qb)
    _ctrl_inc_mod(qc, reg, ctrl)
    for qb in reg:
        qc.x(qb)

def _set_status_via_term(qc: QuantumCircuit, status: QuantumRegister, term, predicate, code: int) -> None:

    qc.x(status[0]); qc.x(status[1])
    _mcx(qc, [status[0], status[1], predicate], term)
    qc.x(status[0]); qc.x(status[1])

    if code & 0b01:
        qc.cx(term, status[0])
    if code & 0b10:
        qc.cx(term, status[1])

    if code == 0b11:
        _mcx(qc, [status[0], status[1], predicate], term)
    elif code == 0b01:
        qc.x(status[1])
        _mcx(qc, [status[0], status[1], predicate], term)
        qc.x(status[1])
    elif code == 0b10:
        qc.x(status[0])
        _mcx(qc, [status[0], status[1], predicate], term)
        qc.x(status[0])
    else:
        raise ValueError("invalid status code")

def _compute_in_roi(qc: QuantumCircuit, x, y, z, work9, cfg: TransportConfig):

    g = cfg.geom
    w0, w1, w2, w3, w4, w5, w6, w7, w8 = work9

    _eq_const_into(qc, x, g.detector_x, w0)

    _eq_const_into(qc, y, g.roi_y0, w1)
    _eq_const_into(qc, y, g.roi_y1, w2)
    _xor_into(qc, w1, w2, w3)

    _eq_const_into(qc, z, g.roi_z0, w4)
    _eq_const_into(qc, z, g.roi_z1, w5)
    _xor_into(qc, w4, w5, w6)

    _and_into(qc, w0, w3, w7)
    _and_into(qc, w7, w6, w8)

    return w8

def _uncompute_in_roi(qc: QuantumCircuit, x, y, z, work9, cfg: TransportConfig):
    g = cfg.geom
    w0, w1, w2, w3, w4, w5, w6, w7, w8 = work9

    _and_into(qc, w7, w6, w8)
    _and_into(qc, w0, w3, w7)

    _xor_into(qc, w4, w5, w6)
    _eq_const_into(qc, z, g.roi_z1, w5)
    _eq_const_into(qc, z, g.roi_z0, w4)

    _xor_into(qc, w1, w2, w3)
    _eq_const_into(qc, y, g.roi_y1, w2)
    _eq_const_into(qc, y, g.roi_y0, w1)

    _eq_const_into(qc, x, g.detector_x, w0)

def _apply_boundary_check(qc: QuantumCircuit, qx, qy, qz, qstatus, term_boundary, w_boundary, boundary_work6, boundary_aux4, cfg: TransportConfig):

    g = cfg.geom
    ex0, exm, ey0, eym, ez0, ezm = boundary_work6
    t01, t23, t45, t0123 = boundary_aux4

    _eq_const_into(qc, qx, 0, ex0)
    _eq_const_into(qc, qx, int(g.nx) - 1, exm)
    _eq_const_into(qc, qy, 0, ey0)
    _eq_const_into(qc, qy, int(g.ny) - 1, eym)
    _eq_const_into(qc, qz, 0, ez0)
    _eq_const_into(qc, qz, int(g.nz) - 1, ezm)

    _or_into(qc, ex0, exm, t01)
    _or_into(qc, ey0, eym, t23)
    _or_into(qc, ez0, ezm, t45)
    _or_into(qc, t01, t23, t0123)
    _or_into(qc, t0123, t45, w_boundary)

    _set_status_via_term(qc, qstatus, term_boundary, w_boundary, 0b11)

    _unor_into(qc, t0123, t45, w_boundary)
    _unor_into(qc, t01, t23, t0123)
    _unor_into(qc, ez0, ezm, t45)
    _unor_into(qc, ey0, eym, t23)
    _unor_into(qc, ex0, exm, t01)

    _eq_const_into(qc, qz, int(g.nz) - 1, ezm)
    _eq_const_into(qc, qz, 0, ez0)
    _eq_const_into(qc, qy, int(g.ny) - 1, eym)
    _eq_const_into(qc, qy, 0, ey0)
    _eq_const_into(qc, qx, int(g.nx) - 1, exm)
    _eq_const_into(qc, qx, 0, ex0)

def _compute_boundary_flag(qc: QuantumCircuit, qx, qy, qz, w_boundary, boundary_work6, boundary_aux4, cfg: TransportConfig):

    g = cfg.geom
    ex0, exm, ey0, eym, ez0, ezm = boundary_work6
    t01, t23, t45, t0123 = boundary_aux4

    _eq_const_into(qc, qx, 0, ex0)
    _eq_const_into(qc, qx, int(g.nx) - 1, exm)
    _eq_const_into(qc, qy, 0, ey0)
    _eq_const_into(qc, qy, int(g.ny) - 1, eym)
    _eq_const_into(qc, qz, 0, ez0)
    _eq_const_into(qc, qz, int(g.nz) - 1, ezm)

    _or_into(qc, ex0, exm, t01)
    _or_into(qc, ey0, eym, t23)
    _or_into(qc, ez0, ezm, t45)
    _or_into(qc, t01, t23, t0123)
    _or_into(qc, t0123, t45, w_boundary)

def _uncompute_boundary_flag(qc: QuantumCircuit, qx, qy, qz, w_boundary, boundary_work6, boundary_aux4, cfg: TransportConfig):

    g = cfg.geom
    ex0, exm, ey0, eym, ez0, ezm = boundary_work6
    t01, t23, t45, t0123 = boundary_aux4

    _unor_into(qc, t0123, t45, w_boundary)
    _unor_into(qc, t01, t23, t0123)
    _unor_into(qc, ez0, ezm, t45)
    _unor_into(qc, ey0, eym, t23)
    _unor_into(qc, ex0, exm, t01)

    _eq_const_into(qc, qz, int(g.nz) - 1, ezm)
    _eq_const_into(qc, qz, 0, ez0)
    _eq_const_into(qc, qy, int(g.ny) - 1, eym)
    _eq_const_into(qc, qy, 0, ey0)
    _eq_const_into(qc, qx, int(g.nx) - 1, exm)
    _eq_const_into(qc, qx, 0, ex0)

def _apply_detector_check(qc: QuantumCircuit, qx, qy, qz, qstatus, term_detect, roi_work, cfg: TransportConfig):

    w_roi = _compute_in_roi(qc, qx, qy, qz, roi_work, cfg)
    _set_status_via_term(qc, qstatus, term_detect, w_roi, 0b01)
    _uncompute_in_roi(qc, qx, qy, qz, roi_work, cfg)

def _compute_in_slab_and_duct(qc: QuantumCircuit, x, y, z, work10, cfg: TransportConfig):

    g = cfg.geom
    w0, w1, w2, w3, w4, w5, w6, w7, w8, w9 = work10

    _eq_const_into(qc, x, g.slab_x0, w0)
    _eq_const_into(qc, x, g.slab_x1, w1)
    _xor_into(qc, w0, w1, w2)

    _eq_const_into(qc, y, g.duct_y0, w3)
    _eq_const_into(qc, y, g.duct_y1, w4)
    _xor_into(qc, w3, w4, w5)

    _eq_const_into(qc, z, g.duct_z0, w6)
    _eq_const_into(qc, z, g.duct_z1, w7)
    _xor_into(qc, w6, w7, w8)

    _and_into(qc, w5, w8, w9)
    return w2, w9

def _uncompute_in_slab_and_duct(qc: QuantumCircuit, x, y, z, work10, cfg: TransportConfig):
    g = cfg.geom
    w0, w1, w2, w3, w4, w5, w6, w7, w8, w9 = work10

    _and_into(qc, w5, w8, w9)

    _xor_into(qc, w6, w7, w8)
    _eq_const_into(qc, z, g.duct_z1, w7)
    _eq_const_into(qc, z, g.duct_z0, w6)

    _xor_into(qc, w3, w4, w5)
    _eq_const_into(qc, y, g.duct_y1, w4)
    _eq_const_into(qc, y, g.duct_y0, w3)

    _xor_into(qc, w0, w1, w2)
    _eq_const_into(qc, x, g.slab_x1, w1)
    _eq_const_into(qc, x, g.slab_x0, w0)

@dataclass(frozen=True)
class KernelProfile:

    name: str = "manual"
    use_dir_tape: bool = False
    use_absorb_tape: bool = False
    use_term_tape: bool = False
    use_scatter_tape: bool = False
    full_history_mode: bool = False

    drain_interaction_ledger: bool = True
    early_absorb_erase: bool = True
    compact_interaction_ledger: bool = True
    cleanup_alive_ledger: bool = True

def resolve_kernel_profile(
    *,
    kernel_profile: Optional[Union[str, KernelProfile]] = None,
    use_dir_tape: bool = False,
    use_absorb_tape: bool = False,
    use_term_tape: bool = False,
    use_scatter_tape: bool = False,
    full_history_mode: bool = False,
) -> KernelProfile:

    if isinstance(kernel_profile, KernelProfile):
        prof = kernel_profile
    else:
        name = str(kernel_profile or "").strip().lower()
        if name in ("", "manual"):
            prof = KernelProfile(
                name="manual",
                use_dir_tape=bool(use_dir_tape),
                use_absorb_tape=bool(use_absorb_tape),
                use_term_tape=bool(use_term_tape),
                use_scatter_tape=bool(use_scatter_tape),
                full_history_mode=bool(full_history_mode),
                drain_interaction_ledger=True,
                early_absorb_erase=not bool(full_history_mode),
                compact_interaction_ledger=not bool(full_history_mode),
                cleanup_alive_ledger=not bool(full_history_mode),
            )
        elif name in ("default_light", "compressed"):
            prof = KernelProfile(
                name=("compressed" if name == "compressed" else "default_light"),
                use_dir_tape=True,
                use_absorb_tape=True,
                use_term_tape=True,
                use_scatter_tape=False,
                full_history_mode=False,
                drain_interaction_ledger=True,
                early_absorb_erase=True,
                compact_interaction_ledger=True,
                cleanup_alive_ledger=True,
            )
        elif name in ("baseline", "light_baseline"):
            prof = KernelProfile(
                name="baseline",
                use_dir_tape=True,
                use_absorb_tape=True,
                use_term_tape=True,
                use_scatter_tape=False,
                full_history_mode=False,
                drain_interaction_ledger=False,
                early_absorb_erase=False,
                compact_interaction_ledger=False,
                cleanup_alive_ledger=False,
            )
        elif name == "full_history":
            prof = KernelProfile(
                name="full_history",
                use_dir_tape=True,
                use_absorb_tape=True,
                use_term_tape=True,
                use_scatter_tape=True,
                full_history_mode=True,
                drain_interaction_ledger=True,
                early_absorb_erase=False,
                compact_interaction_ledger=False,
                cleanup_alive_ledger=False,
            )
        else:
            raise ValueError(f"unknown kernel_profile={kernel_profile!r}")

    return KernelProfile(
        name=prof.name,
        use_dir_tape=bool(prof.use_dir_tape),
        use_absorb_tape=bool(prof.use_absorb_tape),
        use_term_tape=bool(prof.use_term_tape),
        use_scatter_tape=bool(prof.use_scatter_tape and prof.use_dir_tape),
        full_history_mode=bool(prof.full_history_mode),
        drain_interaction_ledger=bool(prof.drain_interaction_ledger),
        early_absorb_erase=bool(prof.early_absorb_erase),
        compact_interaction_ledger=bool(prof.compact_interaction_ledger),
        cleanup_alive_ledger=bool(prof.cleanup_alive_ledger),
    )

def compile_roi_predicate(qc: QuantumCircuit, x, y, z, work9, cfg: TransportConfig):
    return _compute_in_roi(qc, x, y, z, work9, cfg)

def uncompile_roi_predicate(qc: QuantumCircuit, x, y, z, work9, cfg: TransportConfig):
    return _uncompute_in_roi(qc, x, y, z, work9, cfg)

def compile_boundary_predicate(qc: QuantumCircuit, qx, qy, qz, w_boundary, boundary_work6, boundary_aux4, cfg: TransportConfig):
    return _compute_boundary_flag(qc, qx, qy, qz, w_boundary, boundary_work6, boundary_aux4, cfg)

def uncompile_boundary_predicate(qc: QuantumCircuit, qx, qy, qz, w_boundary, boundary_work6, boundary_aux4, cfg: TransportConfig):
    return _uncompute_boundary_flag(qc, qx, qy, qz, w_boundary, boundary_work6, boundary_aux4, cfg)

def compile_material_predicates(qc: QuantumCircuit, x, y, z, work10, cfg: TransportConfig):
    return _compute_in_slab_and_duct(qc, x, y, z, work10, cfg)

def uncompile_material_predicates(qc: QuantumCircuit, x, y, z, work10, cfg: TransportConfig):
    return _uncompute_in_slab_and_duct(qc, x, y, z, work10, cfg)

def terminal_write(qc: QuantumCircuit, status: QuantumRegister, term, predicate, code: int) -> None:
    _set_status_via_term(qc, status, term, predicate, code)

def build_unitary_circuit_qiskit(
    cfg: TransportConfig,
    n_steps: int,
    seed_bits: int = 33,
    prng_k: int = 8,
    unitary_matched_energy: bool = False,
    use_dir_tape: bool = False,
    use_absorb_tape: bool = False,
    use_term_tape: bool = False,
    use_scatter_tape: bool = False,
    full_history_mode: bool = False,
    kernel_profile: Optional[Union[str, KernelProfile]] = None,
    disable_detector: bool = False,
    disable_boundary: bool = False,
) -> Tuple[QuantumCircuit, Dict[str, int]]:

    validate_config(cfg)
    geom = cfg.geom

    profile = resolve_kernel_profile(
        kernel_profile=kernel_profile,
        use_dir_tape=use_dir_tape,
        use_absorb_tape=use_absorb_tape,
        use_term_tape=use_term_tape,
        use_scatter_tape=use_scatter_tape,
        full_history_mode=full_history_mode,
    )
    use_dir_tape = profile.use_dir_tape
    use_absorb_tape = profile.use_absorb_tape
    use_term_tape = profile.use_term_tape
    use_scatter_tape = profile.use_scatter_tape
    full_history_mode = profile.full_history_mode

    xbits = _ceil_log2(geom.nx)
    ybits = _ceil_log2(geom.ny)
    zbits = _ceil_log2(geom.nz)

    qx = QuantumRegister(xbits, "x")
    qy = QuantumRegister(ybits, "y")
    qz = QuantumRegister(zbits, "z")
    qg = QuantumRegister(1, "g")
    qdir = QuantumRegister(3, "dir")

    qdirT = QuantumRegister(3 * n_steps, "dirT") if use_dir_tape else None

    qcI = QuantumRegister(n_steps, "cI")

    qAbs = QuantumRegister(n_steps, "cA") if use_absorb_tape else None

    qTermB = QuantumRegister(n_steps, "tB") if use_term_tape else None
    qTermD = QuantumRegister(n_steps, "tD") if use_term_tape else None
    qTermA = QuantumRegister(n_steps, "tA") if use_term_tape else None

    qScat = QuantumRegister(n_steps, "tS") if use_scatter_tape else None

    qterm = (QuantumRegister(1, "term") if (not use_term_tape) else None)

    qstatus = QuantumRegister(2, "status")

    qcoin = (QuantumRegister(1, "coin") if (not use_absorb_tape) else None)

    qwork = QuantumRegister(17, "work")

    regs = [qx, qy, qz, qg, qdir]
    if use_dir_tape:
        regs += [qdirT]
    regs += [qcI]
    if use_absorb_tape:
        regs += [qAbs]
    if use_term_tape:
        regs += [qTermB, qTermD, qTermA]
    if use_scatter_tape:
        regs += [qScat]
    if qterm is not None:
        regs += [qterm]
    regs += [qstatus]
    if qcoin is not None:
        regs += [qcoin]
    regs += [qwork]
    qc = QuantumCircuit(*regs, name="A_paper1")

    _set_reg(qc, qx, cfg.source.x0)
    _set_reg(qc, qy, cfg.source.y0)
    _set_reg(qc, qz, cfg.source.z0)
    _set_reg(qc, qg, cfg.source.g0)
    _set_reg(qc, qdir, cfg.source.dir0)

    mats = cfg.mats
    th_int = {
        ("concrete", 0): _theta(mats.concrete.p_int[0]),
        ("concrete", 1): _theta(mats.concrete.p_int[min(1, cfg.groups.n_groups - 1)]),
        ("tungsten", 0): _theta(mats.tungsten.p_int[0]),
        ("tungsten", 1): _theta(mats.tungsten.p_int[min(1, cfg.groups.n_groups - 1)]),
        ("air", 0): _theta(mats.air.p_int[0]),
        ("air", 1): _theta(mats.air.p_int[min(1, cfg.groups.n_groups - 1)]),
    }
    th_abs = {
        ("concrete", 0): _theta(mats.concrete.p_abs[0]),
        ("concrete", 1): _theta(mats.concrete.p_abs[min(1, cfg.groups.n_groups - 1)]),
        ("tungsten", 0): _theta(mats.tungsten.p_abs[0]),
        ("tungsten", 1): _theta(mats.tungsten.p_abs[min(1, cfg.groups.n_groups - 1)]),
        ("air", 0): _theta(mats.air.p_abs[0]),
        ("air", 1): _theta(mats.air.p_abs[min(1, cfg.groups.n_groups - 1)]),
    }

    w_boundary = qwork[0]
    w_g0 = qwork[1]
    roi_work = [qwork[i] for i in range(2, 11)]
    slab_work = [qwork[i] for i in range(2, 12)]
    w_is_air = qwork[12]
    w_is_tung = qwork[13]
    w_dir_pred = qwork[14]
    w_abs_event = qwork[15]
    w_abs_term = qwork[16]

    def build_g0():

        qc.x(qg[0])
        qc.cx(qg[0], w_g0)
        qc.x(qg[0])

    def unbuild_g0():
        qc.x(qg[0])
        qc.cx(qg[0], w_g0)
        qc.x(qg[0])

    x_max = (1 << xbits) - 1
    y_max = (1 << ybits) - 1
    z_max = (1 << zbits) - 1

    def _dist_to_interval(p: int, lo: int, hi: int) -> int:
        if p < lo:
            return int(lo - p)
        if p > hi:
            return int(p - hi)
        return 0

    dx = abs(int(cfg.geom.detector_x) - int(cfg.source.x0))
    dy = _dist_to_interval(int(cfg.source.y0), int(cfg.geom.roi_y0), int(cfg.geom.roi_y1))
    dz = _dist_to_interval(int(cfg.source.z0), int(cfg.geom.roi_z0), int(cfg.geom.roi_z1))
    detector_min_step = int(dx + dy + dz)

    detector_enabled = (not disable_detector) and (int(n_steps) >= int(detector_min_step))

    def compile_step_kernel(t: int) -> None:

        if not disable_boundary:

            _apply_boundary_check(qc, qx, qy, qz, qstatus, (qTermB[t] if (use_term_tape and (qTermB is not None)) else qterm[0]), w_boundary, roi_work[:6], [roi_work[6], roi_work[7], roi_work[8], qwork[11]], cfg)

        if detector_enabled and (t >= detector_min_step):
            _apply_detector_check(qc, qx, qy, qz, qstatus, (qTermD[t] if (use_term_tape and (qTermD is not None)) else qterm[0]), roi_work, cfg)

        _compute_boundary_flag(
            qc, qx, qy, qz,
            w_boundary,
            roi_work[:6],
            [roi_work[6], roi_work[7], roi_work[8], qwork[11]],
            cfg,
        )
        qc.x(w_boundary)
        if use_term_tape and (qTermB is not None):
            qc.cx(w_boundary, qTermB[t])
            nb_guard_ctrl = qTermB[t]
            nb_guard_for_move = qTermB[t]

            qc.x(w_boundary)
            _uncompute_boundary_flag(
                qc, qx, qy, qz,
                w_boundary,
                roi_work[:6],
                [roi_work[6], roi_work[7], roi_work[8], qwork[11]],
                cfg,
            )
        else:
            nb_guard_ctrl = w_boundary
            nb_guard_for_move = None

        step_alive_guard = None
        if use_term_tape and (qTermD is not None):
            qc.x(qstatus[0]); qc.x(qstatus[1])
            _mcx(qc, [qstatus[0], qstatus[1]], qTermD[t])
            qc.x(qstatus[0]); qc.x(qstatus[1])
            step_alive_guard = qTermD[t]

        w_in_slab, w_in_duct = _compute_in_slab_and_duct(qc, qx, qy, qz, slab_work, cfg)
        _and_into(qc, w_in_slab, w_in_duct, w_is_air)
        _and_not_into(qc, w_in_slab, w_in_duct, w_is_tung)

        build_g0()

        int_coin_ledger_mode = bool(step_alive_guard is not None)

        def _apply_interaction_coin_rotations(theta_sign: float = +1.0):
            if int_coin_ledger_mode:

                ctrl_common = [nb_guard_ctrl, step_alive_guard]
                qc.x(w_in_slab)
                _multi_controlled_ry(qc, theta_sign * th_int[("concrete", 0)], [w_in_slab, w_g0] + ctrl_common, qcI[t])
                _multi_controlled_ry(qc, theta_sign * th_int[("concrete", 1)], [w_in_slab, qg[0]] + ctrl_common, qcI[t])
                qc.x(w_in_slab)

                _multi_controlled_ry(qc, theta_sign * th_int[("tungsten", 0)], [w_is_tung, w_g0] + ctrl_common, qcI[t])
                _multi_controlled_ry(qc, theta_sign * th_int[("tungsten", 1)], [w_is_tung, qg[0]] + ctrl_common, qcI[t])

                _multi_controlled_ry(qc, theta_sign * th_int[("air", 0)], [w_is_air, w_g0] + ctrl_common, qcI[t])
                _multi_controlled_ry(qc, theta_sign * th_int[("air", 1)], [w_is_air, qg[0]] + ctrl_common, qcI[t])
            else:

                qc.x(qstatus[0]); qc.x(qstatus[1])
                alive_ctrls = [qstatus[0], qstatus[1]]

                qc.x(w_in_slab)
                _multi_controlled_ry(qc, theta_sign * th_int[("concrete", 0)], alive_ctrls + [w_in_slab, w_g0, nb_guard_ctrl], qcI[t])
                _multi_controlled_ry(qc, theta_sign * th_int[("concrete", 1)], alive_ctrls + [w_in_slab, qg[0], nb_guard_ctrl], qcI[t])
                qc.x(w_in_slab)

                _multi_controlled_ry(qc, theta_sign * th_int[("tungsten", 0)], alive_ctrls + [w_is_tung, w_g0, nb_guard_ctrl], qcI[t])
                _multi_controlled_ry(qc, theta_sign * th_int[("tungsten", 1)], alive_ctrls + [w_is_tung, qg[0], nb_guard_ctrl], qcI[t])

                _multi_controlled_ry(qc, theta_sign * th_int[("air", 0)], alive_ctrls + [w_is_air, w_g0, nb_guard_ctrl], qcI[t])
                _multi_controlled_ry(qc, theta_sign * th_int[("air", 1)], alive_ctrls + [w_is_air, qg[0], nb_guard_ctrl], qcI[t])

                qc.x(qstatus[0]); qc.x(qstatus[1])

        _apply_interaction_coin_rotations(+1.0)

        use_int_event_ledger = bool(use_term_tape and (qTermA is not None) and (step_alive_guard is not None))
        int_guard_ctrl = qcI[t]
        int_guard_is_ledger = False
        if use_int_event_ledger:
            int_event_ctrls = [qcI[t], nb_guard_ctrl, step_alive_guard]
            _mcx(qc, int_event_ctrls, qTermA[t])
            int_guard_ctrl = qTermA[t]
            int_guard_is_ledger = True

            qc.cx(qTermA[t], qcI[t])

        int_abs_ctrls = ([int_guard_ctrl] if int_guard_is_ledger else [qcI[t], nb_guard_ctrl] + ([step_alive_guard] if (step_alive_guard is not None) else []))

        abs_coin = (qAbs[t] if use_absorb_tape else qcoin[0])

        abs_coin_ledger_mode = bool(int_guard_is_ledger and (step_alive_guard is not None))

        def _apply_absorb_coin_rotations(theta_sign: float = +1.0):
            if abs_coin_ledger_mode:

                ctrl_common = [int_guard_ctrl]
                qc.x(w_in_slab)
                _multi_controlled_ry(qc, theta_sign * th_abs[("concrete", 0)], [w_in_slab, w_g0] + ctrl_common, abs_coin)
                _multi_controlled_ry(qc, theta_sign * th_abs[("concrete", 1)], [w_in_slab, qg[0]] + ctrl_common, abs_coin)
                qc.x(w_in_slab)

                _multi_controlled_ry(qc, theta_sign * th_abs[("tungsten", 0)], [w_is_tung, w_g0] + ctrl_common, abs_coin)
                _multi_controlled_ry(qc, theta_sign * th_abs[("tungsten", 1)], [w_is_tung, qg[0]] + ctrl_common, abs_coin)

                _multi_controlled_ry(qc, theta_sign * th_abs[("air", 0)], [w_is_air, w_g0] + ctrl_common, abs_coin)
                _multi_controlled_ry(qc, theta_sign * th_abs[("air", 1)], [w_is_air, qg[0]] + ctrl_common, abs_coin)
            else:

                qc.x(qstatus[0]); qc.x(qstatus[1])
                alive_ctrls = [qstatus[0], qstatus[1]]

                qc.x(w_in_slab)
                _multi_controlled_ry(qc, theta_sign * th_abs[("concrete", 0)], alive_ctrls + [w_in_slab, w_g0] + int_abs_ctrls, abs_coin)
                _multi_controlled_ry(qc, theta_sign * th_abs[("concrete", 1)], alive_ctrls + [w_in_slab, qg[0]] + int_abs_ctrls, abs_coin)
                qc.x(w_in_slab)

                _multi_controlled_ry(qc, theta_sign * th_abs[("tungsten", 0)], alive_ctrls + [w_is_tung, w_g0] + int_abs_ctrls, abs_coin)
                _multi_controlled_ry(qc, theta_sign * th_abs[("tungsten", 1)], alive_ctrls + [w_is_tung, qg[0]] + int_abs_ctrls, abs_coin)

                _multi_controlled_ry(qc, theta_sign * th_abs[("air", 0)], alive_ctrls + [w_is_air, w_g0] + int_abs_ctrls, abs_coin)
                _multi_controlled_ry(qc, theta_sign * th_abs[("air", 1)], alive_ctrls + [w_is_air, qg[0]] + int_abs_ctrls, abs_coin)

                qc.x(qstatus[0]); qc.x(qstatus[1])

        _apply_absorb_coin_rotations(+1.0)

        abs_event_ctrls = ([int_guard_ctrl, abs_coin] if int_guard_is_ledger else [qcI[t], nb_guard_ctrl, abs_coin] + ([step_alive_guard] if (step_alive_guard is not None) else []))
        _mcx(qc, abs_event_ctrls, w_abs_event)
        _set_status_via_term(
            qc, qstatus,
            (w_abs_term if (use_term_tape and (qTermA is not None)) else qterm[0]),
            w_abs_event,
            0b10,
        )

        if not int_guard_is_ledger:
            _mcx(qc, abs_event_ctrls, w_abs_event)

        if not use_absorb_tape:
            qc.x(qstatus[0])
            _mcx(qc, [qstatus[0], qstatus[1]], qcoin[0])
            qc.x(qstatus[0])

        did_early_abs_erase = False
        if profile.early_absorb_erase and int_guard_is_ledger and use_absorb_tape and (qAbs is not None):
            qc.cx(w_abs_event, qAbs[t])
            did_early_abs_erase = True

        if use_dir_tape:

            w_scatter = (w_dir_pred if int_guard_is_ledger else w_abs_event)

            if int_guard_is_ledger:
                qc.x(w_abs_event)
                _mcx(qc, [int_guard_ctrl, w_abs_event], w_scatter)
                qc.x(w_abs_event)
            else:

                qc.x(qstatus[0]); qc.x(qstatus[1])
                if use_absorb_tape and (qAbs is not None):
                    qc.x(qAbs[t])
                    _mcx(qc, ([qstatus[0], qstatus[1], qAbs[t], int_guard_ctrl] if int_guard_is_ledger else [qstatus[0], qstatus[1], qcI[t], qAbs[t], nb_guard_ctrl] + ([step_alive_guard] if (step_alive_guard is not None) else [])), w_scatter)
                    qc.x(qAbs[t])
                else:
                    _mcx(qc, ([qstatus[0], qstatus[1], int_guard_ctrl] if int_guard_is_ledger else [qstatus[0], qstatus[1], qcI[t], nb_guard_ctrl] + ([step_alive_guard] if (step_alive_guard is not None) else [])), w_scatter)
                qc.x(qstatus[0]); qc.x(qstatus[1])

            scatter_ctrl = w_scatter
            if use_scatter_tape and (qScat is not None):
                qc.cx(w_scatter, qScat[t])
                scatter_ctrl = qScat[t]

            dirT = [qdirT[3*t + i] for i in range(3)]

            qc.append(_U6.control(1), [scatter_ctrl] + dirT)

            for i in range(3):
                qc.ccx(scatter_ctrl, qdir[i], dirT[i])
            for i in range(3):
                qc.ccx(scatter_ctrl, dirT[i], qdir[i])

            if int_guard_is_ledger:
                qc.x(w_abs_event)
                _mcx(qc, [int_guard_ctrl, w_abs_event], w_scatter)
                qc.x(w_abs_event)
            else:
                qc.x(qstatus[0]); qc.x(qstatus[1])
                if use_absorb_tape and (qAbs is not None):
                    qc.x(qAbs[t])
                    _mcx(qc, ([qstatus[0], qstatus[1], qAbs[t], int_guard_ctrl] if int_guard_is_ledger else [qstatus[0], qstatus[1], qcI[t], qAbs[t], nb_guard_ctrl] + ([step_alive_guard] if (step_alive_guard is not None) else [])), w_scatter)
                    qc.x(qAbs[t])
                else:
                    _mcx(qc, ([qstatus[0], qstatus[1], int_guard_ctrl] if int_guard_is_ledger else [qstatus[0], qstatus[1], qcI[t], nb_guard_ctrl] + ([step_alive_guard] if (step_alive_guard is not None) else [])), w_scatter)
                qc.x(qstatus[0]); qc.x(qstatus[1])
        else:
            if int_guard_is_ledger:
                w_scatter = w_dir_pred
                qc.x(w_abs_event)
                _mcx(qc, [int_guard_ctrl, w_abs_event], w_scatter)
                qc.x(w_abs_event)
                qc.append(_U6.control(1), [w_scatter] + list(qdir))
                qc.x(w_abs_event)
                _mcx(qc, [int_guard_ctrl, w_abs_event], w_scatter)
                qc.x(w_abs_event)
            else:
                qc.x(qstatus[0]); qc.x(qstatus[1])
                if int_guard_is_ledger:
                    qc.append(_U6.control(3), [qstatus[0], qstatus[1], int_guard_ctrl] + list(qdir))
                elif step_alive_guard is not None:
                    qc.append(_U6.control(5), [qstatus[0], qstatus[1], qcI[t], nb_guard_ctrl, step_alive_guard] + list(qdir))
                else:
                    qc.append(_U6.control(4), [qstatus[0], qstatus[1], qcI[t], nb_guard_ctrl] + list(qdir))
                qc.x(qstatus[0]); qc.x(qstatus[1])

        if int_guard_is_ledger:
            qc.x(qstatus[0])
            _mcx(qc, [int_guard_ctrl, qstatus[0], qstatus[1]], w_abs_event)
            qc.x(qstatus[0])

        if profile.early_absorb_erase and (not did_early_abs_erase) and use_absorb_tape and (qAbs is not None):
            qc.x(qstatus[0])
            _mcx(qc, [qstatus[1], qstatus[0]], w_dir_pred)
            qc.cx(w_dir_pred, qAbs[t])
            _mcx(qc, [qstatus[1], qstatus[0]], w_dir_pred)
            qc.x(qstatus[0])

        if step_alive_guard is not None and (not int_guard_is_ledger):
            _apply_interaction_coin_rotations(-1.0)

        unbuild_g0()
        _and_not_into(qc, w_in_slab, w_in_duct, w_is_tung)
        _and_into(qc, w_in_slab, w_in_duct, w_is_air)
        _uncompute_in_slab_and_duct(qc, qx, qy, qz, slab_work, cfg)

        if use_term_tape and (qTermB is not None):
            if full_history_mode:

                nb_guard_for_move = qTermB[t]
            else:
                _compute_boundary_flag(
                    qc, qx, qy, qz,
                    w_boundary,
                    roi_work[:6],
                    [roi_work[6], roi_work[7], roi_work[8], qwork[11]],
                    cfg,
                )
                qc.x(w_boundary)
                qc.cx(w_boundary, qTermB[t])
                qc.x(w_boundary)
                _uncompute_boundary_flag(
                    qc, qx, qy, qz,
                    w_boundary,
                    roi_work[:6],
                    [roi_work[6], roi_work[7], roi_work[8], qwork[11]],
                    cfg,
                )

                nb_guard_for_move = None
        else:
            qc.x(w_boundary)
            _uncompute_boundary_flag(
                qc, qx, qy, qz,
                w_boundary,
                roi_work[:6],
                [roi_work[6], roi_work[7], roi_work[8], qwork[11]],
                cfg,
            )

        def move_if_dir(k: int, axis: str, sign: int):

            qc.x(qstatus[0]); qc.x(qstatus[1])
            kb = [(k >> i) & 1 for i in range(3)]
            for i, b in enumerate(kb):
                if b == 0:
                    qc.x(qdir[i])
            move_ctrls = [qstatus[0], qstatus[1]] + list(qdir)
            if nb_guard_for_move is not None:
                move_ctrls = move_ctrls + [nb_guard_for_move]
            if step_alive_guard is not None:
                move_ctrls = move_ctrls + [step_alive_guard]
            _mcx(qc, move_ctrls, w_dir_pred)
            for i, b in enumerate(kb):
                if b == 0:
                    qc.x(qdir[i])
            qc.x(qstatus[0]); qc.x(qstatus[1])

            if axis == "x":
                (_ctrl_inc_mod if sign > 0 else _ctrl_dec_mod)(qc, qx, w_dir_pred)
            elif axis == "y":
                (_ctrl_inc_mod if sign > 0 else _ctrl_dec_mod)(qc, qy, w_dir_pred)
            elif axis == "z":
                (_ctrl_inc_mod if sign > 0 else _ctrl_dec_mod)(qc, qz, w_dir_pred)
            else:
                raise ValueError("axis")

            qc.x(qstatus[0]); qc.x(qstatus[1])
            for i, b in enumerate(kb):
                if b == 0:
                    qc.x(qdir[i])
            move_ctrls = [qstatus[0], qstatus[1]] + list(qdir)
            if nb_guard_for_move is not None:
                move_ctrls = move_ctrls + [nb_guard_for_move]
            if step_alive_guard is not None:
                move_ctrls = move_ctrls + [step_alive_guard]
            _mcx(qc, move_ctrls, w_dir_pred)
            for i, b in enumerate(kb):
                if b == 0:
                    qc.x(qdir[i])
            qc.x(qstatus[0]); qc.x(qstatus[1])

        move_if_dir(0, "x", +1)
        move_if_dir(1, "x", -1)
        move_if_dir(2, "y", +1)
        move_if_dir(3, "y", -1)
        move_if_dir(4, "z", +1)
        move_if_dir(5, "z", -1)

        if profile.compact_interaction_ledger and int_guard_is_ledger and use_dir_tape and (qTermA is not None):
            dirT = [qdirT[3*t + i] for i in range(3)]
            w_dir_nonzero = w_dir_pred
            w_dir_or12 = qwork[12]

            qc.x(qstatus[0])
            _mcx(qc, [qstatus[0], qstatus[1]], qTermA[t])
            qc.x(qstatus[0])

            _or_into(qc, dirT[0], dirT[1], w_dir_or12)
            _or_into(qc, w_dir_or12, dirT[2], w_dir_nonzero)
            qc.x(qstatus[0]); qc.x(qstatus[1])
            _mcx(qc, [qstatus[0], qstatus[1], w_dir_nonzero], qTermA[t])
            qc.x(qstatus[0]); qc.x(qstatus[1])
            _unor_into(qc, w_dir_or12, dirT[2], w_dir_nonzero)
            _unor_into(qc, dirT[0], dirT[1], w_dir_or12)

        if step_alive_guard is not None:
            if profile.cleanup_alive_ledger:
                qc.x(qstatus[0])
                qc.cx(qstatus[0], step_alive_guard)
                qc.x(qstatus[0])

    for t in range(n_steps):
        compile_step_kernel(t)

    if int(n_steps) < int(detector_min_step):
        w_sp = w_abs_event

        qc.x(qstatus[1])
        _mcx(qc, [qstatus[0], qstatus[1]], w_sp)
        qc.x(qstatus[1])

        qc.cx(w_sp, qstatus[1])

    info = estimate_resources(cfg, n_steps, use_dir_tape=use_dir_tape, use_absorb_tape=use_absorb_tape, use_term_tape=use_term_tape, use_scatter_tape=use_scatter_tape)
    return qc, info

def estimate_resources(
    cfg: TransportConfig,
    n_steps: int,
    use_dir_tape: bool = False,
    use_absorb_tape: bool = False,
    use_term_tape: bool = False,
    use_scatter_tape: bool = False,
    full_history_mode: bool = False,
    kernel_profile: Optional[Union[str, KernelProfile]] = None,
) -> Dict[str, int]:
    profile = resolve_kernel_profile(
        kernel_profile=kernel_profile,
        use_dir_tape=use_dir_tape,
        use_absorb_tape=use_absorb_tape,
        use_term_tape=use_term_tape,
        use_scatter_tape=use_scatter_tape,
        full_history_mode=full_history_mode,
    )
    use_dir_tape = profile.use_dir_tape
    use_absorb_tape = profile.use_absorb_tape
    use_term_tape = profile.use_term_tape
    use_scatter_tape = profile.use_scatter_tape
    full_history_mode = profile.full_history_mode

    geom = cfg.geom
    xbits = _ceil_log2(geom.nx)
    ybits = _ceil_log2(geom.ny)
    zbits = _ceil_log2(geom.nz)

    term_scratch_qubits = 0 if use_term_tape else 1
    shared_abs_coin_qubits = 0 if use_absorb_tape else 1
    base = (xbits + ybits + zbits) + 1 + 3 + 2 + term_scratch_qubits + shared_abs_coin_qubits
    total = base + 17 + n_steps

    absorb_tape_qubits = 0
    if use_absorb_tape:
        absorb_tape_qubits = int(n_steps)
        total += absorb_tape_qubits

    dir_tape_qubits = 0
    if use_dir_tape:

        dir_tape_qubits = 3 * n_steps
        total += dir_tape_qubits

    term_tape_qubits = 0
    if use_term_tape:
        term_tape_qubits = 3 * n_steps
        total += term_tape_qubits

    scatter_tape_qubits = 0
    if use_scatter_tape:
        scatter_tape_qubits = int(n_steps)
        total += scatter_tape_qubits

    return {
        "qubits_total": int(total),
        "pos_qubits": int(xbits + ybits + zbits),
        "g_qubits": 1,
        "dir_qubits": 3,
        "status_qubits": 2,
        "term_qubits": int(0 if use_term_tape else 1),
        "coin_qubits": int(0 if use_absorb_tape else 1),
        "work_qubits": 17,
        "cI_tape_qubits": int(n_steps),
        "cA_tape_qubits": int(absorb_tape_qubits),
        "dir_tape_qubits": int(dir_tape_qubits),
        "term_tape_qubits": int(term_tape_qubits),
        "tS_tape_qubits": int(scatter_tape_qubits),
    }
