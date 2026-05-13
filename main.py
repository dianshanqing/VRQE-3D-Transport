"""Command-line entry point for MC, circuit, TEST, IQAE, and diagnostics."""


from __future__ import annotations

import argparse
from dataclasses import replace
import os

os.environ.setdefault("RAYON_NUM_THREADS", os.environ.get("RAYON_NUM_THREADS", "1"))
os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "1"))
os.environ.setdefault("MKL_NUM_THREADS", os.environ.get("MKL_NUM_THREADS", "1"))
os.environ.setdefault("OPENBLAS_NUM_THREADS", os.environ.get("OPENBLAS_NUM_THREADS", "1"))
os.environ.setdefault("NUMEXPR_NUM_THREADS", os.environ.get("NUMEXPR_NUM_THREADS", "1"))
import sys
import json
import hashlib
import time
from datetime import datetime
from typing import Any, Dict, Optional
from contextlib import contextmanager

import numpy as np

from config import default_config
from mc import run_mc
from plotting import save_mc_plots
from unitary_circuit import build_unitary_circuit_qiskit, estimate_resources, resolve_kernel_profile
from stats_utils import binom_se
class _TeeStdout:

    def __init__(self, fp, original):
        self.fp = fp
        self.original = original

    def write(self, s):
        self.fp.write(s); self.fp.flush()
        self.original.write(s); self.original.flush()

    def flush(self):
        self.fp.flush()
        self.original.flush()

@contextmanager
def _tee_to_file(log_path: str):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    orig = sys.stdout
    try:
        fp = open(log_path, 'w', encoding='utf-8')
    except PermissionError:

        base, ext = os.path.splitext(log_path)
        alt_path = f"{base}_{os.getpid()}{ext or '.log'}"
        fp = open(alt_path, 'w', encoding='utf-8')
        print(f"[warn] Cannot open log_path={log_path} (PermissionError). Using {alt_path} instead.")
    with fp:
        sys.stdout = _TeeStdout(fp, orig)
        try:
            yield
        finally:
            sys.stdout = orig

def _default_run_outdir(phase: str) -> str:
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    return os.path.join('outputs', f'run_{phase}_{ts}')

def _log_ts(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

@contextmanager
def _timed_block(title: str):
    t0 = time.time()
    _log_ts(f"START: {title}")
    try:
        yield
    finally:
        dt = time.time() - t0
        _log_ts(f"END  : {title} (elapsed {dt:.2f}s)")

def _effective_kernel_profile(args) -> str:

    name = str(getattr(args, "kernel_profile", "") or "").strip()
    return name if name else "manual"

def _resolved_kernel_profile(args):
    return resolve_kernel_profile(
        kernel_profile=_effective_kernel_profile(args),
        use_dir_tape=bool(getattr(args, "dir_tape", False)),
        use_absorb_tape=bool(getattr(args, "absorb_tape", False)),
        use_term_tape=bool(getattr(args, "term_tape", False)),
        use_scatter_tape=bool(getattr(args, "scatter_tape", False)),
        full_history_mode=bool(getattr(args, "full_history_mode", False)),
    )

def _kernel_profile_flags(profile):
    return {
        "kernel_profile": str(profile.name),
        "dir_tape": bool(profile.use_dir_tape),
        "absorb_tape": bool(profile.use_absorb_tape),
        "term_tape": bool(profile.use_term_tape),
        "scatter_tape": bool(profile.use_scatter_tape),
        "full_history_mode": bool(profile.full_history_mode),
    }

class _StageHeartbeat:

    def __init__(self, interval_sec: float, tag: str, state: dict):
        import threading as _threading
        self.interval = float(interval_sec) if interval_sec is not None else 0.0
        self.tag = str(tag)
        self.state = state
        self._stop = _threading.Event()
        self._th = _threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        if self.interval and self.interval > 0:
            self._th.start()

    def stop(self) -> None:
        try:
            self._stop.set()
            self._th.join(timeout=1.0)
        except Exception:
            pass

    def _run(self) -> None:
        next_t = time.time() + float(self.interval)
        while not self._stop.is_set():
            rem = max(0.0, next_t - time.time())
            self._stop.wait(timeout=rem)
            if self._stop.is_set():
                break
            st = self.state or {}
            stage = st.get("stage", ")")
            if "shots_total" in st:
                done = int(st.get("shots_done", 0) or 0)
                total = int(st.get("shots_total", 0) or 0)
                infl = int(st.get("shots_inflight", 0) or 0)
                if infl > 0:
                    msg = f"[HB] {self.tag} stage={stage} shots={done}+{infl}/{total}"
                else:
                    msg = f"[HB] {self.tag} stage={stage} shots={done}/{total}"
            else:
                msg = f"[HB] {self.tag} stage={stage}"
            _log_ts(msg)
            next_t = next_t + float(self.interval)

def _derive_chunk_seed(base_seed: Optional[int], *, abs_shot_offset: int, chunk_shots: int) -> Optional[int]:

    if base_seed is None:
        return None

    mask64 = ((1 << 64) - 1)
    x = int(base_seed) & mask64
    x ^= ((int(abs_shot_offset) + 0x9E3779B97F4A7C15) & mask64)
    x ^= (((int(chunk_shots) + 1) * 0xBF58476D1CE4E5B9) & mask64)
    x &= mask64

    x ^= (x >> 30)
    x = (x * 0xBF58476D1CE4E5B9) & mask64
    x ^= (x >> 27)
    x = (x * 0x94D049BB133111EB) & mask64
    x ^= (x >> 31)

    return int(x & 0x7FFFFFFF)

def _normalize_meas_order(meas_order_obj):

    out = []
    if meas_order_obj is None:
        return out
    for item in meas_order_obj:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            out.append((str(item[0]), int(item[1])))
        else:
            out.append(tuple(item))
    return out

def _aer_run_counts_chunked(sim, circuit, *, total_shots: int, shot_chunk: int, heartbeat_sec: float, hb_tag: str, seed: Optional[int] = None, shard_id: int = -1, global_shot_offset: int = 0, progress_state: Optional[dict] = None):

    counts_total: Dict[str, int] = {}
    done = 0

    if progress_state is not None:
        progress_state.update({'stage': 'aer_run', 'shots_done': 0, 'shots_total': int(total_shots), 'shots_inflight': 0})

    chunk = int(shot_chunk) if (shot_chunk is not None and int(shot_chunk) > 0) else 0

    while done < total_shots:
        remain = int(total_shots - done)
        abs_shot_offset = int(global_shot_offset) + int(done)
        if chunk <= 0:
            cur = remain
        else:

            mod = int(abs_shot_offset) % int(chunk)
            to_boundary = (int(chunk) - mod) if mod != 0 else int(chunk)
            cur = min(remain, to_boundary)

        run_kwargs = {}
        chunk_seed = _derive_chunk_seed(seed, abs_shot_offset=int(abs_shot_offset), chunk_shots=int(cur))
        if chunk_seed is not None:
            run_kwargs["seed_simulator"] = int(chunk_seed)

        if progress_state is not None:
            progress_state.update({'stage': 'aer_run', 'shots_done': int(done), 'shots_total': int(total_shots), 'shots_inflight': int(cur)})

        job = sim.run(circuit, shots=int(cur), **run_kwargs)

        while True:
            try:
                if job.done():
                    break
            except Exception:

                break
            time.sleep(0.25)

        res = job.result()
        counts = res.get_counts()
        for k, v in counts.items():
            kk = str(k)
            counts_total[kk] = int(counts_total.get(kk, 0)) + int(v)

        done += int(cur)

        if progress_state is not None:
            progress_state.update({'stage': 'aer_run', 'shots_done': int(done), 'shots_total': int(total_shots), 'shots_inflight': 0})

        _log_ts(f"[PROG] {hb_tag} shots={done}/{total_shots}")

    return counts_total

def _write_results(outdir: str, payload: dict):
    os.makedirs(outdir, exist_ok=True)
    p_json = os.path.join(outdir, 'results.json')
    with open(p_json, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    p_txt = os.path.join(outdir, 'results.txt')
    with open(p_txt, 'w', encoding='utf-8') as f:
        for k, v in payload.items():
            f.write(f"{k}: {v}\n")
    return p_json, p_txt

def _format_energy_hist(cfg, hist: np.ndarray) -> str:
    b = cfg.groups.boundaries
    lines = []
    for g in range(cfg.groups.n_groups):
        lo = b[g + 1]
        hi = b[g]
        lines.append(f"  g={g}  [{lo:.2f},{hi:.2f}] : {hist[g]:.4f}")
    return "\n".join(lines)

def _apply_xt_overrides(cfg, args):

    if getattr(args, "xt_no_interaction", False):

        mats = cfg.mats

        def _zero_p_int(mat):
            return replace(mat, p_int=[0.0 for _ in mat.p_int])

        mats2 = replace(
            mats,
            concrete=_zero_p_int(mats.concrete),
            tungsten=_zero_p_int(mats.tungsten),
            air=_zero_p_int(mats.air),
        )
        cfg = replace(cfg, mats=mats2)

    return cfg

def phase_mc(args) -> int:
    cfg = default_config()
    cfg = _apply_xt_overrides(cfg, args)
    res = run_mc(
        cfg,
        n_steps=args.steps,
        histories=args.histories,
        seed=args.seed,
        unitary_matched_energy=bool(args.unitary_matched_energy),
    )

    print("=" * 72)
    print("Phase 1: Classical Monte Carlo baseline (Paper-1)")
    print("Model: 16x8x8, tungsten slab(x=8..9) + air duct(y,z=3..4), detector x=12 ROI 2x2")
    print("=" * 72)
    print(f"steps      : {args.steps}")
    print(f"histories  : {args.histories:,}")
    print(f"seed       : {args.seed}")
    print(f"unitary_matched_energy : {bool(args.unitary_matched_energy)}")
    print("-")
    print(f"P(detect)  : {res.p_detect:.6f}  (SE={res.se_detect:.3e})")
    print(f"P(absorb)  : {res.p_absorb:.6f}  (SE={res.se_absorb:.3e})")
    print(f"P(boundary): {res.p_boundary:.6f}  (SE={res.se_boundary:.3e})")
    print(f"P(survive) : {res.p_survive:.6f}  (SE={res.se_survive:.3e})")
    print("-")
    if res.counts["detect"] > 0:
        print("Energy composition at detector (normalized):")
        print(_format_energy_hist(cfg, res.energy_hist_detect))
    else:
        print("No detector hits; energy histogram is empty.")

    if args.plot:
        outdir = args.outdir
        os.makedirs(outdir, exist_ok=True)
        paths = save_mc_plots(cfg, res, outdir=outdir)
        print("-")
        print("Saved figures:")
        for p in paths:
            print(" ", p)

    payload = {
        'phase': 'mc',
        'steps': int(args.steps),
        'histories': int(args.histories),
        'seed': int(args.seed),
        'unitary_matched_energy': bool(args.unitary_matched_energy),
        'p_detect': float(res.p_detect),
        'p_absorb': float(res.p_absorb),
        'p_boundary': float(res.p_boundary),
        'p_survive': float(res.p_survive),
        'se_detect': float(res.se_detect),
        'se_absorb': float(res.se_absorb),
        'se_boundary': float(res.se_boundary),
        'se_survive': float(res.se_survive),
        'energy_hist_detect': res.energy_hist_detect.tolist(),
        'counts': res.counts,
    }
    _write_results(args.outdir, payload)
    return 0

def phase_circuit(args) -> int:
    cfg = default_config()
    cfg = _apply_xt_overrides(cfg, args)
    prof = _resolved_kernel_profile(args)
    est = estimate_resources(
        cfg,
        n_steps=args.steps,
        kernel_profile=prof,
    )

    print("=" * 72)
    print("Phase 2: Build unitary A circuit (Paper-1)")
    print("Key: direction uses 6D coin unitary embedding; per-step interaction uses tape cI[t].")
    print("=" * 72)
    for k in ["xbits", "ybits", "zbits", "gbits", "dir", "pos_qubits", "status", "term", "coin", "cI_tape", "cA_tape_qubits", "dir_tape_qubits", "term_tape_qubits", "tS_tape_qubits", "work", "total_qubits"]:
        if k in est:
            print(f"{k:16s}: {est[k]}")
    if "note" in est:
        print("-")
        print(est["note"])

    try:
        qc, _info = build_unitary_circuit_qiskit(
            cfg,
            n_steps=args.steps,
            kernel_profile=prof,
            disable_detector=args.xt_disable_detector,
            disable_boundary=args.xt_disable_boundary,
        )
    except ImportError as e:
        print("\n[ERROR] Failed to build circuit: Qiskit is required for phase=circuit.")
        print("Install: pip install qiskit qiskit-aer qiskit-algorithms")
        print("Exception:", repr(e))
        return 2
    except Exception as e:
        print("\n[ERROR] Failed to build circuit.")
        print("Exception:", repr(e))
        return 2

    print("\nCircuit built.")
    print(f"num_qubits : {qc.num_qubits}")
    try:
        print(f"depth      : {qc.depth()}")
        ops = qc.count_ops()
        top = sorted(ops.items(), key=lambda x: (-x[1], x[0]))[:12]
        print("top ops    :", ", ".join([f"{k}={v}" for k, v in top]))
    except Exception:
        pass

    payload = {"phase": "circuit", "steps": int(args.steps), "est": est, "num_qubits": int(qc.num_qubits)}
    _write_results(args.outdir, payload)
    return 0

def phase_test(args) -> int:

    cfg = default_config()
    cfg = _apply_xt_overrides(cfg, args)
    prof = _resolved_kernel_profile(args)
    prof_flags = _kernel_profile_flags(prof)

    try:
        sid = int(getattr(args, 'shard_id', -1) or -1)
        ns = int(getattr(args, 'num_shards', 1) or 1)
        g = cfg.geom
        s = cfg.source
        print(f"[CFG] test steps={int(args.steps)} shard={sid}/{ns} geom={g.nx}x{g.ny}x{g.nz} src=({s.x0},{s.y0},{s.z0}) det_x={g.detector_x} roi_y=[{g.roi_y0}..{g.roi_y1}] roi_z=[{g.roi_z0}..{g.roi_z1}] kernel_profile={prof.name} dir_tape={prof.use_dir_tape} absorb_tape={prof.use_absorb_tape} term_tape={prof.use_term_tape} scatter_tape={prof.use_scatter_tape} full_history={prof.full_history_mode}")
    except Exception:
        pass

    hb_state = {'stage': 'init'}
    hb = _StageHeartbeat(float(getattr(args, 'aer_heartbeat_sec', 0.0) or 0.0), f"test shard={int(getattr(args,'shard_id',-1) or -1):03d}", hb_state)
    hb.start()
    sid0 = int(getattr(args,'shard_id',-1) or -1)
    hb_tag = f"test shard={sid0:03d}"

    try:
        from qiskit_aer import AerSimulator
        from qiskit import transpile, QuantumCircuit, ClassicalRegister
    except Exception as e:
        print("[ERROR] Aer not available:", repr(e))
        return 2

    try:
        hb_state['stage'] = 'build'
        with _timed_block(f"build unitary circuit A (steps={args.steps})"):
            qc, _info = build_unitary_circuit_qiskit(
                cfg,
                n_steps=args.steps,
                kernel_profile=prof,
                disable_detector=args.xt_disable_detector,
                disable_boundary=args.xt_disable_boundary,
            )
    except Exception as e:
        print("[ERROR] Failed to build A circuit.")
        print("Exception:", repr(e))
        return 2

    qcm = QuantumCircuit(*qc.qregs, name=qc.name + "_meas")
    qcm.compose(qc, inplace=True)

    def _find_qreg(name: str):
        for qr in qcm.qregs:
            if qr.name == name:
                return qr
        raise RuntimeError(f"qreg '{name}' not found")
    rg = _find_qreg("g")
    rstatus = _find_qreg("status")

    rx = _find_qreg("x") if getattr(args, 'test_measure_x', False) else None
    xbits = len(rx) if rx is not None else 0

    rwork = _find_qreg("work") if getattr(args, 'debug_measure_work', False) else None
    wbits = len(rwork) if rwork is not None else 0

    rdir = _find_qreg("dir") if getattr(args, 'test_measure_dir', False) else None
    dirbits = len(rdir) if rdir is not None else 0

    rci = _find_qreg("cI") if getattr(args, 'test_measure_coins', False) else None
    cibits = len(rci) if rci is not None else 0

    rca = None
    cabits = 0
    if getattr(args, 'test_measure_coins', False):
        try:
            rca = _find_qreg("cA")
            cabits = len(rca)
        except Exception:
            rca = None
            cabits = 0

    gbits = len(rg)
    sbits = len(rstatus)

    meas_order = [
        ("g", gbits),
        ("status", sbits),
        ("x", xbits),
        ("work", wbits),
        ("dir", dirbits),
        ("cI", cibits),
        ("cA", cabits),
    ]
    total_cbits = sum(n for _, n in meas_order)

    off_g = 0
    off_status = off_g + gbits
    off_x = off_status + sbits
    off_work = off_x + xbits
    off_dir = off_work + wbits
    off_ci = off_dir + dirbits
    off_ca = off_ci + cibits

    qcm.add_register(ClassicalRegister(total_cbits, "c"))
    c = qcm.cregs[0]
    idx = 0
    for q in rg:
        qcm.measure(q, c[idx]); idx += 1
    for q in rstatus:
        qcm.measure(q, c[idx]); idx += 1
    if rx is not None:
        for q in rx:
            qcm.measure(q, c[idx]); idx += 1
    if rwork is not None:
        for q in rwork:
            qcm.measure(q, c[idx]); idx += 1
    if rdir is not None:
        for q in rdir:
            qcm.measure(q, c[idx]); idx += 1
    if rci is not None:
        for q in rci:
            qcm.measure(q, c[idx]); idx += 1
    if rca is not None:
        for q in rca:
            qcm.measure(q, c[idx]); idx += 1

    loaded_compiled = False
    tqc = None
    if getattr(args, "compiled_qpy", ""):
        try:
            import qiskit.qpy as qpy
            with open(args.compiled_qpy, "rb") as f:
                _circs = qpy.load(f)
            if not _circs:
                raise RuntimeError("empty QPY")
            tqc = _circs[0]
            loaded_compiled = True
            print(f"[INFO] Loaded compiled circuit from QPY: {args.compiled_qpy}")

            try:
                meta_path = str(args.compiled_qpy) + ".meta.json"
                if os.path.exists(meta_path):
                    with open(meta_path, "r", encoding="utf-8") as mf:
                        meta = json.load(mf)
                    cur_flags = {
                        "steps": int(args.steps),
                        "kernel_profile": str(prof.name),
                        "dir_tape": bool(prof.use_dir_tape),
                        "absorb_tape": bool(prof.use_absorb_tape),
                        "term_tape": bool(prof.use_term_tape),
                        "scatter_tape": bool(prof.use_scatter_tape),
                        "full_history_mode": bool(prof.full_history_mode),
                        "test_measure_x": bool(getattr(args, "test_measure_x", False)),
                        "test_measure_coins": bool(getattr(args, "test_measure_coins", False)),
                        "test_measure_dir": bool(getattr(args, "test_measure_dir", False)),
                        "aer_safe_basis": bool(getattr(args, "aer_safe_basis", False)),
                    }
                    cur_reg_layout = {qr.name: len(qr) for qr in qcm.qregs}
                    cur_num_qubits = int(qcm.num_qubits)
                    cur_meas_order = _normalize_meas_order(meas_order)
                    meta_meas_order = _normalize_meas_order(meta.get("meas_order"))
                    _meta_mismatch = (
                        meta.get("flags") != cur_flags
                        or meta.get("reg_layout") != cur_reg_layout
                        or int(meta.get("num_qubits")) != cur_num_qubits
                        or meta_meas_order != cur_meas_order
                    )
                    if _meta_mismatch:
                        print(f"[WARN] compiled-qpy meta mismatch; ignoring cached circuit. meta_flags={meta.get('flags')} cur_flags={cur_flags}")
                        try:
                            print(f"[WARN] meta_reg_layout={meta.get('reg_layout')} cur_reg_layout={cur_reg_layout}")
                            print(f"[WARN] meta_meas_order={meta_meas_order} cur_meas_order={cur_meas_order}")
                            print(f"[WARN] meta_num_qubits={meta.get('num_qubits')} cur_num_qubits={cur_num_qubits}")
                        except Exception:
                            pass
                        loaded_compiled = False
                        tqc = None
            except Exception as _e:
                print("[WARN] compiled-qpy meta check failed (ignored):", repr(_e))
        except Exception as e:
            print("[WARN] Failed to load --compiled-qpy; will transpile. Exception:", repr(e))
            loaded_compiled = False
            tqc = None

    n_qubits = qcm.num_qubits

    method = 'automatic'
    _mps_max_bond = int(getattr(args, 'mps_max_bond', 0) or 0)
    _mps_trunc = float(getattr(args, 'mps_trunc', 0.0) or 0.0)
    if n_qubits > 32 or _mps_max_bond > 0 or _mps_trunc > 0.0:
        method = 'matrix_product_state'
    sim = AerSimulator(method=method)

    try:
        aer_threads = int(getattr(args, 'aer_threads', 1))
        sim.set_options(max_parallel_threads=aer_threads)
        aer_max_mem = int(getattr(args, 'aer_max_mem_mb', 0))
        if aer_max_mem > 0:
            sim.set_options(max_memory_mb=aer_max_mem)
        mps_max_bond = int(getattr(args, 'mps_max_bond', 0))
        if mps_max_bond > 0:
            sim.set_options(matrix_product_state_max_bond_dimension=mps_max_bond)
        mps_trunc = float(getattr(args, 'mps_trunc', 0.0))
        if mps_trunc > 0.0:
            sim.set_options(matrix_product_state_truncation_threshold=mps_trunc)
        print(f"Aer options: threads={aer_threads}, max_memory_mb={aer_max_mem if aer_max_mem>0 else 'default'}, mps_max_bond={mps_max_bond if mps_max_bond>0 else 'default'}, mps_trunc={mps_trunc if mps_trunc>0 else 'default'}")
    except Exception as _e:
        print('[WARN] Failed to set Aer options:', repr(_e))
    print(f"Simulator method: {method} (num_qubits={n_qubits})")

    if (not loaded_compiled) and (not getattr(args, "skip_transpile", False)):
        last_err = None
        hb_state['stage'] = 'transpile'
        for attempt in range(3):
            try:
                if getattr(args, 'aer_safe_basis', False):

                    try:
                        sim.set_options(fusion_enable=False)
                    except Exception:
                        pass

                    label = "transpile (safe basis, no-backend)" if attempt == 0 else f"transpile retry#{attempt} (safe basis)"
                    with _timed_block(label):
                        tqc = transpile(qcm, optimization_level=0, basis_gates=["rz", "sx", "x", "cx"], num_processes=1)
                else:
                    label = "transpile (default, no-backend)" if attempt == 0 else f"transpile retry#{attempt} (default)"
                    with _timed_block(label):
                        tqc = transpile(qcm, optimization_level=0, num_processes=1)

                if tqc is not None and 'mcx_vchain' in tqc.count_ops():
                    tqc = tqc.decompose(reps=8)

                print('Compiled top ops:', dict(list(tqc.count_ops().items())[:10]))
                last_err = None
                break
            except BaseException as e:
                last_err = e

                print(f"[WARN] Transpile attempt {attempt+1}/3 failed:", repr(e))

                time.sleep(1.5)

        if last_err is not None:
            print("[ERROR] Transpile failed after retries.")
            print("This is usually caused by running too many shards compiling/transpiling in parallel on Windows.")
            print("Fix: (1) use compile-cache: run once with --compile-only --save-compiled-qpy and then run shards with --compiled-qpy;")
            print("     (2) limit max parallel shards (e.g. 4~8); (3) set RAYON_NUM_THREADS=1.")
            return 2
    else:
        if tqc is None:
            tqc = qcm
        if loaded_compiled:
            pass
        else:
            print("[WARN] Running without transpile (--skip-transpile). Use only if the circuit is already in safe basis.")

    if getattr(args, "save_compiled_qpy", "") and (tqc is not None):
        try:
            import qiskit.qpy as qpy
            os.makedirs(os.path.dirname(args.save_compiled_qpy), exist_ok=True)
            with open(args.save_compiled_qpy, "wb") as f:
                qpy.dump([tqc], f)
            print(f"[INFO] Saved compiled circuit to QPY: {args.save_compiled_qpy}")

            try:
                meta_path = str(args.save_compiled_qpy) + ".meta.json"
                meta = {
                    "flags": {
                        "steps": int(args.steps),
                        "kernel_profile": str(prof.name),
                        "dir_tape": bool(prof.use_dir_tape),
                        "absorb_tape": bool(prof.use_absorb_tape),
                        "term_tape": bool(prof.use_term_tape),
                        "scatter_tape": bool(prof.use_scatter_tape),
                        "full_history_mode": bool(prof.full_history_mode),
                        "test_measure_x": bool(getattr(args, "test_measure_x", False)),
                        "test_measure_coins": bool(getattr(args, "test_measure_coins", False)),
                        "test_measure_dir": bool(getattr(args, "test_measure_dir", False)),
                        "aer_safe_basis": bool(getattr(args, "aer_safe_basis", False)),
                    },
                    "reg_layout": {qr.name: len(qr) for qr in qcm.qregs},
                    "meas_order": meas_order,
                    "num_qubits": int(qcm.num_qubits),
                }
                with open(meta_path, "w", encoding="utf-8") as mf:
                    json.dump(meta, mf, ensure_ascii=False, indent=2)
                print(f"[INFO] Saved compiled meta: {meta_path}")
            except Exception as _e:
                print("[WARN] Failed to save compiled-qpy meta (ignored):", repr(_e))
        except Exception as e:
            print("[WARN] Failed to save --save-compiled-qpy:", repr(e))

    if getattr(args, "compile_only", False):
        print("[INFO] compile-only requested; exiting before Aer run.")
        return 0

    try:
        hb_state['stage'] = 'aer_run'
        with _timed_block(f"Aer run (shots={args.shots})"):
            counts = _aer_run_counts_chunked(
                sim,
                tqc,
                total_shots=int(args.shots),
                shot_chunk=int(getattr(args, "aer_shot_chunk", 100) or 0),
                heartbeat_sec=float(getattr(args, "aer_heartbeat_sec", 300.0) or 0.0),
                hb_tag=hb_tag,
                seed=int(getattr(args, "seed", 0) or 0),
                shard_id=int(getattr(args, "shard_id", -1) or -1),
                global_shot_offset=int(getattr(args, "global_shot_offset", 0) or 0),
                progress_state=hb_state,
            )
    except Exception as e:
        print('[ERROR] Simulator run failed.')
        print('Exception:', repr(e))
        if isinstance(e, MemoryError) or ('bad allocation' in repr(e).lower()) or ('memoryerror' in repr(e).lower()):
            print('[HINT] This often happens when too many shards run in parallel (each Aer MPS instance needs a lot of RAM).')
            print('       Reduce MAX_PARALLEL (e.g. 4~6), or reduce mps_max_bond, or increase system RAM/pagefile.')
        return 2

    dir_count_total: Dict[int, int] = {}
    invalid_dir_total = 0

    ci1 = [0] * int(args.steps) if cibits else []
    ca1 = [0] * int(args.steps) if cabits else []
    ca1_ci0_total = 0
    nonabsorb_with_ca_total = 0
    absorb_without_ca_total = 0

    x_count_total: Dict[int, int] = {}
    x_count_alive: Dict[int, int] = {}

    x_count_detect: Dict[int, int] = {}
    x_count_absorb: Dict[int, int] = {}
    x_count_boundary: Dict[int, int] = {}

    work_nonzero_total = 0
    work_nonzero_alive = 0

    n_alive = 0
    n_detect = 0
    n_absorb = 0
    n_boundary = 0
    g_hist = np.zeros(cfg.groups.n_groups, dtype=int)

    for bitstr, ct in counts.items():
        bits = str(bitstr)[::-1]

        g_bits = bits[off_g:off_g + gbits]
        g_val = 0
        for i, b in enumerate(g_bits):
            if b == '1':
                g_val |= (1 << i)

        s_bits = bits[off_status:off_status + sbits]
        s_val = 0
        for i, b in enumerate(s_bits):
            if b == '1':
                s_val |= (1 << i)

        if xbits:

            x_bits = bits[off_x:off_x + xbits]
            x_val = 0
            for j, b in enumerate(x_bits):
                if b == '1':
                    x_val |= (1 << j)
            x_count_total[x_val] = x_count_total.get(x_val, 0) + ct

            if s_val == 0:
                x_count_alive[x_val] = x_count_alive.get(x_val, 0) + ct
            elif s_val == 1:
                x_count_detect[x_val] = x_count_detect.get(x_val, 0) + ct
            elif s_val == 2:
                x_count_absorb[x_val] = x_count_absorb.get(x_val, 0) + ct
            else:
                x_count_boundary[x_val] = x_count_boundary.get(x_val, 0) + ct

        if wbits:
            off_work = off_x + xbits
            w_bits = bits[off_work:off_work + wbits]
            w_val = 0
            for j, b in enumerate(w_bits):
                if b == '1':
                    w_val |= (1 << j)
            if w_val != 0:
                work_nonzero_total += ct
                if s_val == 0:
                    work_nonzero_alive += ct

        if dirbits:
            dir_bits = bits[off_dir:off_dir + dirbits]
            dir_val = 0
            for j, b in enumerate(dir_bits):
                if b == '1':
                    dir_val |= (1 << j)
            dir_count_total[dir_val] = dir_count_total.get(dir_val, 0) + ct
            if dir_val >= 6:
                invalid_dir_total += ct

        ci_bits = None
        if cibits:
            ci_bits = bits[off_ci:off_ci + cibits]
            for t, b in enumerate(ci_bits[:int(args.steps)]):
                if b == '1':
                    ci1[t] += ct

        if cabits:
            ca_bits = bits[off_ca:off_ca + cabits]
            any_ca = False
            for t, b in enumerate(ca_bits[:int(args.steps)]):
                if b == '1':
                    any_ca = True
                    ca1[t] += ct
                    if ci_bits is not None and t < len(ci_bits) and ci_bits[t] == '0':
                        ca1_ci0_total += ct
            if any_ca and s_val != 2:
                nonabsorb_with_ca_total += ct
            if (not any_ca) and s_val == 2:
                absorb_without_ca_total += ct

        if s_val == 0:
            n_alive += ct
        elif s_val == 1:
            n_detect += ct
            if 0 <= g_val < len(g_hist):
                g_hist[g_val] += ct
        elif s_val == 2:
            n_absorb += ct
        else:
            n_boundary += ct

    shots = int(args.shots)
    p_detect = n_detect / shots
    p_absorb = n_absorb / shots
    p_boundary = n_boundary / shots
    p_survive = n_alive / shots

    se_detect = binom_se(p_detect, shots)
    se_absorb = binom_se(p_absorb, shots)
    se_boundary = binom_se(p_boundary, shots)
    se_survive = binom_se(p_survive, shots)

    print("\nTest (Aer sampling) summary:")
    print(f"steps : {args.steps}")
    print(f"shots : {shots}")
    print(f"P(detect)   ~ {p_detect:.6f}  (SE={se_detect:.3e})")
    print(f"P(absorb)   ~ {p_absorb:.6f}  (SE={se_absorb:.3e})")
    print(f"P(boundary) ~ {p_boundary:.6f}  (SE={se_boundary:.3e})")
    print(f"P(survive)  ~ {p_survive:.6f}  (SE={se_survive:.3e})")

    if wbits:
        p_work_nonzero = work_nonzero_total / shots
        p_work_nonzero_alive = work_nonzero_alive / max(1, n_alive)
        print(f"Work ancilla nonzero rate (all shots): {p_work_nonzero:.6f}")
        print(f"Work ancilla nonzero rate (alive only): {p_work_nonzero_alive:.6f}")

    diag = {}
    x_prob_total = None
    x_prob_alive = None
    x_prob_detect = None
    x_prob_absorb = None
    x_prob_boundary = None
    if xbits:
        x_prob_total = {int(k): float(v) / float(shots) for k, v in sorted(x_count_total.items())}
        x_prob_alive = {int(k): float(v) / float(max(1, n_alive)) for k, v in sorted(x_count_alive.items())}
        x_prob_detect = {int(k): float(v) / float(max(1, n_detect)) for k, v in sorted(x_count_detect.items())}
        x_prob_absorb = {int(k): float(v) / float(max(1, n_absorb)) for k, v in sorted(x_count_absorb.items())}
        x_prob_boundary = {int(k): float(v) / float(max(1, n_boundary)) for k, v in sorted(x_count_boundary.items())}

        print("x histogram (total):")
        for xk, pv in x_prob_total.items():
            if pv > 0:
                print(f"  x={xk:02d}: {pv:.6f}")

        print("x histogram (alive | normalized):")
        for xk, pv in x_prob_alive.items():
            if pv > 0:
                print(f"  x={xk:02d}: {pv:.6f}")

        try:
            reach_dist = abs(int(cfg.geom.detector_x) - int(cfg.source.x0))
            diag["reach_dist"] = int(reach_dist)
            diag["unreachable_detect"] = (int(args.steps) < reach_dist and int(n_detect) > 0)
            if diag["unreachable_detect"]:
                print(f"[WARN] unreachable-detect: steps={int(args.steps)} < |det_x-x0|={reach_dist} but n_detect={int(n_detect)}")
            alive_x0 = int(x_count_alive.get(0, 0))
            alive_xmax = int(x_count_alive.get(int(cfg.geom.nx) - 1, 0))
            diag["alive_at_x0"] = alive_x0
            diag["alive_at_xmax"] = alive_xmax
            diag["alive_on_x_boundary"] = ((alive_x0 + alive_xmax) > 0)
            if diag["alive_on_x_boundary"]:
                print(f"[WARN] alive-on-x-boundary: alive_at_x0={alive_x0} alive_at_xmax={alive_xmax}")
        except Exception:
            diag = {}
    work_nonzero_rate_total = None
    work_nonzero_rate_alive = None
    if wbits:
        work_nonzero_rate_total = work_nonzero_total / shots
        work_nonzero_rate_alive = work_nonzero_alive / max(1, n_alive)

    reg_layout = {qr.name: len(qr) for qr in qcm.qregs}
    meas_offsets = {
        "g": int(off_g),
        "status": int(off_status),
        "x": int(off_x),
        "work": int(off_work),
        "dir": int(off_dir),
        "cI": int(off_ci),
        "cA": int(off_ca),
    }

    coin_stats = None
    if cibits or cabits:
        coin_stats = {
            "cibits": int(cibits),
            "cabits": int(cabits),
            "ci1": [int(v) for v in ci1],
            "ca1": ([int(v) for v in ca1] if cabits else None),
            "p_ci1": [float(v) / float(shots) for v in ci1],
            "p_ca1": ([float(v) / float(shots) for v in ca1] if cabits else None),
            "ca1_ci0_total": int(ca1_ci0_total),
            "nonabsorb_with_ca_total": int(nonabsorb_with_ca_total),
            "absorb_without_ca_total": int(absorb_without_ca_total),
            "nonabsorb_with_ca_rate": float(nonabsorb_with_ca_total) / float(shots) if shots > 0 else 0.0,
            "absorb_without_ca_rate": float(absorb_without_ca_total) / float(shots) if shots > 0 else 0.0,
            "ca1_ci0_rate": float(ca1_ci0_total) / float(shots) if shots > 0 else 0.0,
        }

    dir_stats = None
    if dirbits:
        dir_stats = {
            "dirbits": int(dirbits),
            "dir_count_total": {str(k): int(v) for k, v in sorted(dir_count_total.items())},
            "invalid_dir_total": int(invalid_dir_total),
            "invalid_dir_rate": float(invalid_dir_total) / float(shots) if shots > 0 else 0.0,
        }

    circuit_signature = {
        "steps": int(args.steps),
        "flags": {
            "kernel_profile": str(prof.name),
            "dir_tape": bool(prof.use_dir_tape),
            "absorb_tape": bool(prof.use_absorb_tape),
            "term_tape": bool(prof.use_term_tape),
            "scatter_tape": bool(prof.use_scatter_tape),
            "full_history_mode": bool(prof.full_history_mode),
            "test_measure_x": bool(xbits),
            "test_measure_coins": bool(getattr(args, "test_measure_coins", False)),
            "test_measure_dir": bool(getattr(args, "test_measure_dir", False)),
            "aer_safe_basis": bool(getattr(args, "aer_safe_basis", False)),
        },
        "reg_layout": reg_layout,
        "meas_order": meas_order,
        "meas_offsets": meas_offsets,
        "sim_method": str(method),
        "num_qubits": int(n_qubits),
    }
    try:
        if tqc is not None:
            op_counts = dict(tqc.count_ops())
            circuit_signature.update({
                "compiled_depth": int(tqc.depth()),
                "compiled_size": int(tqc.size()),
                "compiled_op_counts_top": {str(k): int(v) for k, v in sorted(op_counts.items(), key=lambda kv: (-kv[1], str(kv[0])))[:20]},
            })
            hobj = {
                "num_qubits": int(tqc.num_qubits),
                "depth": int(tqc.depth()),
                "size": int(tqc.size()),
                "op_counts": {str(k): int(v) for k, v in sorted(op_counts.items(), key=lambda kv: str(kv[0]))},
            }
            circuit_signature["compiled_sig"] = hashlib.sha256(json.dumps(hobj, sort_keys=True).encode("utf-8")).hexdigest()
    except Exception:
        pass

    payload = {
        'phase': 'test',
        'steps': int(args.steps),
        'shots': int(shots),
        'seed': int(getattr(args,'seed',0) or 0),
        'shard_id': int(getattr(args,'shard_id',-1) or -1),
        'num_shards': int(getattr(args,'num_shards',1) or 1),
        'aer_shot_chunk': int(getattr(args, 'aer_shot_chunk', 0) or 0),
        'global_shot_offset': int(getattr(args, 'global_shot_offset', 0) or 0),
        'seed_strategy': 'splitmix64(root_seed, abs_shot_offset, chunk_shots)',
        'kernel_profile': str(prof.name),
        'dir_tape': bool(prof.use_dir_tape),
        'absorb_tape': bool(prof.use_absorb_tape),
        'term_tape': bool(prof.use_term_tape),
        'scatter_tape': bool(prof.use_scatter_tape),
        'full_history_mode': bool(prof.full_history_mode),
        'test_measure_coins': bool(getattr(args, 'test_measure_coins', False)),
        'test_measure_dir': bool(getattr(args, 'test_measure_dir', False)),
        'test_measure_x': bool(xbits),
        'cfg': {
            'nx': int(cfg.geom.nx), 'ny': int(cfg.geom.ny), 'nz': int(cfg.geom.nz),
            'source': {'x0': int(cfg.source.x0), 'y0': int(cfg.source.y0), 'z0': int(cfg.source.z0), 'g0': int(cfg.source.g0), 'dir0': int(getattr(cfg.source,'dir0',0))},
            'detector_x': int(cfg.geom.detector_x),
            'roi': {'y0': int(cfg.geom.roi_y0), 'y1': int(cfg.geom.roi_y1), 'z0': int(cfg.geom.roi_z0), 'z1': int(cfg.geom.roi_z1)},
        },
        'diag': diag,
        'n_detect': int(n_detect),
        'n_absorb': int(n_absorb),
        'n_boundary': int(n_boundary),
        'n_survive': int(shots - n_detect - n_absorb - n_boundary),
        'n_alive': int(n_alive),
        'counts': {'detect': int(n_detect), 'absorb': int(n_absorb), 'boundary': int(n_boundary), 'survive': int(shots - n_detect - n_absorb - n_boundary)},
        'p_detect': float(p_detect),
        'p_absorb': float(p_absorb),
        'p_boundary': float(p_boundary),
        'p_survive': float(p_survive),
        'se_detect': float(se_detect),
        'se_absorb': float(se_absorb),
        'se_boundary': float(se_boundary),
        'se_survive': float(se_survive),
        'x_count_total': {str(k): int(v) for k, v in x_count_total.items()},
        'x_count_alive': {str(k): int(v) for k, v in x_count_alive.items()},
        'x_count_detect': {str(k): int(v) for k, v in x_count_detect.items()},
        'x_count_absorb': {str(k): int(v) for k, v in x_count_absorb.items()},
        'x_count_boundary': {str(k): int(v) for k, v in x_count_boundary.items()},
        'x_prob_total': x_prob_total,
        'x_prob_alive': x_prob_alive,
        'x_prob_detect': x_prob_detect,
        'x_prob_absorb': x_prob_absorb,
        'x_prob_boundary': x_prob_boundary,
        'work_nonzero_rate_total': work_nonzero_rate_total,
        'work_nonzero_rate_alive': work_nonzero_rate_alive,
        'reg_layout': reg_layout,
        'meas_order': meas_order,
        'meas_offsets': meas_offsets,
        'coin_stats': coin_stats,
        'dir_stats': dir_stats,
        'circuit_signature': circuit_signature,
    }
    _write_results(args.outdir, payload)
    return 0

def _print_iqae_run_cfg(args, cfg, backend_options: dict, *, good: str, target_g: Optional[int], outdir: str):

    try:
        geom = getattr(cfg, 'geom', None)
        src = getattr(cfg, 'source', None)
        print('[IQAE_CFG] --------------------------------------------------')
        print(f"[IQAE_CFG] phase=iqae outdir={outdir}")
        if geom is not None and src is not None:
            print(f"[IQAE_CFG] geom={int(geom.nx)}x{int(geom.ny)}x{int(geom.nz)} src=({int(src.x0)},{int(src.y0)},{int(src.z0)}) det_x={int(geom.detector_x)}")
        print(
            '[IQAE_CFG] '
            f"steps={int(args.steps)} shots={int(args.shots)} seed={int(args.seed)} good={good} target_g={target_g} "
            f"eps={float(getattr(args, 'iqae_eps', 0.05)):.6g} alpha={float(getattr(args, 'iqae_alpha', 0.05)):.6g}"
        )
        print(
            '[IQAE_CFG] '
            f"dir_tape={bool(getattr(args, 'dir_tape', False))} absorb_tape={bool(getattr(args, 'absorb_tape', False))} "
            f"term_tape={bool(getattr(args, 'term_tape', False))} scatter_tape={bool(getattr(args, 'scatter_tape', False))} "
            f"full_history={bool(getattr(args, 'full_history_mode', False))} safe_basis={bool(getattr(args, 'aer_safe_basis', False))}"
        )
        print(
            '[IQAE_CFG] '
            f"aer_threads={int(getattr(args, 'aer_threads', 1))} shot_chunk={int(getattr(args, 'iqae_shot_chunk', 100) or 0)} "
            f"heartbeat_sec={float(getattr(args, 'iqae_heartbeat_sec', 300.0) or 0.0):.6g}"
        )
        if backend_options:
            print('[IQAE_CFG] backend_options=' + json.dumps(backend_options, ensure_ascii=False, sort_keys=True))
        else:
            print('[IQAE_CFG] backend_options={}')
        print('[IQAE_CFG] --------------------------------------------------')
    except Exception as e:
        print(f"[IQAE_CFG] WARN: failed to print config block: {repr(e)}")

def phase_iqae(args) -> int:

    try:
        from iqae import run_iqae_small, OracleSpec
    except Exception as e:
        print("\n[ERROR] Failed to import iqae module.")
        print("Exception:", repr(e))
        return 2
    cfg = default_config()
    cfg = _apply_xt_overrides(cfg, args)

    try:
        sid = int(getattr(args, 'shard_id', -1) or -1)
        ns = int(getattr(args, 'num_shards', 1) or 1)
        g = cfg.geom
        s = cfg.source
        print(f"[CFG] iqae steps={int(args.steps)} shard={sid}/{ns} geom={g.nx}x{g.ny}x{g.nz} src=({s.x0},{s.y0},{s.z0}) det_x={g.detector_x} dir_tape={bool(getattr(args,'dir_tape',False))} good={str(getattr(args,'iqae_good','detect'))}")
    except Exception:
        pass

    hb_state = {'stage': 'init'}
    sid0 = int(getattr(args,'shard_id',-1) or -1)
    hb_tag = f"iqae shard={sid0:03d}"
    hb = _StageHeartbeat(float(getattr(args, 'iqae_heartbeat_sec', 0.0) or 0.0), hb_tag, hb_state)
    hb.start()

    try:
        hb_state['stage'] = 'build'
        with _timed_block(f"build unitary circuit A (steps={args.steps})"):
            A, _info = build_unitary_circuit_qiskit(
                cfg,
                n_steps=args.steps,
                kernel_profile=_resolved_kernel_profile(args),
            )
    except Exception as e:
        print("[ERROR] Failed to build A circuit.")
        print("Exception:", repr(e))
        return 2

    def _find_qreg(qc, name: str):
        for qr in qc.qregs:
            if qr.name == name:
                return qr
        raise RuntimeError(f"qreg '{name}' not found")

    status_qr = _find_qreg(A, "status")
    g_qr = _find_qreg(A, "g")

    target_g = None
    if getattr(args, "iqae_target_g", -1) is not None and int(args.iqae_target_g) >= 0:
        target_g = int(args.iqae_target_g)

    good = str(getattr(args, "iqae_good", "detect")).strip().lower()
    spec = OracleSpec(good=good, target_g=target_g)

    backend_options = {}
    try:
        backend_options['max_parallel_threads'] = int(getattr(args, 'aer_threads', 1))
        if getattr(args, 'aer_safe_basis', False):
            backend_options['fusion_enable'] = False
        aer_max_mem = int(getattr(args, 'aer_max_mem_mb', 0))
        if aer_max_mem > 0:
            backend_options['max_memory_mb'] = aer_max_mem
        mps_max_bond = int(getattr(args, 'mps_max_bond', 0))
        if mps_max_bond > 0:
            backend_options['matrix_product_state_max_bond_dimension'] = mps_max_bond
        mps_trunc = float(getattr(args, 'mps_trunc', 0.0))
        if mps_trunc > 0.0:
            backend_options['matrix_product_state_truncation_threshold'] = mps_trunc
    except Exception:
        backend_options = {}

    _print_iqae_run_cfg(args, cfg, backend_options, good=good, target_g=target_g, outdir=str(args.outdir))

    hb_state['stage'] = 'iqae_run'
    est, meta = run_iqae_small(
        A_circuit=A,
        status_qubits=status_qr,
        g_qubits=g_qr,
        spec=spec,
        epsilon_target=float(getattr(args, 'iqae_eps', 0.05)),
        alpha=float(getattr(args, 'iqae_alpha', 0.05)),
        shots=int(args.shots),
        use_aer_sampler=bool(getattr(args, 'iqae_use_aer', True)),
        seed=int(args.seed),
        verbose_timing=True,
        backend_options=backend_options,
        debug_sampler=bool(getattr(args, 'iqae_debug', False)),
        sampler_heartbeat_sec=float(getattr(args, 'iqae_heartbeat_sec', 300.0) or 0.0),
        sampler_shot_chunk=int(getattr(args, 'iqae_shot_chunk', 100) or 0),
        sampler_print_ops=bool(getattr(args, 'iqae_debug_print_ops', False)),
        sampler_max_call_seconds=(None if float(getattr(args, 'iqae_max_call_sec', 0.0)) <= 0 else float(getattr(args, 'iqae_max_call_sec', 0.0))),
        progress_state=hb_state,
    )

    ci = None
    try:
        ci = meta.get("confidence_interval", None)
    except Exception:
        ci = None
    if ci is not None:
        try:
            print(f"IQAE confidence interval: [{ci[0]:.6f}, {ci[1]:.6f}] (alpha={meta.get('alpha', None)})")
        except Exception:
            pass

    payload = {
        'phase': 'iqae',
        'steps': int(args.steps),
        'shots': int(args.shots),
        'cfg': {
            'nx': int(cfg.geom.nx), 'ny': int(cfg.geom.ny), 'nz': int(cfg.geom.nz),
            'source': {'x0': int(cfg.source.x0), 'y0': int(cfg.source.y0), 'z0': int(cfg.source.z0), 'g0': int(cfg.source.g0), 'dir0': int(getattr(cfg.source,'dir0',0))},
            'detector_x': int(cfg.geom.detector_x),
        },
        'good': good,
        'target_g': target_g,
        'estimate': float(est),
        'run_cfg': {
            'seed': int(args.seed),
            'dir_tape': bool(getattr(args, 'dir_tape', False)),
            'absorb_tape': bool(getattr(args, 'absorb_tape', False)),
            'term_tape': bool(getattr(args, 'term_tape', False)),
            'scatter_tape': bool(getattr(args, 'scatter_tape', False)),
            'full_history_mode': bool(getattr(args, 'full_history_mode', False)),
            'kernel_profile': _effective_kernel_profile(args),
            'aer_safe_basis': bool(getattr(args, 'aer_safe_basis', False)),
            'aer_threads': int(getattr(args, 'aer_threads', 1)),
            'mps_max_bond': int(getattr(args, 'mps_max_bond', 0)),
            'mps_trunc': float(getattr(args, 'mps_trunc', 0.0)),
            'iqae_shot_chunk': int(getattr(args, 'iqae_shot_chunk', 100) or 0),
            'iqae_heartbeat_sec': float(getattr(args, 'iqae_heartbeat_sec', 300.0) or 0.0),
            'iqae_eps': float(getattr(args, 'iqae_eps', 0.05)),
            'iqae_alpha': float(getattr(args, 'iqae_alpha', 0.05)),
        },
        'meta': meta
        }
    _write_results(args.outdir, payload)

    print("\nIQAE estimate:")
    print(payload)
    return 0

def _apply_sharding(args):

    try:
        shard_id = int(getattr(args, "shard_id", -1))
    except Exception:
        shard_id = -1
    if shard_id < 0:
        return
    try:
        setattr(args, "num_shards", max(1, int(getattr(args, "num_shards", 1) or 1)))
    except Exception:
        pass
    try:
        off = int(getattr(args, "shard_seed_offset", 0) or 0)
        if off > 0 and hasattr(args, "seed") and getattr(args, "seed", None) is not None:
            setattr(args, "seed", int(getattr(args, "seed")) + shard_id * off)
    except Exception:
        pass

    if getattr(args, "outdir", ""):
        args.outdir = os.path.join(args.outdir, "shards", f"shard_{shard_id:03d}")

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--phase", choices=["mc", "circuit", "test", "iqae"], default="mc")
    p.add_argument("--steps", type=int, default=15)
    p.add_argument("--histories", type=int, default=200000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--unitary-matched-energy", action="store_true")
    p.add_argument("--dir-tape", action="store_true",
                   help="Enable per-step direction tape (unitary embedding of classical direction resampling).")
    p.add_argument("--absorb-tape", action="store_true",
                   help="Enable per-step absorption coin tape (cA[t]) to better match classical MC semantics (uses +steps qubits).")
    p.add_argument("--term-tape", action="store_true",
                   help="Enable per-step termination scratch tapes for boundary/detect/absorb writes to avoid term reuse pollution (debug/alignment; uses +2*steps qubits).")
    p.add_argument("--scatter-tape", action="store_true",
                   help="Enable a per-step scatter-event tape tS[t] so scatter histories remain explicitly labeled through the end of the circuit.")
    p.add_argument("--full-history-mode", action="store_true",
                   help="Closer-to-classical stochastic embedding: force dir/absorb/term/scatter tapes on and disable aggressive tape cleanup/compaction.")
    p.add_argument("--kernel-profile", type=str, default="",
                   help="Named kernel profile. Empty/manual keeps the legacy tape flags; other choices expose predicate-network / compression presets.")
    p.add_argument("--test-measure-x", action="store_true",
                   help="In phase=test, also measure final x register and output x histogram.")
    p.add_argument("--test-measure-coins", action="store_true",
                   help="(Phase=test) Also measure interaction/absorption coin tapes (cI[t], cA[t]) and report per-step marginals + consistency checks. Use with --absorb-tape for cA.")
    p.add_argument("--test-measure-dir", action="store_true",
                   help="(Phase=test) Also measure final dir register and report invalid-dir rate (dir in {6,7} should be 0).")
    p.add_argument("--aer-shot-chunk", type=int, default=100,
                   help="(Phase=test) Split Aer sampling into chunks to provide progress. 0 disables chunking.")
    p.add_argument("--aer-heartbeat-sec", type=float, default=300.0,
                   help="(Phase=test) Print a minimal heartbeat every N seconds while Aer is running. 0 disables.")

    p.add_argument("--compiled-qpy", type=str, default="",
                   help="(Phase=test) Load a pre-transpiled circuit from QPY and skip transpile.")
    p.add_argument("--save-compiled-qpy", type=str, default="",
                   help="(Phase=test) Save the transpiled circuit to this QPY path.")
    p.add_argument("--compile-only", action="store_true",
                   help="(Phase=test) Only transpile (and optionally save QPY) then exit.")
    p.add_argument("--skip-transpile", action="store_true",
                   help="(Phase=test) Run without transpile (advanced/debug; not recommended unless the circuit is already in safe basis).")

    p.add_argument(
        "--xt-no-interaction",
        action="store_true",
        help="[DEBUG] Force p_int=0 for all materials (MC and quantum) to test pure transport without collisions.",
    )
    p.add_argument(
        "--xt-disable-detector",
        action="store_true",
        help="[DEBUG] Disable detector ROI check in the quantum circuit (TEST/IQAE only).",
    )
    p.add_argument(
        "--xt-disable-boundary",
        action="store_true",
        help="[DEBUG] Disable boundary check in the quantum circuit (TEST/IQAE only).",
    )
    p.add_argument(
        "--debug-measure-work",
        action="store_true",
        help="[DEBUG] In TEST, also measure work ancillas and report whether they are cleaned (should be 0).",
    )
    p.add_argument("--plot", action="store_true", help="Save Phase-1 figures as PNG")
    p.add_argument("--outdir", type=str, default="", help="Output directory (default: outputs/run_<phase>_<timestamp>)")
    p.add_argument("--shots", type=int, default=20000)

    p.add_argument("--shard-id", type=int, default=-1,
                   help="Shard index (>=0 enables shard mode; results go to <outdir>/shards/shard_XXX).")
    p.add_argument("--num-shards", type=int, default=1,
                   help="Total shards (for metadata only).")
    p.add_argument("--shard-seed-offset", type=int, default=0,
                   help="If >0, auto-set seed = seed + shard_id * shard_seed_offset.")
    p.add_argument("--global-shot-offset", type=int, default=0,
                   help="Absolute starting shot index of this shard in the global run. Used only for layout-invariant chunk seeding in phase=test.")

    p.add_argument("--aer-threads", type=int, default=1,
                   help="Aer parallel threads (set 1 on Windows to reduce native crashes).")
    p.add_argument("--aer-max-mem-mb", type=int, default=0,
                   help="If >0, cap Aer max_memory_mb (helps avoid OOM kill).")
    p.add_argument("--mps-max-bond", type=int, default=0,
                   help="If >0, set matrix_product_state_max_bond_dimension (approximate but prevents blow-up).")
    p.add_argument("--mps-trunc", type=float, default=0.0,
                   help="If >0, set matrix_product_state_truncation_threshold (approximate).")
    p.add_argument("--aer-safe-basis", action="store_true",
                   help="Compile circuits to a conservative basis [rz,sx,x,cx] (more stable, deeper).")

    p.add_argument("--iqae-eps", type=float, default=0.05, help="Target epsilon for IQAE (demo)")
    p.add_argument("--iqae-alpha", type=float, default=0.05, help="Confidence level alpha for IQAE (demo)")
    p.add_argument("--iqae-good", choices=["detect", "absorb", "boundary", "survive"], default="detect",
                   help="Good state definition for IQAE (default: detect)")
    p.add_argument("--iqae-target-g", type=int, default=-1,
                   help="If >=0, estimate P(good & g==target).")
    p.add_argument("--iqae-use-aer", action="store_true", default=True,
                   help="Use Aer(MPS) sampler (recommended; default: True)")

    p.add_argument("--iqae-debug", action="store_true", default=False,
                   help="Enable verbose IQAE sampler debug logs (per-call timing / heartbeat).")
    p.add_argument("--iqae-heartbeat-sec", type=float, default=300.0,
                   help="(Phase=iqae) Print a minimal heartbeat every N seconds while a sampler call is running. 0 disables.")
    p.add_argument("--iqae-shot-chunk", type=int, default=100,
                   help="(Phase=iqae) Split Aer sampling per sampler call into chunks for progress. 0 disables chunking.")
    p.add_argument("--iqae-max-call-sec", type=float, default=0.0,
                   help="If >0, abort if any single sampler call exceeds this many seconds (debug aid).")
    p.add_argument("--iqae-debug-print-ops", action="store_true", default=False,
                   help="When --iqae-debug is on, also print top gate counts (can be slow on very large circuits).")

    p.add_argument("--epsilon", type=float, dest="iqae_eps", help=argparse.SUPPRESS)
    p.add_argument("--alpha", type=float, dest="iqae_alpha", help=argparse.SUPPRESS)
    p.add_argument("--g-target", type=int, dest="iqae_target_g", help=argparse.SUPPRESS)

    return p

def main(argv: Any = None) -> int:
    args = build_argparser().parse_args(argv)

    if args.steps <= 0:
        print("steps must be positive")
        return 2

    if not args.outdir:
        args.outdir = _default_run_outdir(args.phase)

    _apply_sharding(args)
    os.makedirs(args.outdir, exist_ok=True)

    log_path = os.path.join(args.outdir, "run.log")

    with _tee_to_file(log_path):
        if args.phase == "mc":
            return phase_mc(args)
        if args.phase == "circuit":
            return phase_circuit(args)
        if args.phase == "test":
            return phase_test(args)
        if args.phase == "iqae":
            return phase_iqae(args)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
