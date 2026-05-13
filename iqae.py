"""IQAE and Aer sampler helpers for transport amplitude-estimation checks."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Any, Sequence, Tuple, Dict, List

def _ts(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def _import_qiskit():
    from qiskit import QuantumCircuit, QuantumRegister
    from qiskit.circuit.library import GroverOperator
    return (QuantumCircuit, QuantumRegister, GroverOperator)

def _import_qiskit_algorithms():
    try:
        from qiskit_algorithms.amplitude_estimators import IterativeAmplitudeEstimation, EstimationProblem
        return (IterativeAmplitudeEstimation, EstimationProblem)
    except Exception:
        from qiskit_algorithms import IterativeAmplitudeEstimation, EstimationProblem
        return (IterativeAmplitudeEstimation, EstimationProblem)

def _import_aer():
    try:
        from qiskit_aer import AerSimulator
        return AerSimulator
    except Exception as e:
        raise ImportError(f'Could not import qiskit_aer.AerSimulator. Please install qiskit-aer.\n{repr(e)}') from e

@dataclass
class OracleSpec:
    good: str = 'detect'
    target_g: Optional[int] = None
_GOOD_TO_STATUS = {'survive': 0, 'detect': 1, 'absorb': 2, 'boundary': 3}

def build_grover_operator_with_objective_flag(*, A_circuit, status_qubits, g_qubits, spec: OracleSpec, mcx_mode: str='noancilla'):
    QuantumCircuit, QuantumRegister, GroverOperator = _import_qiskit()
    good_key = str(spec.good).strip().lower()
    if good_key not in _GOOD_TO_STATUS:
        raise ValueError(f"Unknown good='{spec.good}'. Expected one of {list(_GOOD_TO_STATUS.keys())}.")
    desired = _GOOD_TO_STATUS[good_key]
    flag = QuantumRegister(1, 'flag')
    full = QuantumCircuit(*A_circuit.qregs, flag, name='A_full')
    full.compose(A_circuit, inplace=True)
    s0, s1 = (status_qubits[0], status_qubits[1])
    if desired >> 0 & 1 == 0:
        full.x(s0)
    if desired >> 1 & 1 == 0:
        full.x(s1)
    controls: List[Any] = [s0, s1]
    if spec.target_g is not None:
        tg = int(spec.target_g)
        if tg < 0 or tg >= 1 << len(g_qubits):
            raise ValueError(f'target_g={tg}target_g is outside the supported g-register range{len(g_qubits)}diagnostic message')
        for i in range(len(g_qubits)):
            if tg >> i & 1 == 0:
                full.x(g_qubits[i])
        controls = controls + list(g_qubits)
    full.mcx(controls, flag[0], mode=mcx_mode)
    if spec.target_g is not None:
        tg = int(spec.target_g)
        for i in range(len(g_qubits)):
            if tg >> i & 1 == 0:
                full.x(g_qubits[i])
    if desired >> 0 & 1 == 0:
        full.x(s0)
    if desired >> 1 & 1 == 0:
        full.x(s1)
    oracle = QuantumCircuit(*A_circuit.qregs, flag, name='O_flagZ')
    oracle.z(flag[0])
    if oracle.num_qubits != full.num_qubits:
        raise RuntimeError(f'oracle.num_qubits({oracle.num_qubits}) != state_preparation.num_qubits({full.num_qubits})')
    grover = GroverOperator(oracle=oracle, state_preparation=full)
    obj_idx = full.find_bit(flag[0]).index
    return (grover, [obj_idx])

class _CountsLike:

    def __init__(self, counts: Dict[str, int]):
        self._counts = dict(counts)

    def get_counts(self, *args, **kwargs) -> Dict[str, int]:
        return dict(self._counts)

class _SamplerData:

    def __init__(self, key: str, value_obj: Any):
        self._key = key
        setattr(self, key, value_obj)

    def keys(self):
        return [self._key]

class _SamplerResultItem:

    def __init__(self, data: _SamplerData, metadata: dict):
        self.data = data
        self.metadata = metadata

class _JobLike:

    def __init__(self, payload_list: list):
        self._payload_list = payload_list

    def result(self):
        return self._payload_list

class _PubsToAerObjectiveSampler:

    def __init__(self, shots: int, objective_qubit_index: int, seed: Optional[int]=None, verbose: bool=False, backend_options: Optional[dict]=None, debug: bool=False, heartbeat_sec: float=60.0, shot_chunk: int=100, print_ops: bool=False, max_call_seconds: Optional[float]=None, progress_state: Optional[dict]=None):
        self._shots = int(shots)
        self._obj = int(objective_qubit_index)
        self._seed = seed
        self._verbose = bool(verbose)
        self._debug = bool(debug)
        self._heartbeat_sec = float(heartbeat_sec) if float(heartbeat_sec) > 0 else 0.0
        self._shot_chunk = int(shot_chunk) if int(shot_chunk) > 0 else 0
        self._print_ops = bool(print_ops)
        self._max_call_seconds = None if max_call_seconds is None or float(max_call_seconds) <= 0 else float(max_call_seconds)
        self._progress_state = progress_state if isinstance(progress_state, dict) else None
        self._call_id = 0
        AerSimulator = _import_aer()
        if seed is None:
            self._backend = AerSimulator(method='matrix_product_state')
        else:
            self._backend = AerSimulator(method='matrix_product_state', seed_simulator=int(seed))
        self._backend_options = backend_options or {}
        try:
            self._backend.set_options(**self._backend_options)
        except Exception as _e:
            if self._verbose:
                _ts(f'[sampler] WARN: failed to set Aer options: {_e!r}')
        from qiskit import QuantumCircuit, ClassicalRegister, transpile
        self._QuantumCircuit = QuantumCircuit
        self._ClassicalRegister = ClassicalRegister
        self._transpile = transpile
        self._safe_basis_gates = ['rz', 'sx', 'x', 'cx']

    def _qc_stats(self, qc, topk: int=10) -> dict:
        try:
            size = qc.size()
        except Exception:
            try:
                size = len(qc.data)
            except Exception:
                size = None
        try:
            depth = qc.depth()
        except Exception:
            depth = None
        try:
            name = getattr(qc, 'name', None)
        except Exception:
            name = None
        ops_top = None
        if self._print_ops:
            try:
                ops = qc.count_ops()
                ops_top = sorted([(str(k), int(v)) for k, v in ops.items()], key=lambda x: (-x[1], x[0]))[:topk]
            except Exception:
                ops_top = None
        return {'name': name, 'num_qubits': getattr(qc, 'num_qubits', None), 'depth': depth, 'size': size, 'ops_top': ops_top}

    def _fmt_ops_top(self, ops_top) -> str:
        if not ops_top:
            return ''
        return ' | top_ops=' + ', '.join([f'{k}={v}' for k, v in ops_top])

    def _compile_for_aer(self, qc, call_id: int, pub_id: int):
        import time as _time
        t0 = _time.time()
        try:
            tqc = qc.decompose(reps=4)
        except Exception:
            tqc = qc
        allowed = set(self._safe_basis_gates + ['measure', 'barrier', 'id'])

        def _bad_ops(_qc):
            try:
                ops = _qc.count_ops()
            except Exception:
                return (['<count_ops_failed>'], None)
            bad = []
            for k in ops.keys():
                ks = str(k)
                if ks not in allowed:
                    bad.append(ks)
            bad = sorted(set(bad))
            return (bad, ops)
        last_ops = None
        for _it in range(8):
            try:
                tqc = self._transpile(tqc, basis_gates=self._safe_basis_gates, optimization_level=0)
            except Exception:
                tqc = self._transpile(tqc, optimization_level=0)
            try:
                tqc = tqc.decompose(reps=10)
            except Exception:
                pass
            try:
                tqc = self._transpile(tqc, basis_gates=self._safe_basis_gates, optimization_level=0)
            except Exception:
                pass
            try:
                params = list(getattr(tqc, 'parameters', []))
            except Exception:
                params = []
            if params:
                raise RuntimeError(f'IQAE transpile left unbound parameters: {params}')
            bad, ops = _bad_ops(tqc)
            last_ops = ops
            if not bad:
                break
            try:
                tqc = tqc.decompose(reps=20)
            except Exception:
                pass
        bad, ops = _bad_ops(tqc)
        if bad:
            raise RuntimeError(f"IQAE compile still contains non-basis ops: {bad} | ops={{{', '.join([f'{str(k)}:{int(v)}' for k, v in (ops or {}).items()])}}}")
        if self._debug:
            st = self._qc_stats(tqc)
            _ts(f"[sampler] call#{call_id} pub#{pub_id} compile done: depth={st['depth']} size={st['size']}" + self._fmt_ops_top(st['ops_top']))
            _ts(f'[sampler] call#{call_id} pub#{pub_id} compile time: {_time.time() - t0:.2f}s')
        return tqc

    def _force_measure_objective_only(self, qc):
        try:
            qc0 = qc.remove_final_measurements(inplace=False)
        except Exception:
            qc0 = qc.copy()
        qcm = self._QuantumCircuit(*qc0.qregs)
        qcm.compose(qc0, inplace=True)
        cobj = self._ClassicalRegister(1, 'cobj')
        qcm.add_register(cobj)
        qcm.measure(qcm.qubits[self._obj], cobj[0])
        return qcm

    def run(self, pubs: Sequence[Tuple[Any, ...]]):
        import time as _time
        from concurrent.futures import TimeoutError as _FuturesTimeoutError
        self._call_id += 1
        call_id = self._call_id
        if self._debug:
            _ts(f'[sampler] call#{call_id} start: pubs={len(pubs)} shots={self._shots}')
        circuits = []
        for pub in pubs:
            if isinstance(pub, tuple) and len(pub) >= 1:
                circuits.append(pub[0])
            else:
                circuits.append(pub)
        items = []
        for i, qc in enumerate(circuits):
            qc_meas = self._force_measure_objective_only(qc)
            if self._debug:
                st0 = self._qc_stats(qc_meas)
                _ts(f"[sampler] call#{call_id} pub#{i} raw: nq={st0['num_qubits']} depth={st0['depth']} size={st0['size']} name={st0['name']}" + self._fmt_ops_top(st0['ops_top']))
            elif self._verbose and (i < 3 or i % 5 == 0):
                _ts(f'[sampler] call#{call_id} pub#{i}: num_qubits={qc_meas.num_qubits}, depth={qc_meas.depth()}')
            tqc = self._compile_for_aer(qc_meas, call_id=call_id, pub_id=i)

            def _run_once_counts(tqc_run):
                t_run0 = _time.time()
                total_shots = int(self._shots)
                shot_chunk = int(self._shot_chunk) if int(self._shot_chunk) > 0 else total_shots
                counts_total = {}
                done = 0
                chunk_id = 0
                if self._progress_state is not None:
                    self._progress_state.update({'stage': 'sampler', 'iter': int(call_id), 'pub': int(i), 'shots_done': 0, 'shots_total': int(total_shots), 'shots_inflight': 0})
                deadline = None if self._max_call_seconds is None else t_run0 + float(self._max_call_seconds)
                while done < total_shots:
                    cur = min(shot_chunk, total_shots - done)
                    run_kwargs = {}
                    if self._seed is not None:
                        run_kwargs['seed_simulator'] = int(self._seed) + int(call_id) * 100000 + int(i) * 1000 + int(chunk_id)
                    if self._progress_state is not None:
                        self._progress_state.update({'stage': 'sampler', 'iter': int(call_id), 'pub': int(i), 'shots_done': int(done), 'shots_total': int(total_shots), 'shots_inflight': int(cur)})
                    job = self._backend.run(tqc_run, shots=int(cur), **run_kwargs)
                    hb = float(self._heartbeat_sec) if self._heartbeat_sec and self._heartbeat_sec > 0 else 0.0
                    _next_hb = _time.time() + hb if hb > 0 else None
                    while True:
                        if deadline is not None and _time.time() >= deadline:
                            raise TimeoutError(f'IQAE sampler call#{call_id} pub#{i} exceeded max_call_seconds={self._max_call_seconds}')
                        try:
                            if job.done():
                                break
                        except Exception:
                            break
                        if hb > 0 and _next_hb is not None and (_time.time() >= _next_hb):
                            _ts(f'[HB] iqae iter={call_id} pub={i} shots={done}+{cur}/{total_shots}')
                            _next_hb += hb
                        _time.sleep(0.25)
                    res = job.result()
                    c = res.get_counts()
                    for k, v in c.items():
                        kk = str(k)
                        counts_total[kk] = int(counts_total.get(kk, 0)) + int(v)
                    done += int(cur)
                    chunk_id += 1
                    if self._progress_state is not None:
                        self._progress_state.update({'stage': 'sampler', 'iter': int(call_id), 'pub': int(i), 'shots_done': int(done), 'shots_total': int(total_shots), 'shots_inflight': 0})
                    _ts(f'[PROG] iqae iter={call_id} pub={i} shots={done}/{total_shots}')
                if self._debug:
                    _ts(f'[sampler] call#{call_id} pub#{i} run done: {_time.time() - t_run0:.2f}s')
                return counts_total
            try:
                counts = _run_once_counts(tqc)
            except Exception:
                try:
                    qc2 = qc_meas.decompose(reps=20)
                except Exception:
                    qc2 = qc_meas
                tqc2 = self._compile_for_aer(qc2, call_id=call_id, pub_id=i)
                counts = _run_once_counts(tqc2)
            if not counts:
                raise RuntimeError('Sampler got empty counts; check measurement insertion.')
            n0 = 0
            n1 = 0
            for k, v in counts.items():
                kk = str(k).replace(' ', '')
                bit = kk[-1] if kk else '0'
                if bit == '1':
                    n1 += int(v)
                else:
                    n0 += int(v)
            counts_obj = _CountsLike({'0': n0, '1': n1})
            data = _SamplerData('meas', counts_obj)
            metadata = {'shots': int(self._shots)}
            items.append(_SamplerResultItem(data=data, metadata=metadata))
        return _JobLike(items)

def run_iqae_small(*, A_circuit, status_qubits, g_qubits, spec: OracleSpec, epsilon_target: float=0.05, alpha: float=0.05, shots: int=2000, use_aer_sampler: bool=True, seed: Optional[int]=None, verbose_timing: bool=True, backend_options: Optional[dict]=None, debug_sampler: bool=False, sampler_heartbeat_sec: float=60.0, sampler_shot_chunk: int=100, sampler_print_ops: bool=False, sampler_max_call_seconds: Optional[float]=None, progress_state: Optional[dict]=None):
    IterativeAmplitudeEstimation, EstimationProblem = _import_qiskit_algorithms()
    t0 = datetime.now().timestamp()
    if verbose_timing:
        _ts(f'IQAE: build Grover operator (good={spec.good}, target_g={spec.target_g})')
    Q, objective_qubits = build_grover_operator_with_objective_flag(A_circuit=A_circuit, status_qubits=status_qubits, g_qubits=g_qubits, spec=spec, mcx_mode='noancilla')
    t1 = datetime.now().timestamp()
    problem = EstimationProblem(state_preparation=Q.state_preparation, grover_operator=Q, objective_qubits=objective_qubits)
    iae = IterativeAmplitudeEstimation(epsilon_target=float(epsilon_target), alpha=float(alpha))
    if not use_aer_sampler:
        raise RuntimeError('diagnostic message')
    iae._sampler = _PubsToAerObjectiveSampler(shots=int(shots), objective_qubit_index=int(objective_qubits[0]), seed=seed, verbose=bool(verbose_timing), backend_options=backend_options, debug=bool(debug_sampler), heartbeat_sec=float(sampler_heartbeat_sec), shot_chunk=int(sampler_shot_chunk), print_ops=bool(sampler_print_ops), max_call_seconds=sampler_max_call_seconds, progress_state=progress_state)
    if verbose_timing:
        _ts('IQAE: start estimate()')
    t2 = datetime.now().timestamp()
    result = iae.estimate(problem)
    t3 = datetime.now().timestamp()
    ci = None
    try:
        ci = getattr(result, 'confidence_interval', None)
        if callable(ci):
            ci = ci()
    except Exception:
        ci = None
    num_oracle_queries = None
    for key in ['num_oracle_queries', 'num_oracle_calls', 'num_queries']:
        try:
            v = getattr(result, key, None)
            if v is not None:
                num_oracle_queries = int(v)
                break
        except Exception:
            pass
    if verbose_timing:
        _ts(f'IQAE: done estimate()')
    meta = {'good': str(spec.good), 'epsilon_target': float(epsilon_target), 'alpha': float(alpha), 'target_g': None if spec.target_g is None else int(spec.target_g), 'objective_qubits': list(map(int, objective_qubits)), 'confidence_interval': None if ci is None else [float(ci[0]), float(ci[1])], 'num_oracle_queries': None if num_oracle_queries is None else int(num_oracle_queries), 'timing_sec': {'build_grover': float(t1 - t0), 'estimate': float(t3 - t2), 'total': float(t3 - t0)}}
    return (float(result.estimation), meta)
