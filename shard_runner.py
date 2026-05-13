"""Launch and manage sharded benchmark runs."""


from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import hashlib
import sys
import time
from glob import glob
from pathlib import Path
from typing import List, Dict, Any

def _now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def _append_line(path: Path, msg: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"[{_now_ts()}] {msg}\n")

def _hash_compile_key(*, steps: int, extra: str, cwd: Path) -> str:

    h = hashlib.sha1()
    h.update(f"steps={int(steps)}\n".encode("utf-8"))
    h.update(f"extra={str(extra)}\n".encode("utf-8"))
    for fn in ["main.py", "unitary_circuit.py", "config.py", "geometry.py", "iqae.py"]:
        p = cwd / fn
        if p.exists():
            h.update(fn.encode("utf-8") + b"\0")
            h.update(p.read_bytes())
    return h.hexdigest()[:8]

def _env_guard() -> None:

    os.environ.setdefault("RAYON_NUM_THREADS", os.environ.get("RAYON_NUM_THREADS", "1"))
    os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "1"))
    os.environ.setdefault("MKL_NUM_THREADS", os.environ.get("MKL_NUM_THREADS", "1"))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", os.environ.get("OPENBLAS_NUM_THREADS", "1"))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", os.environ.get("NUMEXPR_NUM_THREADS", "1"))

def _parse_extra(extra: str) -> List[str]:
    if not extra:
        return []

    return shlex.split(extra, posix=False)

def _extract_int_arg(extra_args: List[str], key: str, default: int) -> int:
    try:
        if key in extra_args:
            i = extra_args.index(key)
            if i + 1 < len(extra_args):
                return int(extra_args[i + 1])
    except Exception:
        pass
    return int(default)

def _run_cmd(cmd: List[str], log_path: Path, cwd: Path) -> subprocess.Popen:

    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault('PYTHONUNBUFFERED', '1')
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("CMD: " + " ".join(cmd) + "\n\n")
        f.flush()
        p = subprocess.Popen(cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT, env=env)
    return p

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["test", "mc", "iqae"], required=True)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--num-shards", type=int, default=20)
    ap.add_argument("--max-parallel", type=int, default=6)
    ap.add_argument("--seed0", type=int, default=1234)
    ap.add_argument("--outroot", type=str, required=True)
    ap.add_argument("--shots-per-shard", type=int, default=200)
    ap.add_argument("--total-shots", type=int, default=0, help="Exact total test/iqae shots to distribute across shards. If >0, overrides --shots-per-shard.")
    ap.add_argument("--shot-chunk", type=int, default=0, help="Canonical global shot chunk size. For test phase, if >0 and --total-shots>0, shards are assigned whole chunks to improve layout invariance.")
    ap.add_argument("--histories-per-shard", type=int, default=200000)
    ap.add_argument("--extra", type=str, default="")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--compile-cache", action="store_true", help="Enable compile cache (test only).")
    grp.add_argument("--no-compile-cache", action="store_true", help="Disable compile cache (test only).")
    args = ap.parse_args()

    _env_guard()

    cwd = Path(__file__).resolve().parent
    outroot = Path(args.outroot)
    outroot.mkdir(parents=True, exist_ok=True)

    if args.phase != "test":
        compile_cache = False
    else:

        compile_cache = (not args.no_compile_cache)

    runner_log = outroot / "runner.log"
    extra_args = _parse_extra(args.extra)
    aer_threads = _extract_int_arg(extra_args, "--aer-threads", 1)
    est_cores = int(args.max_parallel) * int(max(1, aer_threads))
    _append_line(runner_log, f"START phase={args.phase} steps={args.steps} num_shards={args.num_shards} max_parallel={args.max_parallel} aer_threads={aer_threads} est_cores={est_cores} python={sys.executable}")
    _append_line(runner_log, "NOTE: shards are launched in waves (queue). Not all shards start at the same wall-clock time.")
    _append_line(runner_log, f"extra={args.extra}")
    if args.phase == "test":
        _append_line(runner_log, "NOTE: test shards share the same root seed; chunk seeds are derived from the global shot offset for layout-invariant sampling.")

    if args.total_shots and int(args.total_shots) > 0 and args.phase in ("test", "iqae"):
        _append_line(runner_log, f"NOTE: exact total_shots mode enabled total_shots={int(args.total_shots)} shot_chunk={int(args.shot_chunk)}")

    shard_plan = []
    plan_meta = {
        "phase": str(args.phase),
        "requested_shards": int(args.num_shards),
        "active_planned_shards": 0,
        "empty_planned_shards": 0,
        "total_shots": 0,
        "canonical_chunk_mode": False,
        "shot_chunk": int(args.shot_chunk or 0),
        "n_full_chunks": 0,
        "remainder_shots": 0,
        "per_shard_chunks_base": 0,
        "per_shard_chunks_extra": 0,
        "shots_per_shard_legacy": 0,
    }
    if args.phase in ("test", "iqae") and int(args.total_shots or 0) > 0:
        total_shots = int(args.total_shots)
        num_shards = int(args.num_shards)
        shot_chunk = int(args.shot_chunk or 0)
        plan_meta["total_shots"] = int(total_shots)
        if args.phase == "test" and shot_chunk > 0:

            n_full = total_shots // shot_chunk
            rem = total_shots % shot_chunk
            base = n_full // num_shards
            extra = n_full % num_shards
            g_off = 0
            for sid in range(num_shards):
                n_chunks = base + (1 if sid < extra else 0)
                s = n_chunks * shot_chunk
                shard_plan.append({"sid": sid, "shots": s, "global_offset": g_off, "n_chunks": n_chunks})
                g_off += s
            if rem > 0:

                shard_plan[-1]["shots"] += rem
            plan_meta.update({
                "canonical_chunk_mode": True,
                "n_full_chunks": int(n_full),
                "remainder_shots": int(rem),
                "per_shard_chunks_base": int(base),
                "per_shard_chunks_extra": int(extra),
            })
            _append_line(runner_log, f"PLAN test canonical-chunk mode: n_full_chunks={n_full} rem={rem} per_shard_chunks~{base}/{base+1 if extra>0 else base}")
        else:

            base = total_shots // num_shards
            extra = total_shots % num_shards
            g_off = 0
            for sid in range(num_shards):
                s = base + (1 if sid < extra else 0)
                shard_plan.append({"sid": sid, "shots": s, "global_offset": g_off})
                g_off += s
            plan_meta.update({
                "canonical_chunk_mode": False,
                "remainder_shots": int(extra),
            })
            _append_line(runner_log, f"PLAN exact-shot mode: base={base} extra={extra}")
    else:

        g_off = 0
        for sid in range(int(args.num_shards)):
            s = int(args.shots_per_shard)
            shard_plan.append({"sid": sid, "shots": s, "global_offset": g_off})
            g_off += s
        plan_meta.update({
            "total_shots": int(g_off),
            "shots_per_shard_legacy": int(args.shots_per_shard),
        })
        _append_line(runner_log, f"PLAN fixed-shots mode: shots_per_shard={int(args.shots_per_shard)} total_shots={g_off}")

    plan_meta["active_planned_shards"] = int(sum(1 for x in shard_plan if int(x.get("shots", 0)) > 0))
    plan_meta["empty_planned_shards"] = int(plan_meta["requested_shards"] - plan_meta["active_planned_shards"])
    _append_line(runner_log, f"PLAN_META requested={plan_meta['requested_shards']} active={plan_meta['active_planned_shards']} empty={plan_meta['empty_planned_shards']} total_shots={plan_meta['total_shots']} shot_chunk={plan_meta['shot_chunk']}")
    with open(outroot / "plan_meta.json", "w", encoding="utf-8") as f:
        import json as _json
        _json.dump(plan_meta, f, indent=2, ensure_ascii=False)

    compiled_key = _hash_compile_key(steps=args.steps, extra=args.extra, cwd=cwd)
    compiled_qpy = outroot / "compiled" / f"test_steps{args.steps}_safe_{compiled_key}.qpy"

    if args.phase == "test" and compile_cache:
        compile_cmd = [
            sys.executable,
            "-u",
            "main.py",
            "--phase",
            "test",
            "--steps",
            str(args.steps),
            "--shots",
            "1",
            "--seed",
            str(args.seed0),
            "--compile-only",
            "--save-compiled-qpy",
            str(compiled_qpy),
            "--outdir",
            str(outroot),
        ]
        compile_cmd += extra_args
        compile_log = outroot / "compiled" / "compile.log"
        print("[RUN] compile cache:", " ".join(compile_cmd))
        _append_line(runner_log, f"COMPILE cmd={' '.join(compile_cmd)}")
        p = _run_cmd(compile_cmd, log_path=compile_log, cwd=cwd)
        rc = p.wait()
        if rc != 0:
            print("[ERROR] compile-only failed. See:", str(compile_log))
            _append_line(runner_log, f"COMPILE_FAILED rc={rc}")
            return rc

    running: List[Dict[str, Any]] = []
    failures = 0
    launched = 0
    finished = 0
    skipped = 0

    hb_sec = 300.0
    next_hb = time.time() + hb_sec

    def maybe_heartbeat() -> None:
        nonlocal next_hb
        now = time.time()
        if hb_sec > 0 and now >= next_hb:
            done_cnt = int(finished + skipped)
            total = int(args.num_shards)
            pending = max(0, total - done_cnt - len(running))
            msg = f"[HB] runner phase={args.phase} done={done_cnt}/{total} active={len(running)} pending={pending} failures={failures}"
            print(msg, flush=True)
            _append_line(runner_log, msg)
            next_hb = now + hb_sec

    def reap_finished() -> None:
        nonlocal failures
        nonlocal finished
        still = []
        for item in running:
            p = item["p"]
            sid = item["sid"]
            rc = p.poll()
            if rc is None:
                still.append(item)
            else:
                dt = time.time() - float(item["t0"])
                _append_line(runner_log, f"DONE shard={sid:03d} rc={rc} elapsed_sec={dt:.2f}")
                finished += 1
                if rc != 0:
                    failures += 1
        running[:] = still

    for plan in shard_plan:
        sid = int(plan["sid"])
        shard_shots = int(plan.get("shots", 0))
        global_offset = int(plan.get("global_offset", 0))
        if shard_shots <= 0:
            _append_line(runner_log, f"SKIP shard={sid:03d} (planned shots <= 0)")
            skipped += 1
            continue
        shard_dir = outroot / "shards" / f"shard_{sid:03d}"
        res_json = shard_dir / "results.json"
        if res_json.exists():
            _append_line(runner_log, f"SKIP shard={sid:03d} (results.json exists)")
            skipped += 1
            continue

        while True:
            reap_finished()
            maybe_heartbeat()
            if len(running) < int(args.max_parallel):
                break
            time.sleep(0.5)

        if args.phase == "test":
            cmd = [
                sys.executable,
                "-u",
                "main.py",
                "--phase",
                "test",
                "--steps",
                str(args.steps),
                "--shots",
                str(shard_shots),
                "--seed",
                str(args.seed0),
                "--shard-id",
                str(sid),
                "--num-shards",
                str(args.num_shards),
                "--outdir",
                str(outroot),
                "--global-shot-offset",
                str(global_offset),
            ]
            cmd += extra_args
            if compile_cache and compiled_qpy.exists():
                cmd += ["--compiled-qpy", str(compiled_qpy)]
        elif args.phase == "mc":
            cmd = [
                sys.executable,
                "-u",
                "main.py",
                "--phase",
                "mc",
                "--steps",
                str(args.steps),
                "--histories",
                str(args.histories_per_shard),
                "--seed",
                str(args.seed0 + sid),
                "--shard-id",
                str(sid),
                "--num-shards",
                str(args.num_shards),
                "--outdir",
                str(outroot),
                "--global-shot-offset",
                str(global_offset),
            ]
            cmd += extra_args
        else:
            cmd = [
                sys.executable,
                "-u",
                "main.py",
                "--phase",
                "iqae",
                "--steps",
                str(args.steps),
                "--shots",
                str(shard_shots),
                "--seed",
                str(args.seed0 + sid),
                "--shard-id",
                str(sid),
                "--num-shards",
                str(args.num_shards),
                "--outdir",
                str(outroot),
                "--global-shot-offset",
                str(global_offset),
            ]
            cmd += extra_args

        shard_log = shard_dir / "run.log"
        print("[RUN] shard", sid, ":", " ".join(cmd))
        _append_line(runner_log, f"LAUNCH shard={sid:03d} cmd={' '.join(cmd)}")
        p = _run_cmd(cmd, log_path=shard_log, cwd=cwd)
        running.append({"sid": sid, "p": p, "t0": time.time()})
        launched += 1

    while running:
        reap_finished()
        maybe_heartbeat()
        time.sleep(0.5)

    _append_line(runner_log, f"ALL_DONE launched={launched} failures={failures}")
    print(f"[DONE] shards finished. launched={launched} failures={failures}")

    results = glob(str(outroot / "shards" / "shard_*" / "results.json"))
    if len(results) == 0:
        _append_line(runner_log, "MERGE_SKIPPED (no results.json found)")
        print("[WARN] No results.json found; skip merge.")
        return 0 if failures == 0 else 2

    try:
        _append_line(runner_log, f"MERGE_START phase={args.phase}")
        merge_cmd = [
            sys.executable,
            "merge_shards.py",
            "--phase",
            args.phase,
            "--in",
            str(outroot / "shards"),
            "--out",
            str(outroot / "merged"),
        ]
        print("[RUN] merge:", " ".join(merge_cmd))
        cp = subprocess.run(merge_cmd, cwd=str(cwd), check=False)
        merged_dir = outroot / "merged"
        merged_res = merged_dir / "results_merged.json"
        merged_sum = merged_dir / "summary.txt"
        if cp.returncode != 0:
            _append_line(runner_log, f"MERGE_FAILED rc={cp.returncode}")
            print("[WARN] merge returned non-zero rc=", cp.returncode)
        if not merged_res.exists():
            _append_line(runner_log, "MERGE_FAILED (results_merged.json missing)")
            print("[WARN] merge did not produce results_merged.json:", str(merged_res))
            return 3 if failures == 0 else 3
        _append_line(runner_log, f"MERGE_DONE phase={args.phase} out={str(merged_dir)}")

    except Exception as e:
        _append_line(runner_log, f"MERGE_FAILED err={repr(e)}")
        print("[WARN] merge failed:", repr(e))

    return 0 if failures == 0 else 2

if __name__ == "__main__":
    raise SystemExit(main())
