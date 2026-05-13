"""Merge sharded MC or TEST outputs into a single result summary."""


import argparse
import json
import os
from glob import glob
from typing import Dict, Any, List, Tuple
import math

def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _safe_int(x, default=0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)

def _binom_se(p: float, n: int) -> float:
    if n <= 0:
        return 0.0
    p = max(0.0, min(1.0, float(p)))
    return math.sqrt(p * (1.0 - p) / float(n))

def _sum_hist_dict(dicts: List[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for d in dicts:
        if not isinstance(d, dict):
            continue
        for k, v in d.items():
            out[str(k)] = out.get(str(k), 0) + _safe_int(v, 0)
    return out

def _read_plan_meta(indir: str) -> Dict[str, Any]:
    try:
        root = os.path.dirname(os.path.abspath(indir))
        p = os.path.join(root, "plan_meta.json")
        if os.path.exists(p):
            return _read_json(p)
    except Exception:
        pass
    return {}

def merge_test(files: List[str]) -> Dict[str, Any]:

    total_shots = 0
    n_detect = n_absorb = n_boundary = 0
    n_alive = 0

    x_total_list = []
    x_alive_list = []
    x_detect_list = []
    x_absorb_list = []
    x_boundary_list = []

    diag_list = []

    ci1_sum = None
    ca1_sum = None
    ca1_ci0_total = 0
    nonabsorb_with_ca_total = 0
    absorb_without_ca_total = 0

    dir_count_total: Dict[str, int] = {}
    invalid_dir_total = 0

    nonzero_total_hits = 0.0
    nonzero_alive_hits = 0.0

    meta0 = None
    steps0 = None
    aer_shot_chunk_vals = []
    seed_strategy_vals = []
    global_shot_offset_vals = []
    for fp in files:
        js = _read_json(fp)
        if meta0 is None:
            meta0 = js
            steps0 = js.get("steps", None)
        if "aer_shot_chunk" in js:
            aer_shot_chunk_vals.append(_safe_int(js.get("aer_shot_chunk", 0), 0))
        if "seed_strategy" in js:
            seed_strategy_vals.append(str(js.get("seed_strategy", "")))
        if "global_shot_offset" in js:
            global_shot_offset_vals.append(_safe_int(js.get("global_shot_offset", 0), 0))

        shots = _safe_int(js.get("shots", 0))
        total_shots += shots

        n_alive_i = _safe_int(js.get("n_alive", round(float(js.get("p_survive", 0.0)) * shots)))
        n_detect += _safe_int(js.get("n_detect", round(float(js.get("p_detect", 0.0)) * shots)))
        n_absorb += _safe_int(js.get("n_absorb", round(float(js.get("p_absorb", 0.0)) * shots)))
        n_boundary += _safe_int(js.get("n_boundary", round(float(js.get("p_boundary", 0.0)) * shots)))
        n_alive += n_alive_i

        x_total_list.append(js.get("x_count_total", {}))
        x_alive_list.append(js.get("x_count_alive", {}))
        x_detect_list.append(js.get("x_count_detect", {}))
        x_absorb_list.append(js.get("x_count_absorb", {}))
        x_boundary_list.append(js.get("x_count_boundary", {}))

        if isinstance(js.get('diag', None), dict):
            diag_list.append(js.get('diag'))

        cs = js.get('coin_stats', None)
        if isinstance(cs, dict):
            ci1 = cs.get('ci1', None)
            ca1 = cs.get('ca1', None)
            if isinstance(ci1, list):
                if ci1_sum is None:
                    ci1_sum = [0] * len(ci1)
                for k in range(min(len(ci1_sum), len(ci1))):
                    ci1_sum[k] += _safe_int(ci1[k], 0)
            if isinstance(ca1, list):
                if ca1_sum is None:
                    ca1_sum = [0] * len(ca1)
                for k in range(min(len(ca1_sum), len(ca1))):
                    ca1_sum[k] += _safe_int(ca1[k], 0)
            ca1_ci0_total += _safe_int(cs.get('ca1_ci0_total', 0), 0)
            nonabsorb_with_ca_total += _safe_int(cs.get('nonabsorb_with_ca_total', 0), 0)
            absorb_without_ca_total += _safe_int(cs.get('absorb_without_ca_total', 0), 0)

        ds = js.get('dir_stats', None)
        if isinstance(ds, dict):
            dct = ds.get('dir_count_total', None)
            if isinstance(dct, dict):
                for k, v in dct.items():
                    ks = str(k)
                    dir_count_total[ks] = dir_count_total.get(ks, 0) + _safe_int(v, 0)
            invalid_dir_total += _safe_int(ds.get('invalid_dir_total', 0), 0)

        try:
            r_tot = float(js.get("work_nonzero_rate_total", 0.0) or 0.0)
        except Exception:
            r_tot = 0.0
        try:
            r_alive = float(js.get("work_nonzero_rate_alive", 0.0) or 0.0)
        except Exception:
            r_alive = 0.0

        nonzero_total_hits += r_tot * float(shots)
        nonzero_alive_hits += r_alive * float(n_alive_i)

    if total_shots <= 0:
        return {"phase": "test", "shots": 0, "steps": steps0, "total_shards": len(files)}

    x_count_total = _sum_hist_dict(x_total_list)
    x_count_alive = _sum_hist_dict(x_alive_list)
    x_count_detect = _sum_hist_dict(x_detect_list)
    x_count_absorb = _sum_hist_dict(x_absorb_list)
    x_count_boundary = _sum_hist_dict(x_boundary_list)

    p_detect = float(n_detect) / float(total_shots)
    p_absorb = float(n_absorb) / float(total_shots)
    p_boundary = float(n_boundary) / float(total_shots)
    p_survive = float(n_alive) / float(total_shots)

    merged: Dict[str, Any] = {
        "phase": "test",
        "steps": steps0,
        "shots": int(total_shots),
        "total_shards": int(len(files)),
        "n_detect": int(n_detect),
        "n_absorb": int(n_absorb),
        "n_boundary": int(n_boundary),
        "n_alive": int(n_alive),
        "p_detect": p_detect,
        "p_absorb": p_absorb,
        "p_boundary": p_boundary,
        "p_survive": p_survive,
        "se_detect": _binom_se(p_detect, total_shots),
        "se_absorb": _binom_se(p_absorb, total_shots),
        "se_boundary": _binom_se(p_boundary, total_shots),
        "se_survive": _binom_se(p_survive, total_shots),
        "x_count_total": x_count_total,
        "x_count_alive": x_count_alive,
        "x_count_detect": x_count_detect,
        "x_count_absorb": x_count_absorb,
        "x_count_boundary": x_count_boundary,
        "work_nonzero_rate_total": (nonzero_total_hits / float(total_shots)),
        "work_nonzero_rate_alive": (0.0 if n_alive <= 0 else (nonzero_alive_hits / float(n_alive))),
    }

    if aer_shot_chunk_vals:
        merged["aer_shot_chunk"] = int(aer_shot_chunk_vals[0]) if len(set(aer_shot_chunk_vals)) == 1 else [int(v) for v in aer_shot_chunk_vals]
    if seed_strategy_vals:
        merged["seed_strategy"] = seed_strategy_vals[0] if len(set(seed_strategy_vals)) == 1 else seed_strategy_vals
    if global_shot_offset_vals:
        merged["global_shot_offset_range"] = {
            "min": int(min(global_shot_offset_vals)),
            "max": int(max(global_shot_offset_vals)),
        }

    if ci1_sum is not None or ca1_sum is not None or ca1_ci0_total or nonabsorb_with_ca_total or absorb_without_ca_total:
        merged['coin_stats'] = {
            'ci1': ([int(v) for v in ci1_sum] if ci1_sum is not None else None),
            'ca1': ([int(v) for v in ca1_sum] if ca1_sum is not None else None),
            'p_ci1': ([(float(v) / float(total_shots)) for v in ci1_sum] if ci1_sum is not None and total_shots > 0 else None),
            'p_ca1': ([(float(v) / float(total_shots)) for v in ca1_sum] if ca1_sum is not None and total_shots > 0 else None),
            'ca1_ci0_total': int(ca1_ci0_total),
            'nonabsorb_with_ca_total': int(nonabsorb_with_ca_total),
            'absorb_without_ca_total': int(absorb_without_ca_total),
            'ca1_ci0_rate': (float(ca1_ci0_total) / float(total_shots) if total_shots > 0 else 0.0),
            'nonabsorb_with_ca_rate': (float(nonabsorb_with_ca_total) / float(total_shots) if total_shots > 0 else 0.0),
            'absorb_without_ca_rate': (float(absorb_without_ca_total) / float(total_shots) if total_shots > 0 else 0.0),
        }

    if dir_count_total or invalid_dir_total:
        merged['dir_stats'] = {
            'dir_count_total': {str(k): int(v) for k, v in sorted(dir_count_total.items(), key=lambda kv: int(kv[0]))},
            'invalid_dir_total': int(invalid_dir_total),
            'invalid_dir_rate': (float(invalid_dir_total) / float(total_shots) if total_shots > 0 else 0.0),
        }

    if x_count_total:
        merged["x_prob_total"] = {str(k): float(v) / float(total_shots) for k, v in x_count_total.items()}
    if x_count_alive and n_alive > 0:
        merged["x_prob_alive"] = {str(k): float(v) / float(n_alive) for k, v in x_count_alive.items()}
    if x_count_detect and n_detect > 0:
        merged["x_prob_detect"] = {str(k): float(v) / float(n_detect) for k, v in x_count_detect.items()}
    if x_count_absorb and n_absorb > 0:
        merged["x_prob_absorb"] = {str(k): float(v) / float(n_absorb) for k, v in x_count_absorb.items()}
    if x_count_boundary and n_boundary > 0:
        merged["x_prob_boundary"] = {str(k): float(v) / float(n_boundary) for k, v in x_count_boundary.items()}

    if diag_list:
        d = {}
        try:
            d['unreachable_detect'] = any(bool(x.get('unreachable_detect', False)) for x in diag_list)
            d['alive_on_x_boundary'] = any(bool(x.get('alive_on_x_boundary', False)) for x in diag_list)
            d['alive_at_x0'] = sum(_safe_int(x.get('alive_at_x0', 0), 0) for x in diag_list)
            d['alive_at_xmax'] = sum(_safe_int(x.get('alive_at_xmax', 0), 0) for x in diag_list)

            for x in diag_list:
                if 'reach_dist' in x:
                    d['reach_dist'] = _safe_int(x.get('reach_dist', 0), 0)
                    break
        except Exception:
            d = {}
        merged['diag'] = d

    return merged

def merge_mc(files: List[str]) -> Dict[str, Any]:

    total_hist = 0
    n_detect = n_absorb = n_boundary = n_survive = 0
    unitary_matched = True

    sum_e = None
    total_detect_for_e = 0

    meta0 = None
    steps0 = None
    aer_shot_chunk_vals = []
    seed_strategy_vals = []
    global_shot_offset_vals = []
    for fp in files:
        js = _read_json(fp)
        if meta0 is None:
            meta0 = js
            steps0 = js.get("steps", None)
        if "aer_shot_chunk" in js:
            aer_shot_chunk_vals.append(_safe_int(js.get("aer_shot_chunk", 0), 0))
        if "seed_strategy" in js:
            seed_strategy_vals.append(str(js.get("seed_strategy", "")))
        if "global_shot_offset" in js:
            global_shot_offset_vals.append(_safe_int(js.get("global_shot_offset", 0), 0))

        hist = _safe_int(js.get("histories", 0))
        total_hist += hist

        counts = js.get("counts", {}) if isinstance(js.get("counts", {}), dict) else {}
        d = _safe_int(counts.get("detect", js.get("n_detect", 0)), 0)
        a = _safe_int(counts.get("absorb", js.get("n_absorb", 0)), 0)
        b = _safe_int(counts.get("boundary", js.get("n_boundary", 0)), 0)
        s = _safe_int(counts.get("survive", js.get("n_survive", 0)), 0)

        n_detect += d
        n_absorb += a
        n_boundary += b
        n_survive += s

        if "unitary_matched_energy" in js:
            try:
                unitary_matched = unitary_matched and bool(js.get("unitary_matched_energy"))
            except Exception:
                pass

        e = js.get("energy_hist_detect", None)
        if isinstance(e, list) and len(e) > 0 and d > 0:
            if sum_e is None:
                sum_e = [0.0 for _ in range(len(e))]
            for i in range(min(len(sum_e), len(e))):
                try:
                    sum_e[i] += float(e[i]) * float(d)
                except Exception:
                    pass
            total_detect_for_e += d

    if total_hist <= 0:
        return {"phase": "mc", "histories": 0, "steps": steps0, "total_shards": len(files)}

    p_detect = float(n_detect) / float(total_hist)
    p_absorb = float(n_absorb) / float(total_hist)
    p_boundary = float(n_boundary) / float(total_hist)
    p_survive = float(n_survive) / float(total_hist)

    out: Dict[str, Any] = {
        "phase": "mc",
        "steps": steps0,
        "histories": int(total_hist),
        "total_shards": int(len(files)),
        "unitary_matched_energy": bool(unitary_matched),
        "counts": {"detect": int(n_detect), "absorb": int(n_absorb), "boundary": int(n_boundary), "survive": int(n_survive)},
        "p_detect": p_detect,
        "p_absorb": p_absorb,
        "p_boundary": p_boundary,
        "p_survive": p_survive,
        "se_detect": _binom_se(p_detect, total_hist),
        "se_absorb": _binom_se(p_absorb, total_hist),
        "se_boundary": _binom_se(p_boundary, total_hist),
        "se_survive": _binom_se(p_survive, total_hist),
    }

    if sum_e is not None and total_detect_for_e > 0:
        out["energy_hist_detect"] = [x / float(total_detect_for_e) for x in sum_e]
    else:

        if meta0 and isinstance(meta0.get("energy_hist_detect", None), list):
            out["energy_hist_detect"] = [0.0 for _ in meta0.get("energy_hist_detect")]
        else:
            out["energy_hist_detect"] = []

    return out

def merge_iqae(files: List[str]) -> Dict[str, Any]:
    ests = []
    meta0 = None
    for fp in files:
        js = _read_json(fp)
        if meta0 is None:
            meta0 = js
        if "estimate" in js:
            try:
                ests.append(float(js["estimate"]))
            except Exception:
                pass

    out = {
        "phase": "iqae_merged",
        "steps": int(meta0.get("steps", 0)) if meta0 else 0,
        "total_shards": len(files),
        "estimates": ests,
    }
    if ests:
        mu = sum(ests) / len(ests)
        var = sum((x - mu) ** 2 for x in ests) / max(1, len(ests) - 1)
        out["estimate_mean"] = mu
        out["estimate_std"] = math.sqrt(var)
        out["estimate_sem"] = out["estimate_std"] / math.sqrt(len(ests))
        out["estimate_min"] = min(ests)
        out["estimate_max"] = max(ests)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["test", "mc", "iqae"], required=True)
    ap.add_argument("--in", dest="indir", required=True, help="Input shard folder, e.g. outputs\\run\\shards")
    ap.add_argument("--out", dest="outdir", required=True, help="Output folder, e.g. outputs\\run\\merged")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    files = sorted(glob(os.path.join(args.indir, "shard_*", "results.json")))
    plan_meta = _read_plan_meta(args.indir)
    if not files:
        raise SystemExit(f"No results.json found under: {args.indir}")

    if args.phase == "test":
        merged = merge_test(files)
    elif args.phase == "mc":
        merged = merge_mc(files)
    else:
        merged = merge_iqae(files)

    if isinstance(plan_meta, dict) and plan_meta:
        merged["requested_shards"] = _safe_int(plan_meta.get("requested_shards", merged.get("total_shards", len(files))), len(files))
        merged["executed_shards"] = int(len(files))
        merged["empty_planned_shards"] = _safe_int(plan_meta.get("empty_planned_shards", 0), 0)
        if "canonical_chunk_mode" in plan_meta:
            merged["canonical_chunk_mode"] = bool(plan_meta.get("canonical_chunk_mode"))
        if "n_full_chunks" in plan_meta:
            merged["n_full_chunks"] = _safe_int(plan_meta.get("n_full_chunks", 0), 0)
        if "remainder_shots" in plan_meta:
            merged["remainder_shots"] = _safe_int(plan_meta.get("remainder_shots", 0), 0)
        if "shot_chunk" in plan_meta:
            merged["shot_chunk_from_plan"] = _safe_int(plan_meta.get("shot_chunk", 0), 0)

    out_json = os.path.join(args.outdir, "results_merged.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    out_txt = os.path.join(args.outdir, "summary.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(json.dumps(merged, ensure_ascii=False, indent=2))

    print("Merged shards:", len(files))
    print("Wrote:", out_json)
    print("Wrote:", out_txt)

if __name__ == "__main__":
    main()
