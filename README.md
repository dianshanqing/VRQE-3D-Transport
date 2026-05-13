Archived release DOI: 10.5281/zenodo.20152406
# VRQE: Verifiable Reversible Quantum Embedding for Multi-Terminal Stochastic Transport

This repository contains the public code and reproducibility outputs associated with the Nature Communications submission on a verifiable reversible quantum embedding framework for multi-terminal stochastic transport. The package implements a heterogeneous 3D discrete transport benchmark, a reversible Qiskit circuit construction with unified terminal-status semantics, classical Monte Carlo references, tensor-network circuit sampling checks, compressed-kernel variants, diagnostics, and low-step IQAE demonstrations.

This repository is not only a source-code package; it also includes the output files needed to inspect the numerical results reported in the manuscript.

## Repository scope

The code and outputs support the following reproducibility tasks:

1. Build the reversible 3D transport circuit for a heterogeneous voxel benchmark.
2. Run the classical Monte Carlo reference process.
3. Run Qiskit Aer matrix-product-state (MPS) circuit sampling tests.
4. Compare terminal-event probabilities between MC and circuit sampling with combined standard-error statistics.
5. Inspect baseline and compressed circuit variants under a shared semantic interface.
6. Inspect low-step IQAE runs using terminal-status good-state marking.

## Benchmark model

The default benchmark is a discrete 3D voxel transport model with grid size `16 x 8 x 8`. The source is initialized at `(x, y, z) = (1, 3, 3)`. The model contains a two-layer tungsten slab at `x = 8..9`, with a `2 x 2` air duct window at `y = 3..4, z = 3..4`. The detector region of interest is located at `x = 12, y = 3..4, z = 3..4`. The outer boundary is absorbing.

The terminal-status register uses four mutually exclusive outcomes:

| Terminal event | Meaning |
|---|---|
| `detect` | The particle reaches the detector ROI. |
| `absorb` | The particle is absorbed by material interaction. |
| `boundary` | The particle reaches the absorbing outer boundary. |
| `survive` | The particle remains non-terminal after the specified number of steps. |

## Main files

| File | Purpose |
|---|---|
| `main.py` | Command-line entry point for MC, circuit construction, TEST, IQAE, and diagnostics. |
| `unitary_circuit.py` | Reversible Qiskit circuit construction for the transport kernel. |
| `config.py` | Geometry, material, energy-group, and source configuration. |
| `geometry.py` | Geometry and material lookup helpers shared by MC and circuit code. |
| `mc.py` | Classical Monte Carlo reference process. |
| `iqae.py` | IQAE helper routines and terminal-good-state marking. |
| `shard_runner.py` | Sharded execution driver for long TEST/MC runs. |
| `merge_shards.py` | Merges shard outputs into `merged/results_merged.json` and `merged/summary.txt`. |
| `postprocess_alignment.py` | Computes alignment summaries between MC and circuit-sampling outputs. |
| `compare_mc_test.py` | Lightweight MC-versus-TEST sigma comparison utility. |
| `f1.py`, `f2.py`, `f3.py` | Figure-generation scripts for manuscript schematics. |
| `requirements.txt` | Python package requirements. |
| `run_*.cmd` | Windows command templates used for common runs. |

## Output inventory

The `outputs/` directory contains the result files included with this public release. The most important machine-readable outputs are `results.json`, `merged/results_merged.json`, and `summary.txt` files.

| Directory | Contents |
|---|---|
| `outputs/MC/` | Classical MC references for steps 1--15 with `4e6` histories and seed `1234`. |
| `outputs/test_s1-15_total18000_256_1e-12/` | Baseline finite-bond MPS TEST runs for steps 1--15 with 18,000 total shots, `mps_max_bond=256`, and `mps_trunc=1e-12`. |
| `outputs/total15000_256_1e-12/` | Baseline finite-bond MPS TEST runs with 15,000 total shots. |
| `outputs/test_s1-7_0_0/` | High-fidelity low-step TEST runs with unbounded MPS settings (`mps_max_bond=0`, `mps_trunc=0`). |
| `outputs/test_s12_s15_bond_trunc/` | Step-12 and step-15 MPS bond/truncation scans. |
| `outputs/compressed/` | Compressed-kernel TEST runs, including step-15 bond scans at `256`, `384`, and `512`. |
| `outputs/iqae/` | Low-step IQAE demonstrations using terminal-event good-state marking. |
| `outputs/test6testbond_trunc/` | Additional step-6 MPS parameter checks. |

For compact inspection, start from the `merged/summary.txt` and `merged/results_merged.json` files rather than individual shard files.

## Environment

The code was developed and run with the following key packages:

```text
qiskit==2.2.3
qiskit-aer==0.17.2
qiskit-algorithms==0.4.0
numpy
matplotlib
openpyxl
```

A clean environment can be created with:

```bash
conda create -n vrqe python=3.10 -y
conda activate vrqe
pip install -r requirements.txt
```

The long MPS runs were executed on a Windows workstation/cloud instance. Runtime and memory usage depend strongly on the number of steps and MPS bond settings.

## Quick-start examples

Run a small MC reference:

```bash
python main.py --phase mc --steps 3 --histories 10000 --seed 1234 --outdir outputs/demo_mc_s3
```

Run a small compressed-kernel TEST job:

```bash
python main.py \
  --phase test \
  --steps 3 \
  --shots 1500 \
  --seed 1234 \
  --kernel-profile compressed \
  --test-measure-x \
  --test-measure-coins \
  --test-measure-dir \
  --aer-safe-basis \
  --aer-threads 1 \
  --mps-max-bond 256 \
  --mps-trunc 1e-12 \
  --outdir outputs/demo_test_s3
```

On Windows, the included command templates can be edited and run directly:

```cmd
run_test_shards_baseline.cmd
run_test_shards_compressed.cmd
run_iqae.cmd
```

## Reproducing sharded TEST runs

The sharded runner compiles the circuit, launches shards, and merges shard-level outputs. A typical command is:

```bash
python shard_runner.py \
  --phase test \
  --steps 3 \
  --num-shards 50 \
  --max-parallel 50 \
  --total-shots 15000 \
  --shot-chunk 300 \
  --seed0 1234 \
  --outroot outputs/test_s3_compressed_total15000_seed1234_chunk300_256_1e-12 \
  --extra "--kernel-profile compressed --test-measure-x --test-measure-coins --test-measure-dir --aer-safe-basis --aer-threads 1 --mps-max-bond 256 --mps-trunc 1e-12 --aer-shot-chunk 300 --aer-heartbeat-sec 300" \
  --compile-cache
```

After shard execution, merge with:

```bash
python merge_shards.py \
  --phase test \
  --in outputs/test_s3_compressed_total15000_seed1234_chunk300_256_1e-12/shards \
  --out outputs/test_s3_compressed_total15000_seed1234_chunk300_256_1e-12/merged
```

## MC--TEST alignment check

Use the MC references under `outputs/MC/` and the merged TEST files under `outputs/**/merged/`.

Example:

```bash
python compare_mc_test.py \
  --mc outputs/MC/mc_s3_h4e6_seed1234/results.json \
  --test outputs/compressed/test_s3_compressed_total15000_seed1234_chunk300_256_1e-12/merged/results_merged.json
```

The combined standard-error statistic used by the manuscript is

```text
Z_c = |p_c^Q - p_c^MC| / sqrt((SE_c^MC)^2 + (SE_c^Q)^2)
```

where `c` is one of `detect`, `absorb`, `boundary`, or `survive`.

## IQAE examples

The IQAE routine uses the terminal-status register to mark a selected terminal event as the good state. A low-step example is:

```bash
python main.py \
  --phase iqae \
  --steps 3 \
  --shots 50 \
  --seed 1234 \
  --kernel-profile compressed \
  --aer-safe-basis \
  --aer-threads 2 \
  --mps-max-bond 0 \
  --mps-trunc 0 \
  --iqae-good absorb \
  --iqae-eps 0.01 \
  --iqae-alpha 0.05 \
  --iqae-use-aer \
  --outdir outputs/demo_iqae_s3_absorb
```

The IQAE examples are intended to demonstrate amplitude-estimation compatibility of the terminal-status interface. They are not intended as high-step performance benchmarks.

## Notes on reproducibility

- The MC and TEST procedures are stochastic; exact counts may differ if seeds, sharding, Qiskit/Aer versions, or MPS parameters are changed.
- MPS bond and truncation settings affect circuit-sampling fidelity and runtime. The unbounded setting `mps_max_bond=0, mps_trunc=0` is substantially more expensive than finite-bond settings.
- The compiled `.qpy` files, if retained, are cache artifacts for convenience. They are not required to rebuild the circuits from source.
- For manuscript-level inspection, use the archived `outputs/` files rather than rerunning the longest jobs.

## Citation

If you use this repository, please cite the associated manuscript and the archived software release:

Wang, Y. *VRQE-3D-Transport: Code and reproducibility outputs for a verifiable reversible quantum embedding framework for multi-terminal stochastic transport*. Zenodo. DOI: 10.5281/zenodo.20152406 (2026).

## License

This repository is released under the MIT License.
