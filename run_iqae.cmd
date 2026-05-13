@echo off
setlocal EnableExtensions EnableDelayedExpansion
REM ============================================================
REM QRMSH3DT IQAE run script for NC supplementary calculation
REM Purpose:
REM   Verify that the reversible multi-terminal transport kernel
REM   can be queried by IQAE through terminal-status good-state marking.
REM Recommended formal low-step setting:
REM   steps=1/2/3, good=absorb/boundary/survive, eps=0.01, alpha=0.05, shots=50.
REM Recommended high-step smoke setting:
REM   steps=5, good=absorb, eps=0.10, alpha=0.05, shots=30, bond=256, trunc=1e-12.
REM ============================================================

REM ---- editable user knobs: formal low-step default ----
set STEPS=4
set GOOD=survive
set SHOTS=50
set EPS=0.01
set ALPHA=0.05
set SEED=1234

REM ---- implementation / simulator settings ----
set KERNEL_PROFILE=compressed
set AER_THREADS=2
set AER_MAX_MEM_MB=50000
set MPS_MAX_BOND=0
set MPS_TRUNC=0
set IQAE_SHOT_CHUNK=50
set IQAE_HEARTBEAT_SEC=300

cd /d %~dp0

set "PY=%PY_EXE%"
if "%PY%"=="" set "PY=python"

REM Keep native libraries from oversubscribing threads on Windows.
set RAYON_NUM_THREADS=1
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set NUMEXPR_NUM_THREADS=1
set VECLIB_MAXIMUM_THREADS=1

"%PY%" -c "import sys; print('[INFO] sys.executable =', sys.executable)" || goto :ERR_PY
"%PY%" -c "import qiskit, qiskit_aer; print('[INFO] qiskit =', qiskit.__version__); print('[INFO] qiskit_aer =', qiskit_aer.__version__)" || goto :ERR_QISKIT

set OUTROOT=outputs\iqae_nc_%KERNEL_PROFILE%_s%STEPS%_%GOOD%_eps%EPS%_a%ALPHA%_sh%SHOTS%_b%MPS_MAX_BOND%_tr%MPS_TRUNC%_seed%SEED%

echo ==========================================================
echo [RUN] QRMSH3DT IQAE
echo steps=%STEPS%  good=%GOOD%  shots=%SHOTS%  eps=%EPS%  alpha=%ALPHA%
echo kernel_profile=%KERNEL_PROFILE%
echo aer_threads=%AER_THREADS%  aer_max_mem_mb=%AER_MAX_MEM_MB%
echo mps_max_bond=%MPS_MAX_BOND%  mps_trunc=%MPS_TRUNC%  chunk=%IQAE_SHOT_CHUNK%
echo outdir=%OUTROOT%
echo ==========================================================

"%PY%" main.py ^
  --phase iqae ^
  --steps %STEPS% ^
  --shots %SHOTS% ^
  --seed %SEED% ^
  --kernel-profile %KERNEL_PROFILE% ^
  --aer-safe-basis ^
  --aer-threads %AER_THREADS% ^
  --aer-max-mem-mb %AER_MAX_MEM_MB% ^
  --mps-max-bond %MPS_MAX_BOND% ^
  --mps-trunc %MPS_TRUNC% ^
  --iqae-good %GOOD% ^
  --iqae-eps %EPS% ^
  --iqae-alpha %ALPHA% ^
  --iqae-use-aer ^
  --iqae-shot-chunk %IQAE_SHOT_CHUNK% ^
  --iqae-heartbeat-sec %IQAE_HEARTBEAT_SEC% ^
  --outdir %OUTROOT%

if errorlevel 1 goto :ERR_RUN

echo [DONE] IQAE finished successfully.
pause
exit /b 0

:ERR_RUN
echo [ERROR] IQAE run failed. Check the log above and the output directory.
pause
exit /b 3

:ERR_PY
echo [ERROR] Python is not runnable from this script.
echo         Set PY_EXE or run from Anaconda Prompt / activated QC environment.
pause
exit /b 1

:ERR_QISKIT
echo [ERROR] qiskit / qiskit-aer is not available in this Python.
echo         Activate your QC environment first.
pause
exit /b 2
