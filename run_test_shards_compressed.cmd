@echo off
setlocal EnableDelayedExpansion

set STEPS=5
set N_SHARDS=50
set MAX_PARALLEL=50
set TOTAL_SHOTS=15000
set SHOT_CHUNK=300
set SEED0=1234

set AER_THREADS=1
set MPS_MAX_BOND=256
set MPS_TRUNC=1e-12

set OUTROOT=outputs\test_s%STEPS%_compressed_total%TOTAL_SHOTS%_seed%SEED0%_chunk%SHOT_CHUNK%_%MPS_MAX_BOND%_%MPS_TRUNC%

set RAYON_NUM_THREADS=1
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set NUMEXPR_NUM_THREADS=1

set EXTRA=--kernel-profile compressed --test-measure-x --test-measure-coins --test-measure-dir --aer-safe-basis --aer-threads %AER_THREADS% --mps-max-bond %MPS_MAX_BOND% --mps-trunc %MPS_TRUNC% --aer-shot-chunk %SHOT_CHUNK% --aer-heartbeat-sec 300

echo ==========================================================
echo [RUN] s103 TEST shards (compressed)
echo steps=%STEPS% shards=%N_SHARDS% total_shots=%TOTAL_SHOTS%
echo outroot=%OUTROOT%
echo EXTRA=%EXTRA%
echo ==========================================================

python shard_runner.py --phase test --steps %STEPS% --num-shards %N_SHARDS% --max-parallel %MAX_PARALLEL% --total-shots %TOTAL_SHOTS% --shot-chunk %SHOT_CHUNK% --seed0 %SEED0% --outroot %OUTROOT% --extra "%EXTRA%" --compile-cache
set RUNNER_RC=%ERRORLEVEL%
if NOT "%RUNNER_RC%"=="0" (
  echo [WARN] shard_runner exit code=%RUNNER_RC%
)

dir /b "%OUTROOT%\shards\shard_*\results.json" >nul 2>&1
if errorlevel 1 (
  echo [WARN] No results.json found; skip merge.
  pause
  exit /b %RUNNER_RC%
)

python merge_shards.py --phase test --in "%OUTROOT%\shards" --out "%OUTROOT%\merged"

echo.
echo Done. Merged summary:
echo   %OUTROOT%\merged\summary.txt
pause
exit /b %RUNNER_RC%
