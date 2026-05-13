@echo off
setlocal

set SHOTS=200000
set SEED=1234
set BOND=256
set OUTDIR=outputs\diag_fullstep_a1a2_bond%BOND%

python diagnostic_fullstep_a1a2.py --which A1,A2 --shots %SHOTS% --seed %SEED% --aer-safe-basis --aer-threads 1 --mps-max-bond %BOND% --mps-trunc 1e-12 --outdir %OUTDIR%


endlocal
