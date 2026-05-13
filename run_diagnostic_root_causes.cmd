@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d %~dp0

set OUTROOT=outputs\diag_root_causes
if not exist "%OUTROOT%" mkdir "%OUTROOT%"

python diagnostic_only.py ^
  --materials concrete,tungsten,air ^
  --groups 0,1 ^
  --shots 200000 ^
  --seed 1234 ^
  --aer-safe-basis ^
  --aer-threads 1 ^
  --mps-max-bond 256 ^
  --mps-trunc 1e-12 ^
  --outdir "%OUTROOT%"

endlocal
