param(
    [Parameter(Mandatory=$true)][string]$DataRoot,
    [string]$EnvironmentName = "data_process"
)
$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $env:USERPROFILE ".conda/envs/$EnvironmentName/python.exe"
if (-not (Test-Path -LiteralPath $Python)) { throw "Python not found: $Python" }
Push-Location $ProjectRoot
try {
    & $Python main.py --algorithm RC-OPLNet --root $DataRoot --dg_mode DG `
        --split-profile strict --source-domains APTOS DEEPDR FGADR IDRID `
        --target-domains RLDR --epochs 2 --batch-size 16 --num-workers 0 `
        --val_ep 1 --max-train-batches 2 --max-eval-batches 2 --output smoke_rc_oplnet
    if ($LASTEXITCODE -ne 0) { throw "RC-OPLNet smoke run failed." }
} finally { Pop-Location }
