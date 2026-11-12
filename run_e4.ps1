param(
    [ValidateSet('preflight','smoke','train','score','benchmark','explain','figures','verify','report')]
    [string]$Stage = 'verify',
    [string]$RuntimeRoot = 'E:\EE6405_E4'
)
$ErrorActionPreference = 'Stop'
$pythonExe = Join-Path $RuntimeRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Python environment not found: $pythonExe. See experiments/e4/README.md."
}
$env:E4_CACHE = Join-Path $RuntimeRoot 'cache'
$env:TEMP = Join-Path $RuntimeRoot 'tmp'
$env:TMP = $env:TEMP
$env:PYTHONUNBUFFERED = '1'
New-Item -ItemType Directory -Path $env:TEMP,$env:E4_CACHE -Force | Out-Null
Push-Location $PSScriptRoot
try {
    switch ($Stage) {
        'preflight' { & $pythonExe -m experiments.e4.verify --preflight }
        'smoke' { & $pythonExe -m experiments.e4.train smoke --micro-batch 16 --precision fp32 --steps 50 }
        'train' { & $pythonExe -m experiments.e4.train train }
        'score' { & $pythonExe -m experiments.e4.evaluate score }
        'benchmark' { & $pythonExe -m experiments.e4.evaluate benchmark }
        'explain' { & $pythonExe -m experiments.e4.evaluate explain }
        'figures' { & $pythonExe -m experiments.e4.figures }
        'verify' { & $pythonExe -m experiments.e4.verify }
        'report' { & $pythonExe -m experiments.e4.report }
    }
    if ($LASTEXITCODE -ne 0) { throw "E4 stage '$Stage' failed with exit code $LASTEXITCODE" }
} finally {
    Pop-Location
}
