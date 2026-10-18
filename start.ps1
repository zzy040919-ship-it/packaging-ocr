$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$workspacePython = Join-Path $PSScriptRoot '..\..\work\.venv\Scripts\python.exe'
if ((-not (Test-Path -LiteralPath $projectPython)) -and (Test-Path -LiteralPath $workspacePython)) {
    $projectPython = (Resolve-Path -LiteralPath $workspacePython).Path
}
if (-not (Test-Path -LiteralPath $projectPython)) {
    py -3.11 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Please install Python 3.11 (64-bit), then retry.' }
}
& $projectPython -c "import importlib.util,sys; sys.exit(not all(importlib.util.find_spec(m) for m in ('paddle','paddleocr','cv2','PIL')))"
if ($LASTEXITCODE -ne 0) {
    & $projectPython -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. Retry: .venv\Scripts\python.exe -m pip install -r requirements.txt' }
}
& $projectPython gui.py
if ($LASTEXITCODE -ne 0) { throw 'Launch failed; see the error above and README.md.' }
