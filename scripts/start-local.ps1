$ErrorActionPreference = 'Stop'
$projectDir = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectDir
$runtimePath = Join-Path $projectDir '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $runtimePath)) { throw 'Créer .venv selon README.md' }
New-Item -ItemType Directory -Force -Path 'monitoring' | Out-Null
$apiProcess = Start-Process -FilePath $runtimePath -ArgumentList '-m','uvicorn','churn.api:app','--host','127.0.0.1','--port','8000' -WorkingDirectory $projectDir -WindowStyle Hidden -PassThru -RedirectStandardOutput 'monitoring/api.stdout.log' -RedirectStandardError 'monitoring/api.stderr.log'
$uiProcess = Start-Process -FilePath $runtimePath -ArgumentList '-m','streamlit','run','app.py','--server.headless=true','--browser.gatherUsageStats=false' -WorkingDirectory $projectDir -WindowStyle Hidden -PassThru -RedirectStandardOutput 'monitoring/ui.stdout.log' -RedirectStandardError 'monitoring/ui.stderr.log'
@{ api = $apiProcess.Id; ui = $uiProcess.Id } | ConvertTo-Json | Set-Content 'monitoring/processes.json'
Write-Output 'API: http://127.0.0.1:8000/docs ; interface: http://127.0.0.1:8501'
