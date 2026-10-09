$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectDir ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw ".venv가 없습니다. 먼저 .\setup_windows.ps1을 실행하세요."
}
Set-Location $ProjectDir
$env:PYTHONUTF8 = "1"
$env:RAG_BACKEND = "keyword"
$env:LLM_PROVIDER = "mock"
Write-Host "안전 모드: 외부 모델 다운로드와 Ollama 없이 실행합니다." -ForegroundColor Yellow
& $Python -m streamlit run app.py
