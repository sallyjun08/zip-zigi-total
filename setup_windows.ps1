param(
    [ValidateSet("cpu", "cuda")]
    [string]$Device = "cpu",
    [switch]$TestBge
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectDir

Write-Host "[1/6] 프로젝트 위치: $ProjectDir" -ForegroundColor Cyan

if (Get-Command py -ErrorAction SilentlyContinue) {
    $PythonExe = "py"
    $PythonArgs = @("-3")
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $PythonExe = "python"
    $PythonArgs = @()
} else {
    throw "Python을 찾을 수 없습니다. Python 3.11~3.13을 설치한 뒤 다시 실행하세요."
}

& $PythonExe @PythonArgs -c "import sys; assert sys.version_info >= (3,11); print(sys.version)"
if ($LASTEXITCODE -ne 0) {
    throw "Python 3.11 이상이 필요합니다."
}

Write-Host "[2/6] 가상환경 생성" -ForegroundColor Cyan
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    & $PythonExe @PythonArgs -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "가상환경 생성 실패" }
}
$VenvPython = Join-Path $ProjectDir ".venv\Scripts\python.exe"

Write-Host "[3/6] pip 및 PyTorch 설치 ($Device)" -ForegroundColor Cyan
& $VenvPython -m pip install --upgrade pip setuptools wheel
if ($Device -eq "cuda") {
    & $VenvPython -m pip install torch --index-url https://download.pytorch.org/whl/cu128
} else {
    & $VenvPython -m pip install torch --index-url https://download.pytorch.org/whl/cpu
}

Write-Host "[4/6] 프로젝트 패키지 설치" -ForegroundColor Cyan
& $VenvPython -m pip install -r requirements.txt

Write-Host "[5/6] 전체 검증" -ForegroundColor Cyan
& $VenvPython verify_project.py --strict-dependencies

if ($TestBge) {
    Write-Host "[6/6] 실제 BGE-M3 모델 테스트" -ForegroundColor Cyan
    $env:RAG_DEVICE = $Device
    $env:RAG_BACKEND = "hybrid"
    $env:RAG_EMBEDDING_MODEL = "BAAI/bge-m3"
    & $VenvPython test_bge_m3_smoke.py
} else {
    Write-Host "[6/6] BGE-M3 실제 모델 테스트 생략" -ForegroundColor Yellow
    Write-Host "모델 다운로드까지 확인하려면: .\setup_windows.ps1 -Device $Device -TestBge"
}

Write-Host ""
Write-Host "설치 완료!" -ForegroundColor Green
Write-Host "앱 실행: .\run_app.ps1 -Device $Device -LlmProvider mock"
Write-Host "Ollama 연결: .\run_app.ps1 -Device $Device -LlmProvider ollama"
