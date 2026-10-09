param(
    [ValidateSet("cpu", "cuda")]
    [string]$Device = "cpu",

    [ValidateSet("mock", "ollama", "disabled")]
    [string]$LlmProvider = "mock",

    [string]$LlmModel = "gemma3:4b",

    [string]$PythonPath = "",

    [switch]$SkipVerify
)

$ErrorActionPreference = "Stop"
$ProjectDir = $PSScriptRoot
$DataDir = Join-Path $ProjectDir "data"

if (-not $PythonPath) {
    $Candidates = @(
        (Join-Path $ProjectDir ".venv\Scripts\python.exe"),
        "C:\AI career pass\policy_bridge_baseline\.venv\Scripts\python.exe"
    )
    $PythonPath = $Candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
}
if (-not $PythonPath) {
    $PythonPath = (Get-Command python -ErrorAction Stop).Source
}
if (-not (Test-Path $PythonPath)) {
    throw "Python 실행 파일을 찾을 수 없습니다: $PythonPath"
}

Set-Location $ProjectDir
if (-not $SkipVerify) {
    & $PythonPath verify_real_data.py --data-dir $DataDir
    if ($LASTEXITCODE -ne 0) {
        throw "실제 데이터 검증 실패로 앱 실행을 중단했습니다."
    }
}

$env:PYTHONUTF8 = "1"
$env:RAG_DATA_MODE = "real"
$env:RAG_REQUIRE_REAL_EVIDENCE = "1"
$env:RAG_BACKEND = "hybrid"
$env:RAG_EMBEDDING_MODEL = "BAAI/bge-m3"
$env:RAG_HYBRID_DENSE_WEIGHT = "0.70"
$env:RAG_HYBRID_KEYWORD_WEIGHT = "0.30"
$env:RAG_DEVICE = $Device
$env:RAG_BATCH_SIZE = if ($Device -eq "cuda") { "16" } else { "8" }
$env:RAG_NOTICE_CANDIDATES = "3"
$env:LLM_PROVIDER = $LlmProvider
$env:LLM_MODEL = $LlmModel
$env:LLM_BASE_URL = "http://localhost:11434"
$env:LLM_TIMEOUT_SECONDS = "90"
$env:LLM_MAX_POLICIES = "3"
$env:LLM_KEEP_ALIVE = "10m"

Write-Host "RAG Data: real housing CSV" -ForegroundColor Green
Write-Host "RAG: BAAI/bge-m3 + V2 Hybrid (Dense 0.70 / BM25 0.30)" -ForegroundColor Cyan
Write-Host "Python: $PythonPath" -ForegroundColor Cyan
Write-Host "Device: $Device | LLM: $LlmProvider / $LlmModel" -ForegroundColor Cyan
& $PythonPath -m streamlit run app.py
