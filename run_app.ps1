param(
    [ValidateSet("cpu", "cuda")]
    [string]$Device = "cpu",
    [ValidateSet("mock", "ollama", "disabled")]
    [string]$LlmProvider = "mock",
    [string]$LlmModel = "gemma3:4b"
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectDir ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw ".venv가 없습니다. 먼저 .\setup_windows.ps1 -Device $Device 를 실행하세요."
}

Set-Location $ProjectDir
$env:PYTHONUTF8 = "1"
$env:RAG_BACKEND = "hybrid"
$env:RAG_EMBEDDING_MODEL = "BAAI/bge-m3"
$env:RAG_HYBRID_DENSE_WEIGHT = "0.70"
$env:RAG_HYBRID_KEYWORD_WEIGHT = "0.30"
$env:RAG_DEVICE = $Device
$env:RAG_BATCH_SIZE = if ($Device -eq "cuda") { "16" } else { "8" }
$env:LLM_PROVIDER = $LlmProvider
$env:LLM_MODEL = $LlmModel
$env:LLM_BASE_URL = "http://localhost:11434"
$env:LLM_TIMEOUT_SECONDS = "90"
$env:LLM_MAX_POLICIES = "3"
$env:LLM_KEEP_ALIVE = "10m"

Write-Host "RAG: BAAI/bge-m3 + V2 Hybrid (Dense 0.70 / BM25 0.30)" -ForegroundColor Cyan
Write-Host "Device: $Device | LLM: $LlmProvider / $LlmModel" -ForegroundColor Cyan
& $Python -m streamlit run app.py
