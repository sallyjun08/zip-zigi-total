param(
    [Parameter(Mandatory = $true)]
    [string]$SourceDir,

    [string]$ProjectDir = $PSScriptRoot,

    [string]$PythonPath = ""
)

$ErrorActionPreference = "Stop"
$SourceDir = (Resolve-Path $SourceDir).Path
$ProjectDir = (Resolve-Path $ProjectDir).Path
$DataDir = Join-Path $ProjectDir "data"
$VectorDir = Join-Path $ProjectDir "vector_store"

$RequiredFiles = @(
    "housing_supply_notice_master.csv",
    "housing_rag_chunks_review_ready.csv",
    "housing_document_master.csv",
    "housing_notice_document_rules_review_ready.csv"
)
$OptionalFiles = @(
    "housing_rag_vector_ready.csv",
    "housing_rag_package_mvp_summary.json",
    "HOUSING_RAG_PACKAGE_README.md",
    "policy_notice_map.json"
)

foreach ($File in $RequiredFiles) {
    $Path = Join-Path $SourceDir $File
    if (-not (Test-Path $Path)) {
        throw "필수 실제 데이터 파일이 없습니다: $Path"
    }
}

New-Item -ItemType Directory -Path $DataDir -Force | Out-Null
New-Item -ItemType Directory -Path $VectorDir -Force | Out-Null

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$BackupDir = Join-Path $DataDir "backup_before_real_$Timestamp"
New-Item -ItemType Directory -Path $BackupDir -Force | Out-Null

foreach ($File in ($RequiredFiles + $OptionalFiles + @("policy_chunks.json"))) {
    $Existing = Join-Path $DataDir $File
    if (Test-Path $Existing) {
        Copy-Item $Existing (Join-Path $BackupDir $File) -Force
    }
}

foreach ($File in $RequiredFiles) {
    Copy-Item (Join-Path $SourceDir $File) (Join-Path $DataDir $File) -Force
    Write-Host "[COPY] $File" -ForegroundColor Cyan
}
foreach ($File in $OptionalFiles) {
    $Source = Join-Path $SourceDir $File
    if (Test-Path $Source) {
        Copy-Item $Source (Join-Path $DataDir $File) -Force
        Write-Host "[COPY] $File" -ForegroundColor DarkCyan
    }
}

# 데이터가 바뀌면 기존 임베딩 캐시는 사용할 수 없으므로 제거한다.
Get-ChildItem $VectorDir -File -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match "embeddings|manifest" } |
    Remove-Item -Force
Write-Host "[CLEAN] 기존 임베딩 캐시 제거" -ForegroundColor Yellow

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
Write-Host "[INFO] Python: $PythonPath" -ForegroundColor Cyan
& $PythonPath verify_real_data.py --data-dir $DataDir
if ($LASTEXITCODE -ne 0) {
    throw "실제 데이터 검증에 실패했습니다. 위 [FAIL] 항목을 수정하세요. 백업: $BackupDir"
}

Write-Host "" 
Write-Host "실제 데이터 설치와 검증이 완료되었습니다." -ForegroundColor Green
Write-Host "Project: $ProjectDir" -ForegroundColor Green
Write-Host "Data:    $DataDir" -ForegroundColor Green
Write-Host "Backup:  $BackupDir" -ForegroundColor DarkGreen
