param(
    [Parameter(Mandatory = $true)]
    [string]$SourceDir,

    [string]$ProjectDir = $PSScriptRoot,

    [string]$PythonPath = ""
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $SourceDir -PathType Container)) {
    throw "Source folder does not exist: $SourceDir"
}
if (-not (Test-Path $ProjectDir -PathType Container)) {
    throw "Project folder does not exist: $ProjectDir"
}

$SourceDir = (Resolve-Path $SourceDir).Path
$ProjectDir = (Resolve-Path $ProjectDir).Path
$DataDir = Join-Path $ProjectDir "data"
$VectorDir = Join-Path $ProjectDir "vector_store"

if (-not $PythonPath) {
    $Candidates = @(
        (Join-Path $ProjectDir ".venv\Scripts\python.exe"),
        "C:\AI career pass\policy_bridge_baseline\.venv\Scripts\python.exe"
    )
    $PythonPath = $Candidates | Where-Object { Test-Path $_ -PathType Leaf } | Select-Object -First 1
}
if (-not $PythonPath) {
    $PythonPath = (Get-Command python -ErrorAction Stop).Source
}
if (-not (Test-Path $PythonPath -PathType Leaf)) {
    throw "Python executable not found: $PythonPath"
}

$RequiredFiles = @(
    "housing_supply_notice_master.csv",
    "housing_rag_chunks_review_ready.csv",
    "housing_document_master.csv",
    "housing_notice_document_rules_review_ready.csv"
)
$RawFiles = @(
    "공고_모음.csv",
    "소득.csv",
    "신청자격_판정_로직.csv",
    "용어_사전.csv",
    "행정_서류_발급.csv"
)
$OptionalFiles = @(
    "housing_terms.csv",
    "raw_data_conversion_manifest.json",
    "housing_rag_vector_ready.csv",
    "housing_rag_package_mvp_summary.json",
    "HOUSING_RAG_PACKAGE_README.md",
    "policy_notice_map.json"
)

function Test-AllFiles([string]$Directory, [string[]]$Files) {
    foreach ($File in $Files) {
        if (-not (Test-Path (Join-Path $Directory $File) -PathType Leaf)) {
            return $false
        }
    }
    return $true
}

$PreparedDir = $SourceDir
if (-not (Test-AllFiles $PreparedDir $RequiredFiles)) {
    $GeneratedDir = Join-Path $SourceDir "generated"

    if (Test-AllFiles $GeneratedDir $RequiredFiles) {
        $PreparedDir = $GeneratedDir
        Write-Host "[INFO] Existing generated standard data will be used: $PreparedDir" -ForegroundColor Cyan
    }
    elseif (Test-AllFiles $SourceDir $RawFiles) {
        $Converter = Join-Path $ProjectDir "prepare_real_data.py"
        if (-not (Test-Path $Converter -PathType Leaf)) {
            throw "Converter not found: $Converter"
        }
        Write-Host "[INFO] Raw 5 CSV files detected. Originals will remain unchanged." -ForegroundColor Cyan
        Write-Host "[INFO] Generating standard files in: $GeneratedDir" -ForegroundColor Cyan
        & $PythonPath $Converter --source-dir $SourceDir --output-dir $GeneratedDir
        if ($LASTEXITCODE -ne 0) {
            throw "Raw data conversion failed."
        }
        if (-not (Test-AllFiles $GeneratedDir $RequiredFiles)) {
            throw "Conversion completed but required standard files were not generated: $GeneratedDir"
        }
        $PreparedDir = $GeneratedDir
    }
    else {
        $MissingRaw = @()
        foreach ($File in $RawFiles) {
            if (-not (Test-Path (Join-Path $SourceDir $File) -PathType Leaf)) {
                $MissingRaw += $File
            }
        }
        $MissingStandard = @()
        foreach ($File in $RequiredFiles) {
            if (-not (Test-Path (Join-Path $SourceDir $File) -PathType Leaf)) {
                $MissingStandard += $File
            }
        }
        throw (
            "Neither standard 4 files nor raw 5 files are complete.`n" +
            "Missing raw files: " + ($MissingRaw -join ", ") + "`n" +
            "Missing standard files: " + ($MissingStandard -join ", ")
        )
    }
}

New-Item -ItemType Directory -Path $DataDir -Force | Out-Null
New-Item -ItemType Directory -Path $VectorDir -Force | Out-Null

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$BackupDir = Join-Path $DataDir "backup_before_real_$Timestamp"
New-Item -ItemType Directory -Path $BackupDir -Force | Out-Null

foreach ($File in ($RequiredFiles + $OptionalFiles + @("policy_chunks.json"))) {
    $Existing = Join-Path $DataDir $File
    if (Test-Path $Existing -PathType Leaf) {
        Copy-Item $Existing (Join-Path $BackupDir $File) -Force
    }
}

foreach ($File in $RequiredFiles) {
    Copy-Item (Join-Path $PreparedDir $File) (Join-Path $DataDir $File) -Force
    Write-Host "[COPY] $File" -ForegroundColor Cyan
}
foreach ($File in $OptionalFiles) {
    $Source = Join-Path $PreparedDir $File
    if (Test-Path $Source -PathType Leaf) {
        Copy-Item $Source (Join-Path $DataDir $File) -Force
        Write-Host "[COPY] $File" -ForegroundColor DarkCyan
    }
}

Get-ChildItem $VectorDir -File -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match "embeddings|manifest" } |
    Remove-Item -Force
Write-Host "[CLEAN] Existing embedding cache removed" -ForegroundColor Yellow

Set-Location $ProjectDir
Write-Host "[INFO] Python: $PythonPath" -ForegroundColor Cyan
Write-Host "[INFO] Prepared data: $PreparedDir" -ForegroundColor Cyan
& $PythonPath verify_real_data.py --data-dir $DataDir
if ($LASTEXITCODE -ne 0) {
    throw "Real data validation failed. Backup: $BackupDir"
}

Write-Host ""
Write-Host "Real data installation and validation completed." -ForegroundColor Green
Write-Host "Original raw data: $SourceDir" -ForegroundColor Green
Write-Host "Generated data:    $PreparedDir" -ForegroundColor Green
Write-Host "Project data:      $DataDir" -ForegroundColor Green
Write-Host "Backup:            $BackupDir" -ForegroundColor DarkGreen
