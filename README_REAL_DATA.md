# Policy Bridge v2 실제 공고 RAG 적용

## 1. 실제 데이터 위치

아래 4개 파일을 프로젝트의 `data` 폴더에 둡니다.

```text
C:\AI career pass\policy_bridge_v2_installer\policy_bridge_v2\
├─ app.py
├─ rag_retriever.py
├─ verify_real_data.py
├─ install_real_data.ps1
├─ run_app_real.ps1
└─ data\
   ├─ housing_supply_notice_master.csv
   ├─ housing_rag_chunks_review_ready.csv
   ├─ housing_document_master.csv
   └─ housing_notice_document_rules_review_ready.csv
```

선택 파일:

```text
housing_rag_vector_ready.csv
housing_rag_package_mvp_summary.json
HOUSING_RAG_PACKAGE_README.md
policy_notice_map.json
```

`housing_rag_chunks_review_ready.csv`의 `retrieval_text`를 BGE-M3로 임베딩합니다. 검색 범위는 먼저 실제 공고 마스터에서 사용자 지역·나이·무주택 조건과 주택유형을 이용해 후보 `notice_id`를 고른 뒤, 선택된 `notice_id` 내부로 제한합니다.

## 2. 자동 복사·백업·검증

실제 데이터 4종이 들어 있는 폴더를 `C:\AI career pass\housing_real_data`라고 가정합니다.

```powershell
cd "C:\AI career pass\policy_bridge_v2_installer\policy_bridge_v2"

Set-ExecutionPolicy -Scope Process Bypass -Force

.\install_real_data.ps1 `
  -SourceDir "C:\AI career pass\housing_real_data" `
  -PythonPath "C:\AI career pass\policy_bridge_baseline\.venv\Scripts\python.exe"
```

이 스크립트는 기존 `data` 파일을 `data\backup_before_real_날짜시간`에 백업하고, 실제 CSV를 복사하며, 오래된 임베딩 캐시를 제거하고, 실제 데이터 검증까지 수행합니다.

## 3. 수동 복사

```powershell
$Project = "C:\AI career pass\policy_bridge_v2_installer\policy_bridge_v2"
$Source = "C:\AI career pass\housing_real_data"

Copy-Item "$Source\housing_supply_notice_master.csv" "$Project\data\" -Force
Copy-Item "$Source\housing_rag_chunks_review_ready.csv" "$Project\data\" -Force
Copy-Item "$Source\housing_document_master.csv" "$Project\data\" -Force
Copy-Item "$Source\housing_notice_document_rules_review_ready.csv" "$Project\data\" -Force

Remove-Item "$Project\vector_store\*embeddings*" -Force -ErrorAction SilentlyContinue
Remove-Item "$Project\vector_store\*manifest*" -Force -ErrorAction SilentlyContinue
```

## 4. 검증

```powershell
$PYTHON = "C:\AI career pass\policy_bridge_baseline\.venv\Scripts\python.exe"
cd "C:\AI career pass\policy_bridge_v2_installer\policy_bridge_v2"

& $PYTHON verify_real_data.py --data-dir data
& $PYTHON test_real_data_retriever.py
& $PYTHON test_retriever.py
& $PYTHON test_integration.py
```

검증기는 다음을 확인합니다.

- 4개 필수 CSV와 필수 컬럼
- `notice_id`와 `document_id` 조인 무결성
- 중복 `chunk_id`, 빈 `retrieval_text`
- 개발용·샘플·mock 표식
- 실제 데이터 규모
- `chunk_type` 분포
- 정책 계열 → 실제 공고 → notice 범위 RAG 검색
- 원문 파일·페이지 추적

## 5. 실행

Ollama 없이 먼저 확인:

```powershell
.\run_app_real.ps1 `
  -Device cpu `
  -LlmProvider mock `
  -PythonPath "C:\AI career pass\policy_bridge_baseline\.venv\Scripts\python.exe"
```

Ollama 사용:

```powershell
.\run_app_real.ps1 `
  -Device cpu `
  -LlmProvider ollama `
  -LlmModel gemma3:4b `
  -PythonPath "C:\AI career pass\policy_bridge_baseline\.venv\Scripts\python.exe"
```

첫 Hybrid 실행에서는 실제 청크 전체의 BGE-M3 임베딩 캐시가 생성됩니다. CPU에서는 수 분이 걸릴 수 있습니다.

## 6. 실제 출력 동작

```text
6+1 사용자 입력
→ 일반 서비스 계열 추천
→ 실제 공고 마스터에서 관련 공고 후보 선정
→ 상위 실제 notice_id 3개로 검색 범위 제한
→ BGE-M3 Dense + BM25 Hybrid 검색
→ eligibility / income / benefit / application / caution 각 Top-1
→ 실제 공고명·notice_id·원문 파일·페이지·공식 URL을 LLM에 전달
→ 실제 공고 카드와 신청 준비 계획 출력
```

기본값에서는 실제 근거가 없거나 개발용 데이터로 판정된 서비스 카드는 표시하지 않습니다.

## 7. 중요한 범위 제한

현재 실제 DB는 주택공급 공고 중심입니다. 따라서 `청년월세 특별지원`이나 `청년 우대형 청약저축`처럼 주택공급 공고 DB에 직접 대응하는 자료가 없는 서비스는 억지로 다른 공고와 연결하지 않고 결과에서 제외합니다. 실제 자료가 별도로 있다면 `data/policy_notice_map.json`과 해당 청크 데이터를 추가하여 확장할 수 있습니다.
