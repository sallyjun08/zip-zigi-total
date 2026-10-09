# Policy Bridge v2 — BGE-M3 + V2 Hybrid

이 패키지는 다음 경로에 그대로 배치하도록 구성했습니다.

```text
C:\AI career pass\policy_bridge_v2
```

## 1. 설치

압축을 풀어 `C:\AI career pass\policy_bridge_v2`가 되도록 한 뒤 PowerShell에서 실행합니다.

```powershell
cd "C:\AI career pass\policy_bridge_v2"
Set-ExecutionPolicy -Scope Process Bypass
.\setup_windows.ps1 -Device cpu
```

NVIDIA CUDA 12.8 환경을 사용할 경우:

```powershell
.\setup_windows.ps1 -Device cuda
```

실제 BGE-M3 다운로드와 검색까지 설치 단계에서 확인하려면 `-TestBge`를 추가합니다.

```powershell
.\setup_windows.ps1 -Device cpu -TestBge
```

## 2. 앱 실행

외부 LLM 없이 전체 UI와 로직을 먼저 확인하는 방식입니다.

```powershell
.\run_app.ps1 -Device cpu -LlmProvider mock
```

Ollama의 `gemma3:4b`를 쉬운 설명 생성기로 연결하려면:

```powershell
ollama pull gemma3:4b
.\run_app.ps1 -Device cpu -LlmProvider ollama -LlmModel gemma3:4b
```

BGE-M3 모델 다운로드가 어렵거나 네트워크 없이 UI만 확인하려면:

```powershell
.\run_safe_mode.ps1
```

## 3. 적용된 RAG 설정

- 임베딩: `BAAI/bge-m3`
- 검색: V2 Hybrid
- Dense cosine 가중치: `0.70`
- BM25 가중치: `0.30`
- 기본 backend: `hybrid`
- 초기화 실패 시: `semantic`, 이후 `keyword`로 안전 전환
- 추천된 `policy_id` 및 section 내부에서만 검색

## 4. 폴더 구조

```text
policy_bridge_v2/
├─ app.py
├─ index_final.html   # UI 디자인 시안 (정답 기준)
├─ recommendation_engine.py
├─ rag_retriever.py
├─ easy_explainer.py
├─ service_planner.py
├─ check_ollama.py
├─ verify_project.py
├─ setup_windows.ps1
├─ run_app.ps1
├─ run_safe_mode.ps1
├─ requirements.txt
├─ 공고_모음_수정.xlsx  # 공고별 실제 신청 절차(apply_steps) 원본
├─ data/
│  ├─ policy_chunks.json
│  ├─ housing_supply_notice_master.csv
│  ├─ housing_document_master.csv
│  ├─ housing_notice_document_rules_review_ready.csv
│  └─ housing_rag_chunks_review_ready.csv
└─ vector_store/
```

## 5. 수동 검증

```powershell
.venv\Scripts\python.exe verify_project.py --strict-dependencies
```

실제 BGE-M3 검색 테스트:

```powershell
$env:RAG_DEVICE="cpu"
.venv\Scripts\python.exe test_bge_m3_smoke.py
```

## 6. 데이터 주의

`data` 폴더에는 실제 공고 원문 기반 데이터(`real_csv`)가 들어 있습니다. 결과 카드에는 근거 출처(`실제 공고 근거` 배지)와 검색 상태가 함께 표시됩니다. 기본값 `RAG_REQUIRE_REAL_EVIDENCE=1` 설정에 따라, 실제 근거가 없거나 개발용 데이터로 판정된 서비스는 결과에서 자동으로 제외됩니다(가짜/미확인 정보를 정책 추천으로 노출하지 않기 위함). 이 값을 `0`으로 바꾸면 개발용 샘플도 `개발용 데이터` 배지와 함께 노출됩니다.

`공고_모음_수정.xlsx`(프로젝트 루트, `data` 폴더 밖)는 공고별 실제 신청 절차(`apply_steps`)와 참고용 링크(`link`)를 담고 있으며, xlsx를 읽기 위해 `requirements.txt`에 `openpyxl`이 포함되어 있습니다.

## 7. UI 디자인 시안(`index_final.html`)

`index_final.html`은 Streamlit UI가 맞춰야 할 디자인 기준(정답 시안)입니다. `app.py`에는 이 시안과 픽셀 단위로 맞춘 CSS가 `html_block()`으로 주입되어 있으며, 백엔드 로직(추천 엔진, RAG, `st.session_state` 흐름)은 그대로 두고 UI/스타일만 동기화하는 방식으로 유지보수합니다. 시안과 실제 앱 UI가 달라 보이면 이 파일을 기준으로 비교하세요.

주요 UI 동작:
- **가구 소득 입력**: "가구 월 총 소득(세전)"을 직접 입력하거나, "잘 모르겠어요"를 선택하면 (1) 가족 직업·근로 형태를 자유 서술로 적고 (2) 본인 월 소득을 별도로 입력할 수 있습니다. AI 추정치(서술 텍스트에서 감지된 직업 키워드 기반 금액, 단 "OO만 원"처럼 문장에 구체적 숫자가 있으면 그 숫자를 우선 사용)와 본인 입력 소득을 더한 값이 최종 "가구 월 총 소득"으로 계산되어 추천 점수(기준중위소득 비교)에 실제로 반영됩니다.
- 결과 카드의 필요 서류 버튼은 클릭 시 (1) 발급처 사이트를 우측 팝업으로 열고 (2) 좌측에 체크리스트 드로어를 엽니다. 단, `housing_document_master.csv`의 `portal_url`/`direct_url`이 서류와 무관한 LH청약플러스 메인 페이지(`apply.lh.or.kr`)로만 채워진 경우(발급기관이 서류마다 달라 실제 링크를 특정할 수 없는 서류들)에는 팝업을 열지 않고 버튼 문구도 "필수서류 체크리스트 열기"로 표시됩니다.
- 서류를 모두 발급 완료로 체크하면 축하 모달(컨페티)이 뜨고, "실제 공고 신청하러 가기"를 누르면 **서류 발급 때와 동일한 방식**으로 (1) 실제 신청 사이트가 우측 팝업으로 열리고 (2) 좌측에 신청 절차 체크리스트 드로어가 뜹니다. 체크리스트 항목은 `공고_모음_수정.xlsx`의 `apply_steps` 컬럼(공고별 신청 단계를 번호대로 적어둔 텍스트)에서 가져오며, 아직 해당 공고의 절차가 채워지지 않았으면 안내 문구로 대체됩니다.
- 신청 링크는 `housing_supply_notice_master.csv`의 `application_url`(기관 직접 신청 링크)을 `notice_url`(마이홈 등 포털 공고 링크)보다 우선 사용합니다. `application_url`이 채워진 공고는 포털을 거치지 않고 바로 기관 사이트로 연결됩니다.
- "서류 발급 완료하기"/"신청 완료했어요" 버튼을 누르면(완료로 전환되는 순간) 열려 있던 팝업창이 자동으로 닫힙니다. "신청 완료했어요"를 누르면 축하 모달(컨페티)이 한 번 더 뜹니다.
- 마이홈 등 외부 포털 페이지 안에서 사용자가 다시 "해당기관 공고보기" 같은 링크를 눌러 여는 **2차 팝업 창은 저희 앱이 제어할 수 없습니다**(동일 출처 정책 때문에 저희가 열지 않은 창은 참조 자체를 가질 수 없음). 관련 안내 문구를 체크리스트 드로어와 축하 모달에 표시해 둡니다.
- 쉬운 설명(RAG 근거 기반)은 `LLM_PROVIDER` 환경변수에 따라 `mock`(빠른 확인용) 또는 `ollama`(실제 LLM, 응답 속도 느림)로 생성됩니다. 결과 카드의 "검색 근거 펼치기"를 열면 실제로 어떤 provider/model이 사용됐는지 확인할 수 있습니다. Ollama 호출이 느리면 `LLM_TIMEOUT_SECONDS`(기본 90초)를 늘려주세요.

## 8. 문제 해결

- **`pip install torch`/`sentence-transformers` 실패 (경로가 너무 길다는 오류, `WinError 206`)**: torch 패키지 내부의 라이선스 파일 경로가 매우 깊게 중첩되어 있어서, 프로젝트를 깊은 경로(폴더명이 중복되거나 여러 단계로 중첩된 경로 등)에 두면 Windows의 260자 경로 길이 제한에 걸립니다. 해결 방법은 둘 중 하나입니다.
  1. 관리자 PowerShell에서 Windows 긴 경로 허용을 켭니다: `Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1` (재부팅 또는 재로그인 필요할 수 있음).
  2. 짧은 경로(예: `C:\pbv2venv`)에 별도의 가상환경을 만들어 그쪽 `python.exe`로 `pip install -r requirements.txt` 후 앱을 실행합니다. 프로젝트 폴더 자체는 옮길 필요 없습니다.
- **RAG 검색이 `hybrid`에서 `keyword`로 자동 전환**되며 "sentence-transformers가 필요합니다" 경고가 뜨면, 위 경로 길이 문제로 `torch`/`sentence-transformers` 설치가 실패했을 가능성이 큽니다. 위 방법으로 재설치하세요.
- **Ollama 연결 타임아웃**: 첫 호출은 모델 로딩 때문에 오래 걸릴 수 있습니다(`keep_alive` 기본 10분이라 이후 호출은 빨라짐). `LLM_TIMEOUT_SECONDS`를 180 이상으로 늘려보세요. `http://localhost:11434/api/tags`로 Ollama 서버 응답 여부를 먼저 확인하세요.
