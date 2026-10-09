# Policy Bridge — BGE-M3 + V2 Hybrid 적용본

## 적용 내용

- 기본 임베딩 모델: `BAAI/bge-m3`
- 기본 검색 방식: `hybrid`
- V2 Hybrid 구성: BGE-M3 dense cosine similarity + BM25 sparse score
- 기본 결합 가중치: Dense 0.70 + BM25 0.30
- 검색 범위 안전장치 유지: 추천 엔진이 선택한 `policy_id` 및 section 내부에서만 검색
- 장애 시 fallback: `hybrid → semantic → keyword`
- 기존 LLM 쉬운 설명 및 신청 준비 계획 인터페이스 유지

## 교체할 파일

프로젝트 폴더 `C:\AI career pass\policy_bridge_baseline`에서 다음 파일을 교체합니다.

```text
app.py
rag_retriever.py
requirements.txt
```

테스트용으로 다음 파일을 추가합니다.

```text
test_hybrid_backend.py
```

## 설치

```powershell
cd "C:\AI career pass\policy_bridge_baseline"
.venv\Scripts\activate
pip install -r requirements.txt
```

BGE-M3는 최초 실행 시 Hugging Face에서 모델을 다운로드합니다. 이후 임베딩은 `vector_store`에 캐시됩니다.

## 실행

```powershell
$env:RAG_BACKEND="hybrid"
$env:RAG_EMBEDDING_MODEL="BAAI/bge-m3"
streamlit run app.py
```

기본값이 이미 위와 같으므로 환경변수를 생략해도 됩니다.

## 선택 설정

```powershell
$env:RAG_HYBRID_DENSE_WEIGHT="0.70"
$env:RAG_HYBRID_KEYWORD_WEIGHT="0.30"
$env:RAG_DEVICE="cuda"
$env:RAG_BATCH_SIZE="16"
```

GPU 메모리 문제가 있으면 `RAG_BATCH_SIZE`를 8 또는 4로 낮춥니다. CPU로 실행하려면 `RAG_DEVICE`를 설정하지 않습니다.

## 테스트

외부 모델 다운로드 없이 결합 로직만 검사:

```powershell
python test_hybrid_backend.py
```

기존 구조 테스트:

```powershell
python test_retriever.py
python test_integration.py
```

## 캐시 갱신

이전 임베딩 캐시가 있어도 모델명과 데이터 fingerprint가 달라지면 자동 재생성됩니다. 완전 초기화를 원하면 앱 종료 후 다음 폴더를 삭제합니다.

```powershell
Remove-Item -Recurse -Force .\vector_store
```
