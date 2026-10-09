# 데이터 안내

이 폴더의 데이터는 앱과 테스트를 즉시 실행하기 위한 **개발용 샘플 데이터**입니다.

- `policy_chunks.json`: 6개 정책 × 5개 section, 총 30개 RAG chunk
- `housing_supply_notice_master.csv`: 신청 준비 계획 연결 테스트용 샘플 공고
- `housing_document_master.csv`: 서류 발급 가이드 샘플
- `housing_notice_document_rules_review_ready.csv`: 공고-서류 조건 규칙 샘플
- `housing_rag_chunks_review_ready.csv`: 신청·제출 안내 샘플

모든 정책 chunk는 `is_mock=true`이며, 공고 제목에도 `[개발용 샘플]`이 표시됩니다. 실제 서비스 배포 전 공식 데이터로 교체해야 합니다.
