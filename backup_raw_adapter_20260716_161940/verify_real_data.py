# -*- coding: utf-8 -*-
"""Policy Bridge 실제 공고 CSV 설치 상태 검증."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

from rag_retriever import PolicyRetriever, RetrieverConfigurationError


REQUIRED_FILES = {
    "housing_supply_notice_master.csv": {
        "notice_id", "announcement_title", "housing_type",
    },
    "housing_rag_chunks_review_ready.csv": {
        "chunk_id", "notice_id", "retrieval_text",
    },
    "housing_document_master.csv": {
        "document_id", "canonical_name",
    },
    "housing_notice_document_rules_review_ready.csv": {
        "notice_id", "document_id",
    },
}

MOCK_MARKERS = ("개발용", "샘플", "temporary", "synthetic", "mock")


def read_csv(path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise RuntimeError(f"인코딩을 읽을 수 없습니다: {path}: {last_error}")


def contains_mock_marker(frame: pd.DataFrame) -> list[str]:
    columns = [
        column for column in (
            "announcement_title", "source_file", "notice_status", "data_quality",
            "notes", "retrieval_text", "is_mock", "is_temporary",
        ) if column in frame.columns
    ]
    found: list[str] = []
    for column in columns:
        values = frame[column].fillna("").astype(str).str.lower()
        if column in {"is_mock", "is_temporary"}:
            mask = values.isin({"true", "1", "yes", "y", "예"})
        else:
            mask = values.apply(lambda text: any(marker in text for marker in MOCK_MARKERS))
        if mask.any():
            found.append(f"{column}: {int(mask.sum())}행")
    return found


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--allow-mock", action="store_true")
    parser.add_argument("--allow-small", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    print(f"[INFO] data_dir: {data_dir}")

    errors: list[str] = []
    warnings: list[str] = []
    frames: dict[str, pd.DataFrame] = {}

    for filename, required_columns in REQUIRED_FILES.items():
        path = data_dir / filename
        if not path.exists():
            errors.append(f"필수 파일 없음: {filename}")
            continue
        try:
            frame = read_csv(path)
        except Exception as exc:
            errors.append(f"{filename} 읽기 실패: {exc}")
            continue
        frame.columns = [str(column).lstrip("\ufeff").strip() for column in frame.columns]
        missing = required_columns - set(frame.columns)
        if missing:
            errors.append(f"{filename} 필수 컬럼 누락: {sorted(missing)}")
        frames[filename] = frame
        print(f"[PASS] {filename}: {len(frame):,}행, {len(frame.columns)}열")

    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        return 1

    notices = frames["housing_supply_notice_master.csv"]
    chunks = frames["housing_rag_chunks_review_ready.csv"]
    rules = frames["housing_notice_document_rules_review_ready.csv"]
    documents = frames["housing_document_master.csv"]

    notice_ids = set(notices["notice_id"].dropna().astype(str).str.strip()) - {""}
    chunk_notice_ids = set(chunks["notice_id"].dropna().astype(str).str.strip()) - {""}
    orphan_chunk_ids = chunk_notice_ids - notice_ids
    if orphan_chunk_ids:
        errors.append(f"공고 마스터에 없는 notice_id를 가진 청크: {len(orphan_chunk_ids)}개")
    else:
        print("[PASS] 모든 RAG 청크 notice_id가 공고 마스터에 존재")

    document_ids = set(documents["document_id"].dropna().astype(str).str.strip()) - {""}
    rule_document_ids = set(rules["document_id"].dropna().astype(str).str.strip()) - {""}
    orphan_document_ids = rule_document_ids - document_ids
    if orphan_document_ids:
        errors.append(f"서류 마스터에 없는 document_id 규칙: {len(orphan_document_ids)}개")
    else:
        print("[PASS] 모든 서류 규칙 document_id가 서류 마스터에 존재")

    duplicate_chunks = int(chunks["chunk_id"].astype(str).duplicated().sum())
    if duplicate_chunks:
        errors.append(f"중복 chunk_id: {duplicate_chunks}개")
    else:
        print("[PASS] chunk_id 중복 없음")

    empty_text = int(chunks["retrieval_text"].fillna("").astype(str).str.strip().eq("").sum())
    if empty_text:
        errors.append(f"retrieval_text가 빈 청크: {empty_text}개")
    else:
        print("[PASS] retrieval_text 빈 값 없음")

    mock_findings = []
    for filename, frame in frames.items():
        for finding in contains_mock_marker(frame):
            mock_findings.append(f"{filename} / {finding}")
    if mock_findings and not args.allow_mock:
        errors.append("개발용·샘플 표식 발견: " + " | ".join(mock_findings[:12]))
    elif mock_findings:
        warnings.append("개발용 표식을 허용하고 검증함: " + " | ".join(mock_findings[:12]))
    else:
        print("[PASS] 개발용·샘플 표식 없음")

    unique_notice_count = len(notice_ids)
    chunk_count = len(chunks)
    if not args.allow_small and (unique_notice_count < 10 or chunk_count < 100):
        errors.append(
            f"데이터 규모가 지나치게 작음: 공고 {unique_notice_count}개, 청크 {chunk_count}개. "
            "실제 1차 구축본은 약 73공고·4,060청크입니다."
        )
    else:
        print(f"[PASS] 데이터 규모: 공고 {unique_notice_count:,}개 · 청크 {chunk_count:,}개")

    if "chunk_type" in chunks.columns:
        counts = chunks["chunk_type"].fillna("미입력").astype(str).value_counts()
        print("[INFO] chunk_type 상위 분포")
        for name, count in counts.head(20).items():
            print(f"  - {name}: {count:,}")
        for required_type in ("income_asset", "eligibility_common", "application_method", "caution"):
            if required_type not in set(counts.index):
                warnings.append(f"권장 chunk_type 없음: {required_type}")

    if not errors:
        try:
            retriever = PolicyRetriever(
                data_path=data_dir,
                data_mode="real",
                backend="keyword",
            )
            print(f"[PASS] 실제 CSV RAG 로더 초기화: {retriever.chunk_count:,} chunks")
            profile = {
                "sido": "경기도", "sigungu": "수원시", "birth_year": 2002,
                "owns_house": False, "residence_status": "월세", "household_size": 1,
                "income_known": False, "monthly_income": None,
            }
            connected = 0
            for policy_id in retriever.policy_ids:
                result = retriever.retrieve_policy_evidence(
                    policy_id=policy_id,
                    user_profile=profile,
                    top_k=1,
                    min_score=0.0,
                )
                if result["retrieval_status"] in {"success", "partial"}:
                    connected += 1
                    title = result.get("policy_name") or policy_id
                    print(
                        f"[PASS] {policy_id}: {result['retrieval_status']} / "
                        f"{result.get('selected_notice_ids')} / {title}"
                    )
                else:
                    print(f"[INFO] {policy_id}: {result['retrieval_status']}")
            if connected == 0:
                errors.append("기본 정책 계열 중 실제 공고와 연결된 결과가 없습니다.")
        except RetrieverConfigurationError as exc:
            errors.append(f"실제 CSV RAG 로더 초기화 실패: {exc}")
        except Exception as exc:
            errors.append(f"검색 smoke test 실패: {type(exc).__name__}: {exc}")

    for warning in warnings:
        print(f"[WARN] {warning}")
    for error in errors:
        print(f"[FAIL] {error}")

    if errors:
        print(f"\n검증 실패: {len(errors)}개")
        return 1
    print("\n모든 실제 데이터 검증을 통과했습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
