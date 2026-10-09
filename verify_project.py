# -*- coding: utf-8 -*-
"""Policy Bridge v2 프로젝트 구조와 핵심 로직을 자동 검증합니다."""

from __future__ import annotations

import argparse
import compileall
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

BASE_DIR = Path(__file__).resolve().parent

REQUIRED_FILES = [
    "app.py",
    "rag_retriever.py",
    "recommendation_engine.py",
    "easy_explainer.py",
    "service_planner.py",
    "requirements.txt",
    "data/policy_chunks.json",
    "data/housing_supply_notice_master.csv",
    "data/housing_document_master.csv",
    "data/housing_notice_document_rules_review_ready.csv",
    "data/housing_rag_chunks_review_ready.csv",
]

TEST_FILES = [
    "test_hybrid_backend.py",
    "test_retriever.py",
    "test_integration.py",
    "test_easy_explainer.py",
    "test_service_planner.py",
]

REQUIRED_CHUNK_FIELDS = {
    "chunk_id",
    "policy_id",
    "policy_name",
    "section_type",
    "section_title",
    "text",
    "source_name",
    "source_url",
    "updated_at",
    "is_mock",
}

EXPECTED_POLICY_IDS = {
    "RENT_SUPPORT",
    "JEONSE_LOAN",
    "LH_MAEIP",
    "HAPPY_HOUSE",
    "YOUTH_SAVINGS_JEONGYAK",
    "VULNERABLE_HOUSING_RELOCATION",
}


def fail(message: str) -> None:
    raise RuntimeError(message)


def check_files() -> None:
    missing = [name for name in REQUIRED_FILES if not (BASE_DIR / name).exists()]
    if missing:
        fail(f"필수 파일 누락: {missing}")
    print(f"[PASS] 필수 파일 {len(REQUIRED_FILES)}개 확인")


def check_syntax() -> None:
    ok = compileall.compile_dir(
        str(BASE_DIR),
        quiet=1,
        force=True,
        rx=None,
    )
    if not ok:
        fail("Python 문법 검사 실패")
    print("[PASS] 전체 Python 문법 검사")


def check_policy_chunks() -> None:
    path = BASE_DIR / "data" / "policy_chunks.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    chunks = payload.get("chunks") if isinstance(payload, dict) else payload
    if not isinstance(chunks, list) or not chunks:
        fail("policy_chunks.json에 chunks가 없습니다.")

    seen: set[str] = set()
    policy_ids: set[str] = set()
    sections_by_policy: dict[str, set[str]] = {}
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            fail(f"{index}번째 chunk가 객체가 아닙니다.")
        missing = REQUIRED_CHUNK_FIELDS - set(chunk)
        if missing:
            fail(f"{index}번째 chunk 필드 누락: {sorted(missing)}")
        chunk_id = str(chunk["chunk_id"])
        if chunk_id in seen:
            fail(f"중복 chunk_id: {chunk_id}")
        seen.add(chunk_id)
        policy_id = str(chunk["policy_id"])
        policy_ids.add(policy_id)
        sections_by_policy.setdefault(policy_id, set()).add(str(chunk["section_type"]))

    if policy_ids != EXPECTED_POLICY_IDS:
        fail(f"정책 ID 불일치: {sorted(policy_ids)}")

    expected_sections = {"eligibility", "income", "benefit", "application", "caution"}
    incomplete = {
        policy_id: sorted(expected_sections - sections)
        for policy_id, sections in sections_by_policy.items()
        if sections != expected_sections
    }
    if incomplete:
        fail(f"정책별 section 누락: {incomplete}")

    print(f"[PASS] 정책 RAG 데이터 {len(chunks)}개 chunk 검증")


def check_defaults() -> None:
    old_backend = os.environ.get("RAG_BACKEND")
    old_model = os.environ.get("RAG_EMBEDDING_MODEL")
    os.environ.pop("RAG_BACKEND", None)
    os.environ.pop("RAG_EMBEDDING_MODEL", None)
    try:
        import rag_retriever
        if rag_retriever.DEFAULT_MODEL_NAME != "BAAI/bge-m3":
            fail(f"기본 임베딩 모델 불일치: {rag_retriever.DEFAULT_MODEL_NAME}")
        if abs(rag_retriever.DEFAULT_HYBRID_DENSE_WEIGHT - 0.70) > 1e-9:
            fail("Dense 기본 가중치가 0.70이 아닙니다.")
        if abs(rag_retriever.DEFAULT_HYBRID_KEYWORD_WEIGHT - 0.30) > 1e-9:
            fail("Keyword 기본 가중치가 0.30이 아닙니다.")
    finally:
        if old_backend is not None:
            os.environ["RAG_BACKEND"] = old_backend
        if old_model is not None:
            os.environ["RAG_EMBEDDING_MODEL"] = old_model
    print("[PASS] BGE-M3 + V2 Hybrid 기본 설정 확인")


def run_tests() -> None:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["LLM_PROVIDER"] = "mock"
    env["RAG_BACKEND"] = "keyword"

    for test_file in TEST_FILES:
        path = BASE_DIR / test_file
        completed = subprocess.run(
            [sys.executable, str(path)],
            cwd=str(BASE_DIR),
            env=env,
            text=True,
            capture_output=True,
        )
        if completed.returncode != 0:
            print(completed.stdout)
            print(completed.stderr, file=sys.stderr)
            fail(f"테스트 실패: {test_file}")
        print(f"[PASS] {test_file}")


def check_runtime_dependencies(strict: bool) -> None:
    packages = ["numpy", "pandas", "streamlit", "sentence_transformers"]
    missing = [name for name in packages if importlib.util.find_spec(name) is None]
    if missing:
        message = "설치되지 않은 실행 패키지: " + ", ".join(missing)
        if strict:
            fail(message + " — setup_windows.ps1을 먼저 실행하세요.")
        print("[WARN] " + message)
    else:
        print("[PASS] 실행 패키지 설치 확인")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strict-dependencies",
        action="store_true",
        help="streamlit과 sentence-transformers 설치 여부도 실패 조건으로 처리합니다.",
    )
    args = parser.parse_args()

    print("Policy Bridge v2 검증 시작")
    print(f"프로젝트: {BASE_DIR}")
    print(f"Python: {sys.version.split()[0]}")
    check_files()
    check_syntax()
    check_policy_chunks()
    check_defaults()
    run_tests()
    check_runtime_dependencies(args.strict_dependencies)
    print("\n[PASS] 프로젝트 핵심 구조와 로직 검증 완료")


if __name__ == "__main__":
    main()
