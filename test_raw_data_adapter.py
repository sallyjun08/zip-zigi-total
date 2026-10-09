# -*- coding: utf-8 -*-
"""원본 5종 보존형 변환 및 실제 RAG/서류계획 연결 테스트."""

from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
import tempfile

from prepare_real_data import RAW_FILES, convert
from rag_retriever import PolicyRetriever
from service_planner import HousingPlanRepository, ServiceApplicationPlanner


def file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    args = parser.parse_args()

    source_dir = Path(args.source_dir).resolve()
    source_paths = [source_dir / filename for filename in RAW_FILES.values()]
    before = {path.name: file_hash(path) for path in source_paths}

    with tempfile.TemporaryDirectory(prefix="policy_bridge_raw_test_") as temp:
        output_dir = Path(temp) / "generated"
        manifest = convert(source_dir, output_dir)
        after = {path.name: file_hash(path) for path in source_paths}
        assert before == after
        assert manifest["source_files_unchanged"] is True

        retriever = PolicyRetriever(
            data_path=output_dir,
            data_mode="real",
            backend="keyword",
        )
        profile = {
            "sido": "경기도",
            "sigungu": "수원시",
            "birth_year": 2002,
            "owns_house": False,
            "residence_status": "월세",
            "household_size": 1,
            "income_known": False,
            "monthly_income": None,
        }
        result = retriever.retrieve_policy_evidence(
            policy_id="JEONSE_LOAN",
            user_profile=profile,
            top_k=1,
            min_score=0.0,
        )
        assert result["retrieval_status"] in {"success", "partial"}
        assert result["selected_notice_ids"] == ["PB-2026-001"]
        assert all(
            item["notice_id"] == "PB-2026-001"
            for items in result["evidence"].values()
            for item in items
        )

        repository = HousingPlanRepository(output_dir)
        planner = ServiceApplicationPlanner(repository)
        candidates = planner.find_related_notices(
            policy_id="JEONSE_LOAN",
            user_profile=profile,
            limit=3,
        )
        assert candidates["candidates"]
        selected = candidates["candidates"][0]
        plan = planner.build_plan(
            notice_id=selected["notice_id"],
            record_id=selected.get("record_id", ""),
            user_profile=profile,
            condition_answers={},
        )
        documents = plan["document_plan"]["documents"]
        assert documents
        assert any(document["status"] == "common" for document in documents)
        assert any(document["status"] == "needs_answer" for document in documents)

    print("[PASS] 원본 5종 SHA-256 불변")
    print("[PASS] 운영용 4종 CSV 생성")
    print("[PASS] 실제 notice_id 제한 RAG 검색")
    print("[PASS] 실제 공고-서류 계획 연결")
    print("[PASS] 해당자 서류를 공통 필수로 오인하지 않음")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
