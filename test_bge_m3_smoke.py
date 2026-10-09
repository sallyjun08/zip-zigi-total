# -*- coding: utf-8 -*-
"""실제 BGE-M3 모델을 로드해 V2 Hybrid 검색 1회를 수행하는 선택 테스트."""

from __future__ import annotations

from rag_retriever import PolicyRetriever


def main() -> None:
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
    retriever = PolicyRetriever(backend="hybrid")
    result = retriever.retrieve_policy_evidence(
        policy_id="RENT_SUPPORT",
        user_profile=profile,
        top_k=1,
        min_score=0.0,
    )
    assert result["backend"] == "hybrid"
    assert result["embedding_model"] == "BAAI/bge-m3"
    assert result["retrieval_status"] == "success"
    first = result["evidence"]["benefit"][0]
    assert "dense_score" in first and "keyword_score" in first
    print("[PASS] BGE-M3 모델 로드")
    print("[PASS] Dense + BM25 V2 Hybrid 실제 검색")
    print(f"검색 결과: {first['chunk_id']} / score={first['similarity']}")


if __name__ == "__main__":
    main()
