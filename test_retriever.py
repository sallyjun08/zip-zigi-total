# -*- coding: utf-8 -*-
"""RAG retriever 구조 테스트. 외부 모델 없이 keyword backend로 실행한다."""

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

    retriever = PolicyRetriever(backend="keyword")

    assert "RENT_SUPPORT" in retriever.policy_ids
    assert "HAPPY_HOUSE" in retriever.policy_ids

    result = retriever.retrieve_policy_evidence(
        policy_id="RENT_SUPPORT",
        user_profile=profile,
        top_k=1,
        min_score=0.0,
    )

    assert result["policy_id"] == "RENT_SUPPORT"
    assert result["retrieval_status"] == "success"
    assert result["is_mock_data"] is True

    for section_type, items in result["evidence"].items():
        assert items, f"{section_type} 검색 결과가 비었습니다."
        assert all(item["policy_id"] == "RENT_SUPPORT" for item in items), (
            "다른 정책 문서가 섞였습니다."
        )

    print("[PASS] policy_id 제한 검색")
    print("[PASS] 5개 section 검색")
    print("[PASS] 임시 데이터 표시")
    print()
    print(PolicyRetriever.format_llm_context(result))


if __name__ == "__main__":
    main()
