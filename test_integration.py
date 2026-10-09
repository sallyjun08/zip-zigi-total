# -*- coding: utf-8 -*-
"""추천 엔진 -> RAG 검색 연결 통합 테스트."""

from recommendation_engine import UserProfile, recommend_policies
from rag_retriever import PolicyRetriever


def main() -> None:
    profile = UserProfile(
        sido="경기도",
        sigungu="수원시",
        birth_year=2002,
        owns_house=False,
        residence_status="월세",
        household_size=1,
        income_known=False,
        monthly_income=None,
    )

    matches = recommend_policies(profile)
    assert matches, "추천 엔진 결과가 비어 있습니다."

    retriever = PolicyRetriever(backend="keyword")
    rag_results = retriever.retrieve_for_matches(
        matches=matches,
        user_profile=profile,
        max_policies=None,
        top_k_per_section=1,
        min_score=0.0,
    )

    assert len(rag_results) == len(matches)

    match_ids = [match.policy.policy_id for match in matches]
    rag_ids = [result["policy_id"] for result in rag_results]
    assert match_ids == rag_ids, "추천 결과와 RAG 결과 순서가 다릅니다."

    for result in rag_results:
        policy_id = result["policy_id"]
        for items in result["evidence"].values():
            assert items, f"{policy_id}에 검색 결과가 없습니다."
            assert all(
                item["policy_id"] == policy_id
                for item in items
            ), "다른 정책의 문서가 섞였습니다."

    print("[PASS] 추천 엔진 실행")
    print("[PASS] 추천 정책 ID와 RAG 문서 연결")
    print("[PASS] 정책별 section 검색")
    print("[PASS] 다른 정책 근거 혼입 방지")
    print()
    for match, rag in zip(matches, rag_results):
        print(
            match.policy.policy_id,
            match.role_base_score,
            rag["retrieval_status"],
        )


if __name__ == "__main__":
    main()
