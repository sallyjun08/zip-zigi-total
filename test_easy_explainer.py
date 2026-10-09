# -*- coding: utf-8 -*-
"""LLM 설명 생성기의 컨텍스트, 검증, fallback 테스트."""

from __future__ import annotations

import json

from easy_explainer import (
    EasyExplainer,
    MockProvider,
    ProviderResponse,
)
from recommendation_engine import UserProfile, recommend_policies
from rag_retriever import PolicyRetriever


class InvalidJsonProvider:
    provider_name = "invalid-json"
    model_name = "invalid-json-model"

    def generate(self, **kwargs):
        return ProviderResponse(
            content="JSON이 아닌 응답입니다.",
            model=self.model_name,
            metadata={},
        )


class HallucinationProvider:
    provider_name = "hallucination"
    model_name = "hallucination-model"

    def generate(self, *, context, **kwargs):
        result = {
            "policy_id": context["policy"]["policy_id"],
            "easy_summary": "매월 999만원을 반드시 받을 수 있습니다.",
            "recommendation_reason": "신청 자격이 됩니다.",
            "additional_checks": ["별도 확인이 필요하지 않습니다."],
            "caution": "확실히 승인됩니다.",
            "used_chunk_ids": ["invented_chunk_999"],
        }
        return ProviderResponse(
            content=json.dumps(result, ensure_ascii=False),
            model=self.model_name,
            metadata={},
        )


def prepare():
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
    retriever = PolicyRetriever(backend="keyword")
    rag_results = retriever.retrieve_for_matches(
        matches=matches,
        user_profile=profile,
        max_policies=None,
        top_k_per_section=1,
        min_score=0.0,
    )

    possibilities = {
        match.policy.policy_id: "추가 확인 필요"
        for match in matches
    }
    return profile, matches, rag_results, possibilities


def test_mock_generation():
    profile, matches, rag_results, possibilities = prepare()

    explainer = EasyExplainer(
        provider=MockProvider(),
        max_retries=1,
        max_llm_policies=3,
    )
    explanations = explainer.explain_matches(
        matches=matches,
        retrieval_results=rag_results,
        user_profile=profile,
        possibilities=possibilities,
    )

    assert len(explanations) == len(matches)
    assert explanations[0]["generation_status"] == "generated"
    assert explanations[0]["policy_id"] == matches[0].policy.policy_id

    allowed = {
        item["chunk_id"]
        for section in rag_results[0]["evidence"].values()
        for item in section
    }
    assert set(explanations[0]["used_chunk_ids"]).issubset(allowed)

    # LLM_MAX_POLICIES=3과 동일하게 네 번째 정책부터는 안전 fallback을 사용합니다.
    if len(explanations) >= 4:
        assert explanations[3]["generation_status"] == "fallback_disabled"

    print("[PASS] Mock LLM 구조화 설명 생성")
    print("[PASS] 사용 chunk ID 검증")
    print("[PASS] 상위 정책 수 제한과 fallback")


def test_invalid_json_fallback():
    profile, matches, rag_results, possibilities = prepare()

    explainer = EasyExplainer(
        provider=InvalidJsonProvider(),
        max_retries=1,
        max_llm_policies=1,
    )
    result = explainer.explain_policy(
        match=matches[0],
        retrieval_result=rag_results[0],
        user_profile=profile,
        rank=1,
        possibility=possibilities[matches[0].policy.policy_id],
    )

    assert result["generation_status"] == "fallback_error"
    assert result["error_type"] == "validation"
    assert result["easy_summary"]

    print("[PASS] 잘못된 JSON 응답 fallback")


def test_hallucination_fallback():
    profile, matches, rag_results, possibilities = prepare()

    explainer = EasyExplainer(
        provider=HallucinationProvider(),
        max_retries=0,
        max_llm_policies=1,
    )
    result = explainer.explain_policy(
        match=matches[0],
        retrieval_result=rag_results[0],
        user_profile=profile,
        rank=1,
        possibility=possibilities[matches[0].policy.policy_id],
    )

    assert result["generation_status"] == "fallback_error"
    assert result["error_type"] == "validation"
    assert "999만원" not in result["easy_summary"]
    assert "반드시 받을 수 있습니다" not in result["easy_summary"]

    print("[PASS] 근거 없는 금액·금지 표현·가짜 chunk 차단")


def main():
    test_mock_generation()
    test_invalid_json_fallback()
    test_hallucination_fallback()
    print("\n모든 LLM 설명 생성 테스트를 통과했습니다.")


if __name__ == "__main__":
    main()
