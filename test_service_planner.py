# -*- coding: utf-8 -*-
"""실제 공고 기반 신청 준비 계획 서비스 테스트."""

from service_planner import (
    HousingPlanRepository,
    ServiceApplicationPlanner,
)


def main() -> None:
    repository = HousingPlanRepository()
    planner = ServiceApplicationPlanner(repository)

    profile = {
        "sido": "경기도",
        "sigungu": "수원시",
        "birth_year": 2002,
        "age": 24,
        "owns_house": False,
        "residence_status": "월세",
        "household_size": 1,
        "income_known": False,
        "monthly_income": None,
    }

    # 청년 매입임대 추천 결과 -> 관련 실제 공고
    result = planner.find_related_notices(
        policy_id="LH_MAEIP",
        user_profile=profile,
        limit=5,
    )

    assert result["supported"] is True
    assert result["candidates"], "관련 실제 공고가 없습니다."

    candidate = result["candidates"][0]
    assert candidate["notice_id"]
    assert candidate["record_id"]
    assert "매입임대" in candidate["housing_type"] or "청년주택" in candidate["housing_type"]

    print("[PASS] 추천 서비스와 실제 공고 연결")
    print(
        candidate["notice_id"],
        candidate["announcement_title"],
        candidate["province"],
        candidate["city_county_district"],
    )

    initial_plan = planner.build_plan(
        notice_id=candidate["notice_id"],
        record_id=candidate["record_id"],
        user_profile=profile,
        condition_answers={},
    )

    assert initial_plan["notice"]["notice_id"] == candidate["notice_id"]
    assert initial_plan["questions"], "추가 조건 질문이 없습니다."
    assert initial_plan["document_plan"]["documents"], "서류 계획이 없습니다."

    print("[PASS] 공고별 추가 조건 질문 생성")
    print("[PASS] 공고별 서류 규칙과 발급 가이드 조인")

    # 한 조건에 답변한 뒤 서류 계획 재생성
    answers = {
        "applicant_type_student": False,
        "applicant_type_jobseeker": True,
        "proxy_application": False,
    }
    answered_plan = planner.build_plan(
        notice_id=candidate["notice_id"],
        record_id=candidate["record_id"],
        user_profile=profile,
        condition_answers=answers,
    )

    documents = answered_plan["document_plan"]["documents"]
    assert documents

    for document in documents:
        assert document["document_id"]
        assert document["canonical_name"]
        assert "source_references" in document
        assert document["status"] in {
            "common",
            "applicable",
            "needs_answer",
        }

    print("[PASS] 조건 답변에 따른 서류 분류")
    print("[PASS] 원문 파일·페이지 근거 포함")

    # 지원되지 않는 일반 정책은 잘못 연결하지 않음
    unsupported = planner.find_related_notices(
        policy_id="RENT_SUPPORT",
        user_profile=profile,
    )
    assert unsupported["supported"] is False
    assert unsupported["candidates"] == []

    print("[PASS] 관련 공고가 없는 서비스의 안전 처리")
    print("\n모든 신청 준비 계획 테스트를 통과했습니다.")


if __name__ == "__main__":
    main()
