# -*- coding: utf-8 -*-
"""
Policy Bridge Streamlit 통합 앱

사용자 입력
    -> recommendation_engine.py
    -> rag_retriever.py
    -> easy_explainer.py
    -> LLM 쉬운 설명이 포함된 결과 카드

LLM은 정책을 추천하거나 자격을 판정하지 않습니다.
추천 엔진과 RAG 결과를 쉬운 한국어로 설명하는 역할만 수행합니다.
"""

from __future__ import annotations

from datetime import date
from html import escape
from pathlib import Path
from textwrap import dedent
from typing import Any, Mapping
import json
import os

import streamlit as st
import streamlit.components.v1 as components

from recommendation_engine import MatchResult, UserProfile, recommend_policies
from rag_retriever import PolicyRetriever
from easy_explainer import EasyExplainer, create_easy_explainer_from_env
from service_planner import (
    HousingPlanRepository,
    ServiceApplicationPlanner,
)


st.set_page_config(
    page_title="Zip.지기 - 청년 주거 독립 연구소",
    page_icon="🦉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
CURRENT_YEAR = date.today().year
DEFAULT_RAG_BACKEND = os.getenv("RAG_BACKEND", "hybrid").strip().lower()
DEFAULT_RAG_DATA_MODE = os.getenv("RAG_DATA_MODE", "real").strip().lower()
REQUIRE_REAL_EVIDENCE = os.getenv("RAG_REQUIRE_REAL_EVIDENCE", "1").strip().lower() not in {"0", "false", "no"}

REGIONS = [
    "서울특별시", "부산광역시", "대구광역시", "인천광역시", "광주광역시",
    "대전광역시", "울산광역시", "세종특별자치시", "경기도",
    "강원특별자치도", "충청북도", "충청남도", "전북특별자치도",
    "전라남도", "경상북도", "경상남도", "제주특별자치도",
]

SIGUNGU_BY_REGION = {
    "서울특별시": [
        "종로구", "중구", "용산구", "성동구", "광진구", "동대문구", "중랑구",
        "성북구", "강북구", "도봉구", "노원구", "은평구", "서대문구", "마포구",
        "양천구", "강서구", "구로구", "금천구", "영등포구", "동작구", "관악구",
        "서초구", "강남구", "송파구", "강동구",
    ],
    "부산광역시": [
        "중구", "서구", "동구", "영도구", "부산진구", "동래구", "남구", "북구",
        "해운대구", "사하구", "금정구", "강서구", "연제구", "수영구", "사상구", "기장군",
    ],
    "대구광역시": [
        "중구", "동구", "서구", "남구", "북구", "수성구", "달서구", "달성군", "군위군",
    ],
    "인천광역시": [
        "중구", "동구", "미추홀구", "연수구", "남동구", "부평구",
        "계양구", "서구", "강화군", "옹진군",
    ],
    "광주광역시": ["동구", "서구", "남구", "북구", "광산구"],
    "대전광역시": ["동구", "중구", "서구", "유성구", "대덕구"],
    "울산광역시": ["중구", "남구", "동구", "북구", "울주군"],
    "세종특별자치시": ["세종특별자치시"],
    "경기도": [
        "수원시", "성남시", "의정부시", "안양시", "부천시", "광명시", "평택시",
        "동두천시", "안산시", "고양시", "과천시", "구리시", "남양주시", "오산시",
        "시흥시", "군포시", "의왕시", "하남시", "용인시", "파주시", "이천시",
        "안성시", "김포시", "화성시", "광주시", "양주시", "포천시", "여주시",
        "연천군", "가평군", "양평군",
    ],
    "강원특별자치도": [
        "춘천시", "원주시", "강릉시", "동해시", "태백시", "속초시", "삼척시",
        "홍천군", "횡성군", "영월군", "평창군", "정선군", "철원군", "화천군",
        "양구군", "인제군", "고성군", "양양군",
    ],
    "충청북도": [
        "청주시", "충주시", "제천시", "보은군", "옥천군", "영동군", "증평군",
        "진천군", "괴산군", "음성군", "단양군",
    ],
    "충청남도": [
        "천안시", "공주시", "보령시", "아산시", "서산시", "논산시", "계룡시",
        "당진시", "금산군", "부여군", "서천군", "청양군", "홍성군", "예산군", "태안군",
    ],
    "전북특별자치도": [
        "전주시", "군산시", "익산시", "정읍시", "남원시", "김제시", "완주군",
        "진안군", "무주군", "장수군", "임실군", "순창군", "고창군", "부안군",
    ],
    "전라남도": [
        "목포시", "여수시", "순천시", "나주시", "광양시", "담양군", "곡성군",
        "구례군", "고흥군", "보성군", "화순군", "장흥군", "강진군", "해남군",
        "영암군", "무안군", "함평군", "영광군", "장성군", "완도군", "진도군", "신안군",
    ],
    "경상북도": [
        "포항시", "경주시", "김천시", "안동시", "구미시", "영주시", "영천시",
        "상주시", "문경시", "경산시", "의성군", "청송군", "영양군", "영덕군",
        "청도군", "고령군", "성주군", "칠곡군", "예천군", "봉화군", "울진군", "울릉군",
    ],
    "경상남도": [
        "창원시", "진주시", "통영시", "사천시", "김해시", "밀양시", "거제시",
        "양산시", "의령군", "함안군", "창녕군", "고성군", "남해군", "하동군",
        "산청군", "함양군", "거창군", "합천군",
    ],
    "제주특별자치도": ["제주시", "서귀포시"],
}

HOUSE_OWNERSHIP_MAP = {
    "무주택자": False,
    "유주택자": True,
    "잘 모르겠음": None,
}

RESIDENCE_STATUS_MAP = {
    "부모님과 거주": "부모님과 거주",
    "월세": "월세",
    "전세": "전세",
    "집 구하는 중": "집 구하는 중",
    "고시원·기숙사": "고시원-기숙사",
    "쪽방·비닐하우스 등": "주거취약시설",
    "기타": "기타",
}

HOUSEHOLD_SIZE_MAP = {
    "1인 가구": 1,
    "2인 가구": 2,
    "3인 가구": 3,
    "4인 이상": 4,
}

POLICY_CATEGORY = {
    "RENT_SUPPORT": "월세 지원",
    "JEONSE_LOAN": "전세자금 대출",
    "LH_MAEIP": "공공임대주택",
    "HAPPY_HOUSE": "공공임대주택",
    "YOUTH_SAVINGS_JEONGYAK": "주택청약·금융",
    "VULNERABLE_HOUSING_RELOCATION": "주거취약계층 이주지원",
}

SECTION_LABELS = {
    "eligibility": "지원 대상",
    "income": "소득·재산 조건",
    "benefit": "지원 혜택",
    "application": "신청 방법",
    "caution": "추가 확인 사항",
}

STATE_DEFAULTS = {
    "page": "input",
    "user_input": {},
    "user_profile_data": {},
    "recommendations": [],
    "rag_backend": "",
    "rag_warning": "",
    "llm_provider": "",
    "llm_model": "",
    "llm_warning": "",
    "selected_policy_card": {},
    "selected_notice_id": "",
    "selected_record_id": "",
    "condition_answers": {},
    "input_guide": "default",
    "selected_document_id": "",
    "celebrated_plan": "",
    "active_drawer": {},
    "_popup_url_map": {},
    "success_modal": {},
}

for key, value in STATE_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


@st.cache_resource(show_spinner=False)
def get_policy_retriever(
    requested_backend: str,
) -> tuple[PolicyRetriever, str, str]:
    supported = {"hybrid", "semantic", "keyword"}
    backend = requested_backend if requested_backend in supported else "hybrid"
    fallback_order = {
        "hybrid": ["hybrid", "semantic", "keyword"],
        "semantic": ["semantic", "keyword"],
        "keyword": ["keyword"],
    }[backend]

    errors: list[str] = []
    for candidate in fallback_order:
        try:
            retriever = PolicyRetriever(
                data_path=BASE_DIR / "data",
                data_mode=DEFAULT_RAG_DATA_MODE,
                backend=candidate,
                cache_dir=BASE_DIR / "vector_store",
            )
            warning = ""
            if candidate != backend:
                warning = (
                    f"요청한 {backend} 검색을 초기화하지 못해 {candidate} 검색으로 전환했습니다. "
                    + " | ".join(errors)
                )
            return retriever, candidate, warning
        except Exception as exc:
            errors.append(f"{candidate}: {type(exc).__name__}: {exc}")

    raise RuntimeError("RAG 검색기를 초기화하지 못했습니다. " + " | ".join(errors))


@st.cache_resource(show_spinner=False)
def get_easy_explainer() -> EasyExplainer:
    return create_easy_explainer_from_env()


@st.cache_resource(show_spinner=False)
def get_service_application_planner() -> ServiceApplicationPlanner:
    repository = HousingPlanRepository(
        data_dir=BASE_DIR / "data",
    )
    return ServiceApplicationPlanner(repository)


def get_document_preview(
    policy_id: str,
    profile_data: Mapping[str, Any],
    limit: int = 4,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """가장 유력한 실제 공고 1건의 서류 미리보기(최대 limit건)를 반환합니다."""
    try:
        planner = get_service_application_planner()
        notice_result = planner.find_related_notices(
            policy_id=policy_id,
            user_profile=profile_data,
            limit=1,
        )
        candidates = (
            notice_result.get("candidates", [])
            if notice_result.get("supported")
            else []
        )
        if not candidates:
            return None, []

        notice = candidates[0]
        plan = planner.build_plan(
            notice_id=notice["notice_id"],
            record_id=notice["record_id"],
            user_profile=profile_data,
            condition_answers=None,
        )
        return notice, plan["document_plan"]["documents"][:limit]
    except Exception:
        return None, []


def build_user_profile(
    region: str,
    sigungu: str,
    birth_year: int,
    housing_owned: str,
    living_status: str,
    household_size: int | str,
    income_type: str,
    monthly_income_manwon: int | None,
) -> UserProfile:
    income_known = income_type == "직접 입력"
    monthly_income_won = (
        int(monthly_income_manwon) * 10_000
        if income_known and monthly_income_manwon is not None
        else None
    )

    if isinstance(household_size, int):
        household_count = household_size
    else:
        household_count = HOUSEHOLD_SIZE_MAP[household_size]

    return UserProfile(
        sido=region,
        sigungu=sigungu,
        birth_year=int(birth_year),
        owns_house=HOUSE_OWNERSHIP_MAP[housing_owned],
        residence_status=RESIDENCE_STATUS_MAP[living_status],
        household_size=household_count,
        income_known=income_known,
        monthly_income=monthly_income_won,
    )


def build_user_summary(
    profile: UserProfile,
    registered_region: str,
    registered_sigungu: str,
    housing_owned: str,
    living_status: str,
    household_size: int | str,
    income_type: str,
    monthly_income_manwon: int | None,
    income_context: str = "",
) -> dict[str, str]:
    summary = {
        "희망 거주지": f"{profile.sido} {profile.sigungu}",
        "주민등록 주소": f"{registered_region} {registered_sigungu}",
        "만 나이": f"만 {profile.age}세 전후",
        "주택 소유 여부": housing_owned,
        "현재 거주 상태": living_status,
        "가구 인원 수": (
            f"{household_size}인 가구"
            if isinstance(household_size, int)
            else household_size
        ),
        "월 소득": (
            "모름"
            if income_type == "모름"
            else f"{int(monthly_income_manwon or 0):,}만 원"
        ),
    }
    if income_type == "모름" and income_context.strip():
        summary["소득 참고 정보"] = income_context.strip()
    return summary

# HTML 시안의 툴팁 포맷을 반영한 가이드 문구
INPUT_GUIDES = {
    "default": (
        "왼쪽에서 나만의 조건들을 톡톡 클릭해 볼까요? ✨<br>"
        "작성하다가 헷갈리는 질문이 나오면, 제가 바로 이 자리에서 <span class=\"tooltip-trigger\">친절하게 설명<span class=\"tooltip-content\">행정 용어나 복잡한 정책 조건 위에 마우스를 올려보세요! 제가 알기 쉽게 쏙쏙 풀어서 설명해 드릴게요.</span></span>해 드릴게요."
    ),
    "desired_region": (
        "희망 <span class=\"highlight\">거주 지역</span>에 따라 지자체 전용 지원금이나 정책 트랙이 달라지니 꼼꼼하게 선택해 주세요! 🏡"
    ),
    "registered_region": (
        "<span class=\"highlight\">주민등록상 주소지</span>는 지역 우선 공급이나 당해 거주자 혜택을 판정할 때 아주 중요한 기준이 된답니다. 📍"
    ),
    "age": (
        "청년 주거 정책은 보통 <span class=\"highlight\">만 19세부터 34세 혹은 39세</span>까지를 기준으로 삼는 경우가 많아요. 나이에 맞는 혜택을 찾아볼게요! ⏳"
    ),
    "housing": (
        "집이 없더라도 안심하세요! <span class=\"tooltip-trigger\">무주택세대구성원<span class=\"tooltip-content\">‘세대’에 해당하는 사람 전원이 주택을 소유하고 있지 아니한 세대입니다. 분양권도 포함되니 주의하세요!</span></span> 조건만 맞으면 충분히 신청할 수 있답니다. ✨"
    ),
    "income": (
        "소득 정보를 입력해주시면, <span class=\"tooltip-trigger\">기준중위소득<span class=\"tooltip-content\">보건복지부장관이 고시하는 국민 가구소득의 중위값으로, 임대주택 소득 심사의 핵심 기준입니다.</span></span> 대비 내 위치를 파악해서 가장 유리한 공고를 찾아드릴게요! 💵"
    ),
    "income_context": (
        "부모님의 직업이나 상황을 편하게 설명해 주시면, 제가 찰떡같이 이해하고 소득을 추정해 볼게요! 🦉✨"
    ),
    "household_size": (
        "가구원 수에 따라 <span class=\"highlight\">기준중위소득 컷</span>이 달라지니, 본인을 포함한 정확한 세대 구성 인원 수를 입력해 주세요! 👨‍👩‍👧‍👦"
    ),
    "details": (
        "현재 생활 조건을 더하면 월세·전세·공공임대 후보를 더 정확히 비교할 수 있어요. 🔎"
    ),
}

def set_input_guide(guide_key: str) -> None:
    st.session_state.input_guide = guide_key


def score_to_possibility(score: float, profile: UserProfile) -> str:
    has_unknown = profile.owns_house is None or not profile.income_known

    if score >= 70 and not has_unknown:
        return "가능성 높음"
    if score >= 35:
        return "추가 확인 필요"
    return "가능성 낮음"


def first_evidence_text(
    rag_result: Mapping[str, Any] | None,
    section_type: str,
    fallback: str,
) -> str:
    if not rag_result:
        return fallback

    items = rag_result.get("evidence", {}).get(section_type, [])
    if items:
        return str(items[0].get("text", fallback))

    return fallback


def collect_check_items(
    rag_result: Mapping[str, Any] | None,
    fallback_income_note: str,
) -> list[str]:
    checks: list[str] = []

    if rag_result:
        for section_type in ("income", "caution", "application"):
            items = rag_result.get("evidence", {}).get(section_type, [])
            if items:
                checks.append(str(items[0]["text"]))

    if not checks:
        checks.append(fallback_income_note)
        checks.append("정확한 신청 기간과 제출 서류는 공식 페이지에서 확인하세요.")

    unique_checks: list[str] = []
    for item in checks:
        if item and item not in unique_checks:
            unique_checks.append(item)

    return unique_checks[:3]


def build_reason(
    profile: UserProfile,
    match: MatchResult,
    rag_result: Mapping[str, Any] | None,
) -> str:
    eligibility = first_evidence_text(
        rag_result,
        "eligibility",
        "나이, 지역, 주택 소유 여부와 거주 상태가 1차 후보 조건을 통과했습니다.",
    )

    parts = [
        (
            f"만 {profile.age}세 전후, {profile.sido} {profile.sigungu}, "
            f"{profile.residence_status} 조건으로 추천 후보에 포함되었습니다."
        ),
        eligibility,
        match.income_note,
    ]

    if profile.owns_house is None:
        parts.append("주택 소유 여부가 불확실해 무주택 조건 확인이 필요합니다.")

    return " ".join(parts)


def convert_matches_to_cards(
    profile: UserProfile,
    matches: list[MatchResult],
    rag_results: list[dict[str, Any]],
    explanations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rag_by_policy = {result["policy_id"]: result for result in rag_results}
    explanation_by_policy = {result["policy_id"]: result for result in explanations}
    cards: list[dict[str, Any]] = []

    for rank, match in enumerate(matches, start=1):
        policy = match.policy
        rag_result = rag_by_policy.get(policy.policy_id)
        explanation = explanation_by_policy.get(policy.policy_id, {})
        matched_notices = list(rag_result.get("matched_notices", [])) if rag_result else []
        primary_notice = matched_notices[0] if matched_notices else {}

        easy_description = explanation.get(
            "easy_summary",
            first_evidence_text(rag_result, "benefit", policy.description),
        )
        reason = explanation.get(
            "recommendation_reason",
            build_reason(profile, match, rag_result),
        )
        check_items = explanation.get(
            "additional_checks",
            collect_check_items(rag_result, match.income_note),
        )
        caution = explanation.get(
            "caution",
            "현재 결과는 최종 자격 확정이 아니므로 실제 공고 원문을 확인해야 합니다.",
        )

        source_url = (
            primary_notice.get("notice_url")
            or primary_notice.get("application_url")
            or policy.apply_deep_link
        )
        source_name = (
            primary_notice.get("operator")
            or primary_notice.get("source_file")
            or "정책 신청 페이지"
        )
        if rag_result:
            for section_items in rag_result.get("evidence", {}).values():
                if not section_items:
                    continue
                item = section_items[0]
                source_url = item.get("source_url") or source_url
                source_name = item.get("source_name") or source_name
                break

        actual_notice_title = str(primary_notice.get("announcement_title") or "").strip()
        display_name = actual_notice_title or policy.name
        cards.append(
            {
                "rank": rank,
                "policy_id": policy.policy_id,
                "policy_name": display_name,
                "service_name": policy.name,
                "category": POLICY_CATEGORY.get(policy.policy_id, "주거 지원"),
                "match_score": match.role_base_score,
                "notice_match_score": primary_notice.get("match_score"),
                "possibility": score_to_possibility(match.role_base_score, profile),
                "easy_description": easy_description,
                "reason": reason,
                "check_items": check_items,
                "caution": caution,
                "link": source_url,
                "source_name": source_name,
                "notice_id": primary_notice.get("notice_id", ""),
                "record_id": primary_notice.get("record_id", ""),
                "notice_region": " ".join(
                    value for value in [
                        str(primary_notice.get("province") or "").strip(),
                        str(primary_notice.get("city_county_district") or "").strip(),
                    ] if value
                ),
                "matched_notices": matched_notices,
                "relation_note": rag_result.get("relation_note", "") if rag_result else "",
                "data_mode": rag_result.get("data_mode", "") if rag_result else "",
                "rag_status": rag_result.get("retrieval_status", "not_found") if rag_result else "not_found",
                "is_mock_data": bool(rag_result.get("is_mock_data", True)) if rag_result else True,
                "evidence": rag_result.get("evidence", {}) if rag_result else {},
                "generation_status": explanation.get("generation_status", "fallback_missing"),
                "generation_error": explanation.get("generation_error", ""),
                "llm_provider": explanation.get("provider", ""),
                "llm_model": explanation.get("model", ""),
                "used_chunk_ids": explanation.get("used_chunk_ids", []),
            }
        )

    return cards

def run_recommendation_pipeline(
    profile: UserProfile,
) -> tuple[
    list[dict[str, Any]],
    str,
    str,
    str,
    str,
    str,
]:
    matches = recommend_policies(profile)
    if not matches:
        return [], "", "", "", "", ""

    retriever, backend, rag_warning = get_policy_retriever(
        DEFAULT_RAG_BACKEND
    )
    min_score = 0.0 if backend == "keyword" else 0.20

    rag_results = retriever.retrieve_for_matches(
        matches=matches,
        user_profile=profile,
        max_policies=None,
        top_k_per_section=1,
        min_score=min_score,
    )

    if retriever.data_mode == "real_csv" and REQUIRE_REAL_EVIDENCE:
        result_by_policy = {result["policy_id"]: result for result in rag_results}
        filtered_matches: list[MatchResult] = []
        filtered_results: list[dict[str, Any]] = []
        excluded: list[str] = []
        for match in matches:
            result = result_by_policy.get(match.policy.policy_id, {})
            has_evidence = any(result.get("evidence", {}).get(section) for section in SECTION_LABELS)
            is_actual = not bool(result.get("is_mock_data", True))
            if result.get("retrieval_status") in {"success", "partial"} and has_evidence and is_actual:
                filtered_matches.append(match)
                filtered_results.append(result)
            else:
                excluded.append(match.policy.name)
        matches = filtered_matches
        rag_results = filtered_results
        if excluded:
            extra = (
                "실제 공고 근거가 없거나 개발용 데이터로 판정된 서비스는 결과에서 제외했습니다: "
                + ", ".join(excluded)
            )
            rag_warning = f"{rag_warning} {extra}".strip()
        if not matches:
            return (
                [],
                f"{backend} · {retriever.data_mode}",
                rag_warning or "실제 공고 데이터에서 연결 가능한 근거를 찾지 못했습니다.",
                "",
                "",
                "",
            )

    possibilities = {
        match.policy.policy_id: score_to_possibility(match.role_base_score, profile)
        for match in matches
    }

    explainer = get_easy_explainer()
    explanations = explainer.explain_matches(
        matches=matches,
        retrieval_results=rag_results,
        user_profile=profile,
        possibilities=possibilities,
    )

    generated_count = sum(
        1
        for result in explanations
        if str(result.get("generation_status", "")).startswith("generated")
    )

    llm_warning = ""
    if explanations and generated_count == 0:
        first_error = next(
            (
                str(result.get("generation_error"))
                for result in explanations
                if result.get("generation_error")
            ),
            "",
        )
        llm_warning = (
            "LLM 설명을 생성하지 못해 RAG 근거 기반 안전 설명으로 표시했습니다."
        )
        if first_error:
            llm_warning += f" 원인: {first_error}"
    elif generated_count < min(
        len(explanations),
        explainer.max_llm_policies,
    ):
        llm_warning = (
            "일부 정책은 LLM 검증 실패 또는 생성 대상 수 제한으로 "
            "RAG 근거 기반 설명을 사용했습니다."
        )

    cards = convert_matches_to_cards(
        profile=profile,
        matches=matches,
        rag_results=rag_results,
        explanations=explanations,
    )

    return (
        cards,
        f"{backend} · {retriever.data_mode}",
        rag_warning,
        explainer.provider_name,
        explainer.model_name,
        llm_warning,
    )


def html_block(markup: str) -> None:
    compact = " ".join(
        line.strip()
        for line in dedent(markup).splitlines()
        if line.strip()
    )
    st.markdown(compact, unsafe_allow_html=True)


def render_side_popup_button(
    url: str,
    label: str = "🔗 공식 발급 사이트 열기",
) -> None:
    """발급 사이트를 화면 우측에 별도 창으로 띄웁니다.

    st.link_button은 항상 새 '탭'으로 열려 체크리스트 사이드바를 가리게
    됩니다. HTML 시안처럼 체크리스트를 계속 보면서 발급 사이트를 나란히
    쓸 수 있도록, 위치/크기가 지정된 팝업 창을 여는 버튼을 iframe에 직접
    그립니다. 클릭이 iframe 내부 JS에서 곧바로 처리되어(서버 왕복 없이)
    사용자 제스처로 인정되므로 팝업 차단기에 걸리지 않습니다.

    창 참조를 window.parent에 저장해 둡니다(noopener를 뺀 이유) — 체크리스트의
    체크박스를 클릭할 때마다 render_popup_bridge가 이 참조로 focus()를 호출해,
    앱 화면을 조작해도 발급 사이트 창이 다시 맨 앞으로 잠깐 올라오게 합니다.
    """
    safe_url = escape(url, quote=True)
    safe_label = escape(label)
    components.html(
        f"""
        <button id="portalOpenBtn" type="button" style="
            width:100%; min-height:52px; border:none; border-radius:14px;
            background:#6B705C; color:#fff; font-family:'Pretendard',sans-serif;
            font-size:15px; font-weight:700; cursor:pointer; transition:background .2s;
        " onmouseover="this.style.background='#4A503D'" onmouseout="this.style.background='#6B705C'">
            {safe_label}
        </button>
        <script>
        document.getElementById('portalOpenBtn').addEventListener('click', function () {{
            var popupWidth = 860;
            var popupHeight = 840;
            var left = Math.max(window.screen.width - popupWidth - 60, 40);
            var win = window.open(
                "{safe_url}", "_blank",
                "width=" + popupWidth + ",height=" + popupHeight +
                ",left=" + left + ",top=40,scrollbars=yes"
            );
            if (win) {{ window.parent.__docPopupWindowRef = win; }}
        }});
        </script>
        """,
        height=60,
    )


# housing_document_master.csv에서 발급기관이 "해당 발급기관"/"학교 또는 교육기관"처럼
# 서류마다 실제로는 제각각인 경우, portal_url/direct_url이 특정 발급처가 아니라
# LH청약플러스 메인 페이지(범용 플레이스홀더)로 채워져 있습니다. 이 URL은 클릭해도
# 해당 서류와 무관한 LH 홈페이지만 뜨므로, 실제 발급처 링크가 아닌 것으로 취급합니다.
_GENERIC_DOCUMENT_PORTAL_URLS = {"https://apply.lh.or.kr", "https://apply.lh.or.kr/"}


def _has_real_portal_link(url: str) -> bool:
    return bool(url) and url not in _GENERIC_DOCUMENT_PORTAL_URLS


def _short_agency_label(issuer: str) -> str:
    """HTML 시안의 "${agency.split(' ')[0]} 바로가기" 표기를 실제 발급기관
    문자열에 맞춰 재현합니다 (예: "정부24·행정복지센터" -> "정부24")."""
    issuer = (issuer or "").strip()
    if not issuer:
        return "발급기관"
    for sep in ("·", " "):
        if sep in issuer:
            return issuer.split(sep)[0]
    return issuer if len(issuer) <= 8 else issuer[:8]


def _owl_link_note_text(document: Mapping[str, Any]) -> str:
    """정확한 발급 사이트 링크를 아직 확보하지 못했을 때, 빈 화면 대신
    부엉이 박사가 "연구 중"이라고 알려주는 안내 문장을 만듭니다
    (HTML 이스케이프 완료)."""
    keyword = str(document.get("site_search_keyword") or document.get("canonical_name") or "").strip()
    note = "정확한 발급 사이트를 아직 열심히 연구 중이에요! 🔬"
    if keyword:
        note += f' 그동안은 "{escape(keyword)}"로 검색하거나 발급기관 홈페이지·행정실에서 직접 확인해 주세요.'
    return note


def _owl_empty_card_html(document: Mapping[str, Any]) -> str:
    """부엉이 박사 캐릭터 + "연구 중" 안내문을 하나의 둥근 사각형 카드로
    묶은 HTML 조각. 고정 오버레이(드로어)와 일반 인라인 배치 양쪽에서
    재사용합니다."""
    note = _owl_link_note_text(document)
    return f"""
        <div class="owl-empty-card">
            <img src="https://i.ibb.co/ksxnBZm8/owl-doctor.png" alt="부엉이 박사" class="owl-empty-avatar"
                 onerror="this.onerror=null;this.src='data:image/svg+xml;charset=UTF-8,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22%3E%3Ccircle cx=%2250%22 cy=%2250%22 r=%2248%22 fill=%22%23F0CA5E%22/%3E%3Ctext x=%2250%22 y=%2265%22 font-size=%2248%22 text-anchor=%22middle%22%3E%F0%9F%A6%89%3C/text%3E%3C/svg%3E';">
            <div class="owl-empty-label">🦉 부엉이 박사 연구 중</div>
            <div class="owl-empty-text">{note}</div>
        </div>
        """


def render_owl_empty_note(document: Mapping[str, Any]) -> None:
    """확정된 발급 사이트가 없는 서류를 열었을 때는 팝업 창이 뜨지 않아
    화면 우측(드로어 옆 빈 공간)이 비어 보입니다. 그 자리 정중앙에 부엉이
    박사 카드를 고정 배치합니다."""
    html_block(f'<div class="owl-empty-note">{_owl_empty_card_html(document)}</div>')


def render_popup_bridge(url_map: Mapping[str, str]) -> None:
    """문서 카드의 "OO 바로가기" 버튼을 누른 그 클릭에서 곧바로 발급
    사이트를 화면 우측 팝업으로 엽니다.

    Streamlit 버튼은 서버 왕복(rerun) 후에야 반응하므로, 그 시점에
    window.open을 부르면 사용자 제스처 컨텍스트가 끊겨 팝업이 차단되기
    쉽습니다. 대신 이 컴포넌트가 부모 문서(앱 페이지) 자체에 클릭 캡처
    리스너를 설치해 두고, Streamlit 버튼이 반응하는 바로 그 DOM 클릭
    이벤트에서 동기적으로 window.open을 호출합니다. 매 rerun마다 새로
    설치되므로 이전 리스너는 먼저 제거해 중복 등록을 막습니다.

    브라우저는 웹페이지가 팝업 창을 시스템 차원에서 "항상 위"로 고정하는
    것을 허용하지 않습니다(악성 팝업 남용 방지). 대신 체크리스트 항목을
    클릭할 때마다 열려 있는 발급 사이트 창을 focus()로 다시 앞으로
    가져와, 체크할 때만이라도 가려지지 않게 합니다. (Streamlit의 체크박스는
    클릭을 React가 가로채 .checked를 코드로 설정하므로 네이티브 "change"
    이벤트가 발생하지 않는다 — 그래서 doc-grid 버튼과 같은 캡처 단계
    "click" 리스너로 감지한다.)
    """
    payload = json.dumps(dict(url_map))
    components.html(
        f"""
        <script>
        (function () {{
            var doc = window.parent.document;
            var urlMap = {payload};
            var popupWidth = 860;
            var popupHeight = 840;
            function openSidePopup(url) {{
                var left = Math.max(window.screen.width - popupWidth - 60, 40);
                var win = window.open(
                    url, "_blank",
                    "width=" + popupWidth + ",height=" + popupHeight +
                    ",left=" + left + ",top=40,scrollbars=yes"
                );
                if (win) {{ window.parent.__docPopupWindowRef = win; }}
            }}
            function refocusPopup() {{
                var ref = window.parent.__docPopupWindowRef;
                if (ref && !ref.closed) {{
                    setTimeout(function () {{ ref.focus(); }}, 80);
                }}
            }}
            function clickHandler(e) {{
                var el = e.target;
                while (el && el !== doc.body) {{
                    var cls = el.className;
                    if (typeof cls === "string") {{
                        var m = cls.match(/st-key-(doc_card_[A-Za-z0-9_]+)/);
                        if (m && urlMap[m[1]]) {{
                            openSidePopup(urlMap[m[1]]);
                            return;
                        }}
                    }}
                    if (el.getAttribute && el.getAttribute("data-testid") === "stCheckbox"
                        && el.closest('[class*="st-key-guide_drawer_panel"]')) {{
                        refocusPopup();
                        return;
                    }}
                    el = el.parentElement;
                }}
            }}
            if (window.parent.__docPopupBridgeClickHandler) {{
                doc.removeEventListener("click", window.parent.__docPopupBridgeClickHandler, true);
            }}
            window.parent.__docPopupBridgeClickHandler = clickHandler;
            doc.addEventListener("click", clickHandler, true);
        }})();
        </script>
        """,
        height=0,
    )


def render_document_guide_drawer() -> None:
    """HTML 시안의 좌측 슬라이드 드로어처럼, 결과 화면을 벗어나지 않고
    선택한 서류의 발급 체크리스트를 화면 왼쪽에 고정 오버레이로 띄웁니다."""
    drawer = st.session_state.get("active_drawer") or {}
    notice_id = str(drawer.get("notice_id") or "")
    document_id = str(drawer.get("document_id") or "")
    if not notice_id or not document_id:
        return

    profile_data = st.session_state.get("user_profile_data", {})
    planner = get_service_application_planner()
    try:
        plan = planner.build_plan(
            notice_id=notice_id,
            record_id=str(drawer.get("record_id") or ""),
            user_profile=profile_data,
            condition_answers=None,
        )
    except Exception:
        st.session_state.active_drawer = {}
        return

    document = next(
        (
            d for d in plan["document_plan"]["documents"]
            if str(d["document_id"]) == document_id
        ),
        None,
    )

    html_block('<div class="drawer-overlay"></div>')

    # owl-empty-note는 position:fixed로 뷰포트 기준 배치되어야 하는데,
    # guide_drawer_panel 컨테이너가 overflow-y:auto라 그 안에서 렌더링하면
    # 패널 폭(400px) 밖으로는 잘려 보이지 않습니다. 그래서 패널 밖(함수
    # 최상위)에서 따로 그리도록 플래그만 컨테이너 안에서 세웁니다.
    needs_owl_empty_note = False

    with st.container(key="guide_drawer_panel"):
        with st.container(key="drawer_back_btn"):
            if st.button("⬅️ 다른 서류 선택으로 돌아가기", key="close_guide_drawer"):
                st.session_state.active_drawer = {}
                st.rerun()

        if document is None:
            html_block(
                f"""
                <div class="drawer-title-area">
                    <h3>{escape(str(drawer.get("document_name") or "서류 발급 가이드"))}</h3>
                </div>
                """
            )
            st.warning("서류 정보를 찾을 수 없습니다.")
            return

        html_block(
            f"""
            <div class="drawer-title-area">
                <h3>{escape(str(document.get("canonical_name") or drawer.get("document_name") or "서류 발급 가이드"))}</h3>
                <p>온라인 발급 시 확인할 필수 체크리스트</p>
            </div>
            """
        )

        portal_url = str(document.get("direct_url") or document.get("portal_url") or "")
        if _has_real_portal_link(portal_url):
            render_side_popup_button(portal_url, "🔗 발급 사이트 다시 열기")
        else:
            needs_owl_empty_note = True

        checklist_items = [
            (label, value)
            for label, value in (
                ("온라인 발급 방법", document.get("issue_method_online")),
                ("오프라인 발급 방법", document.get("issue_method_offline")),
                ("본인인증", document.get("identity_required")),
                ("수수료", document.get("fee_note")),
                ("처리시간", document.get("processing_time")),
                ("발급 옵션", document.get("required_options")),
                ("출력 형식", document.get("output_format")),
                ("유효기간 주의", document.get("validity_note")),
                ("제출 주의", document.get("cautions")),
                ("개인정보 주의", document.get("privacy_warning")),
            )
            if value
        ]
        if checklist_items:
            html_block('<div class="check-list-title">✅ 발급 체크리스트</div>')
            for label, value in checklist_items:
                st.checkbox(
                    f"**{label}**: {value}",
                    key=f"drawer_checklist_{notice_id}_{document_id}_{label}",
                )

        done_key = f"document_done_{notice_id}_{document_id}"
        is_done = bool(st.session_state.get(done_key, False))
        with st.container(key="drawer_complete_btn"):
            if st.button(
                "✅ 발급 완료됨 (다시 열기)" if is_done else "서류 발급 완료하기 🌟",
                key=f"drawer_complete_{notice_id}_{document_id}",
                use_container_width=True,
            ):
                st.session_state[done_key] = not is_done
                if not is_done:
                    # 완료로 전환될 때만 토스트를 띄우고 사이드바를 닫아
                    # 결과 화면으로 돌아갑니다. 완료 취소는 드로어를 열어둔
                    # 채로 바로 되돌릴 수 있게 둡니다.
                    st.toast(
                        f"{document.get('canonical_name', '서류')} 발급 상태가 완료로 업데이트되었습니다! 🌟"
                    )
                    st.session_state.active_drawer = {}
                st.rerun()

    if needs_owl_empty_note:
        render_owl_empty_note(document)


def render_success_modal() -> None:
    """HTML 시안의 #successModalOverlay를 재현합니다.

    시안에서는 "모든 서류 준비 완료" 버튼을 눌러도 바로 이동하지 않고
    축하 모달(컨페티 포함)이 먼저 뜬 뒤, 그 안의 "실제 공고 신청하러 가기"
    버튼을 눌러야 실제로 새 탭이 열립니다. st.link_button은 즉시 이동해
    버려 이 2단계 흐름을 재현할 수 없으므로, 먼저 모달을 띄우는 일반
    버튼으로 갈아끼우고 이 함수가 모달과 실제 이동 버튼을 그립니다.
    """
    modal = st.session_state.get("success_modal") or {}
    link = str(modal.get("link") or "")
    if not link:
        return

    confetti_colors = ["#F0CA5E", "#6B705C", "#3182F6", "#E05252", "#4A503D"]
    confetti_html = "".join(
        f'<span class="confetti-piece" style="left:{(i * 37) % 100}%;'
        f'background:{confetti_colors[i % len(confetti_colors)]};'
        f'animation-delay:{(i % 6) * 0.12}s;"></span>'
        for i in range(18)
    )

    html_block('<div class="modal-overlay-bg"></div>')
    with st.container(key="success_modal_panel"):
        html_block(
            f"""
            <div class="modal-confetti">{confetti_html}</div>
            <h2>축하합니다! 🎉</h2>
            <p>부엉이 지기와 함께 모든 서류를 완벽히 준비했어요.<br>
            이제 주거 독립을 향해 당당하게 신청 버튼을 눌러보세요! 🦉</p>
            """
        )
        st.link_button(
            "실제 공고 신청하러 가기",
            link,
            type="primary",
            use_container_width=True,
        )
        if st.button("닫기", key="close_success_modal", use_container_width=True):
            st.session_state.success_modal = {}
            st.rerun()


# HTML 시안의 완벽한 CSS 연동 (Streamlit Shadow DOM 회피 및 덮어쓰기)
html_block(
    """
    <style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.css');

    :root {
        --bezier-smooth: cubic-bezier(0.16, 1, 0.3, 1);
        --bg-workspace: #F7F8FA;
        --bg-left: #FFFFFF;
        --main-olive: #6B705C;
        --dark-olive: #4A503D;
        --light-olive: #A4A999;
        --pastel-olive: #F4F5EF;
        --point-yellow: #F0CA5E;
        --point-blue: #4A90E2;
        --error-red: #E05252;
        --text-main: #191F28;
        --text-desc: #6B7684;
        --text-sub: #8B95A1;
        --white: #FFFFFF;
        
        --shadow-base: 0 12px 30px rgba(0, 0, 0, 0.03);
        --shadow-card: 0 4px 16px rgba(0,0,0,0.04);
        --shadow-hover: 0 20px 40px rgba(0, 0, 0, 0.08);
        --shadow-tooltip: 0 8px 24px rgba(0, 0, 0, 0.15);
    }

    html, body, [class*="css"] {
        font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
    }

    /* Streamlit Global Adjustments */
    .stApp { background: var(--bg-workspace); color: var(--text-main); }
    #MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }

    /* Layout: Left 40% Panel override.
       Earlier this trapped .stApp at height:100vh/overflow:hidden and made
       each column scroll independently via its own overflow-y:auto. That
       depends on wheel/trackpad input reliably targeting whichever column
       sits under the cursor, which failed intermittently in practice
       (content below the fold became unreachable). Streamlit already
       provides one shared, reliable page scroll (stMain's own
       overflow:auto), so we no longer fight it — this file only sets the
       visual 40:60 split (colors/padding/min-height), not scroll behavior.

       This used to hook an empty `<div class="left-panel-marker">` injected
       via st.markdown(unsafe_allow_html=True) and select the surrounding
       stColumn/stHorizontalBlock with :has(). On the result page (long,
       multi-card content) that injected marker stopped surviving in the
       DOM, so the :has() rules silently no longer matched and the columns
       fell back to Streamlit's default flex-basis, which wraps onto
       separate lines once the left panel's 440px min-width no longer fits
       (see the comment on stColumn:has(...right_panel_body) below). Using
       st.container(key=...) instead gives each panel a stable st-key-*
       class straight from the framework's own DOM node, so it can be
       targeted directly rather than through an injected marker. */
    .block-container:has(.st-key-left_panel_body) { width: 100%; max-width: none; padding: 0; }

    div[data-testid="stHorizontalBlock"]:has(.st-key-left_panel_body) {
        min-height: 100vh; gap: 0 !important; align-items: stretch; flex-wrap: nowrap !important;
    }

    /* Streamlit >=1.5x renders columns as data-testid="stColumn" (was "column") */
    div[data-testid="stColumn"]:has(.st-key-left_panel_body) {
        min-width: 440px; max-width: 500px; background: var(--bg-left);
        z-index: 10; box-shadow: 8px 0 30px rgba(0,0,0,.03);
    }

    div[data-testid="stColumn"]:has(.st-key-right_panel_body) {
        /* Fill remaining space rather than the calc(60% - gap) flex-basis
           Streamlit assigns columns by default — with the left panel's
           440px min-width that basis overflows the row and wraps the two
           columns onto separate lines instead of sitting side by side. */
        flex: 1 1 auto !important; width: auto !important; min-width: 0 !important;
        background-color: var(--bg-workspace);
        background-image:
            radial-gradient(circle at 80% 20%, rgba(240,202,94,.10) 0%, transparent 50%),
            radial-gradient(circle at 20% 80%, rgba(107,112,92,.08) 0%, transparent 60%);
    }

    /* The keyed container *is* the panel's content block now, so style it
       directly instead of reaching for a child stVerticalBlock. No
       height/overflow here — content sizes naturally and the page scrolls
       as a whole. */
    .st-key-left_panel_body {
        min-height: 100vh; padding: 48px 40px; background: var(--bg-left);
    }

    .st-key-right_panel_body {
        min-height: 100vh; padding: 60px; background: transparent;
    }

    /* Left Panel Typography & Brand */
    .brand-header { margin-bottom: 40px; }
    .brand-header .subtitle {
        display: inline-block; margin-bottom: 8px; color: var(--main-olive);
        font-size: 13px; font-weight: 800; letter-spacing: .3px;
    }
    .brand-header h1 { margin: 0 0 12px; color: var(--text-main); font-size: 28px; font-weight: 800; }
    .brand-header p { margin: 0; color: var(--text-desc); font-size: 15px; font-weight: 500; line-height: 1.6; }
    .form-step-label { margin: 28px 0 10px; color: var(--text-main); font-size: 15px; font-weight: 700; }

    /* Streamlit Custom Inputs matching .input-base
       Streamlit >=1.5x dropped BaseWeb selects/textareas for a React-Aria
       ComboBox/TextField; there is no more data-baseweb="select"/"textarea"
       attribute, so each widget is targeted via its own data-testid box. */
    div[data-testid="stSelectbox"] .react-aria-ComboBox > div[role="group"],
    div[data-testid="stNumberInputContainer"],
    div[data-testid="stTextAreaRootElement"] {
        min-height: 52px; border: 1.5px solid transparent !important;
        border-radius: 14px !important; background: #F4F5F6 !important;
        box-shadow: none !important; transition: all 0.3s var(--bezier-smooth);
    }

    div[data-testid="stSelectbox"] .react-aria-ComboBox > div[role="group"]:focus-within,
    div[data-testid="stNumberInputContainer"]:focus-within,
    div[data-testid="stTextAreaRootElement"]:focus-within {
        border-color: var(--main-olive) !important; background: var(--white) !important;
        box-shadow: 0 0 0 4px rgba(107,112,92,.10) !important;
    }

    div[data-testid="stSelectbox"] input[role="combobox"],
    div[data-testid="stNumberInputField"],
    div[data-testid="stTextArea"] textarea {
        background: transparent !important; border: none !important; box-shadow: none !important;
        color: var(--text-main) !important; font-size: 15px; font-weight: 600;
    }
    div[data-testid="stSelectbox"] .react-aria-ComboBox button svg { color: #8B95A1; }
    button[data-testid="stNumberInputStepDown"], button[data-testid="stNumberInputStepUp"] {
        background: transparent !important; border: none !important; color: #ADB5BD !important;
    }
    div[data-testid="stTextArea"] textarea { min-height: 90px; padding: 16px !important; font-weight: 500; }

    /* Radio pill group. Scoped to stRadioGroup / stRadioOption specifically —
       targeting a bare "label" under stRadio also matches the widget's own
       (collapsed) stWidgetLabel and forces it visible, duplicating the
       field's caption above the pills. */
    div[data-testid="stRadioGroup"] { gap: 8px; background: #F4F5F6; border-radius: 14px; padding: 4px; }
    div[data-testid="stRadioGroup"] label[data-testid="stRadioOption"] {
        margin: 0; padding: 9px 12px; border-radius: 10px; font-size: 14.5px; font-weight: 600;
        color: var(--text-desc); cursor: pointer; transition: all 0.3s var(--bezier-smooth);
        display: flex; align-items: center; justify-content: center; height: 44px;
    }
    div[data-testid="stRadioGroup"] label[data-testid="stRadioOption"]:has(input:checked) {
        background-color: var(--white); color: var(--text-main); box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    
    /* Primary CTA buttons (form submit, "apply now" link). st.button and
       st.link_button render under different data-testid families
       (stBaseButton-* vs stBaseLinkButton-*), so both must be targeted. */
    button[data-testid="stBaseButton-primary"], a[data-testid="stBaseLinkButton-primary"] {
        min-height: 56px; border: none !important; border-radius: 14px !important;
        background: var(--main-olive) !important; color: var(--white) !important;
        font-size: 16px !important; font-weight: 700 !important; box-shadow: none !important;
        transition: transform .3s var(--bezier-smooth), background .3s var(--bezier-smooth), box-shadow .3s;
    }
    /* Every st.button/st.link_button wraps its label in nested markdown
       elements (p/div/span) that carry Streamlit's own default text color
       AND font-weight — both win over whatever's set on the outer
       button/<a> since they're a direct match on the element rather than
       inherited. Force color + font-weight on every descendant so the
       label is both legible and consistently bold everywhere. */
    button[data-testid="stBaseButton-primary"] *, a[data-testid="stBaseLinkButton-primary"],
    a[data-testid="stBaseLinkButton-primary"] * {
        color: var(--white) !important; font-weight: 700 !important;
    }
    button[data-testid="stBaseButton-primary"]:hover, a[data-testid="stBaseLinkButton-primary"]:hover {
        background: var(--dark-olive) !important; transform: translateY(-2px); box-shadow: 0 8px 16px rgba(107, 112, 92, 0.2) !important;
    }
    button[data-testid="stBaseButton-primary"]:active { transform: scale(.98); }

    /* Secondary / outline buttons — matches the HTML mock's .doc-link-btn
       and .official-link-btn (per-document guide buttons, back links,
       "공식 신청 페이지 보러가기"): white with a border by default, olive fill
       on hover, smaller than the primary CTA. */
    button[data-testid="stBaseButton-secondary"], a[data-testid="stBaseLinkButton-secondary"] {
        min-height: 44px; border: 1.5px solid #D1D6DB !important; border-radius: 12px !important;
        background: var(--white) !important; color: var(--text-main) !important;
        font-size: 14px !important; font-weight: 700 !important; box-shadow: none !important;
        transition: all 0.2s var(--bezier-smooth);
    }
    button[data-testid="stBaseButton-secondary"] *, a[data-testid="stBaseLinkButton-secondary"],
    a[data-testid="stBaseLinkButton-secondary"] * {
        color: var(--text-main) !important; font-weight: 700 !important;
    }
    button[data-testid="stBaseButton-secondary"]:hover, a[data-testid="stBaseLinkButton-secondary"]:hover {
        background: var(--main-olive) !important; border-color: var(--main-olive) !important;
    }
    button[data-testid="stBaseButton-secondary"]:hover *, a[data-testid="stBaseLinkButton-secondary"]:hover,
    a[data-testid="stBaseLinkButton-secondary"]:hover * {
        color: var(--white) !important;
    }

    /* Native checkbox/radio tick color follows the Streamlit theme's
       accent (default red) unless overridden explicitly. */
    input[type="checkbox"], input[type="radio"] { accent-color: var(--main-olive); }

    /* Tooltip styling */
    .tooltip-trigger {
        font-size: inherit !important; font-weight: 700 !important; color: inherit;
        border-bottom: 2px dotted var(--point-blue) !important; cursor: help !important;
        position: relative; display: inline-block; transition: color 0.3s ease;
    }
    .tooltip-trigger:hover { color: var(--point-blue); }
    .tooltip-content {
        visibility: hidden; opacity: 0; position: absolute; bottom: 125%; left: 50%;
        transform: translateX(-50%) translateY(10px); background-color: var(--text-main); color: var(--white);
        padding: 8px 12px !important; border-radius: 8px; font-size: 12px !important; font-weight: 500; line-height: 1.4 !important;
        width: max-content; max-width: 240px; z-index: 9999; transition: all 0.3s var(--bezier-smooth);
        box-shadow: var(--shadow-tooltip); pointer-events: none; text-align: left; word-break: keep-all;
    }
    .tooltip-content::after {
        content: ''; position: absolute; top: 100%; left: 50%; transform: translateX(-50%);
        border-width: 6px; border-style: solid; border-color: var(--text-main) transparent transparent transparent;
    }
    .tooltip-trigger:hover .tooltip-content { visibility: visible; opacity: 1; transform: translateX(-50%) translateY(0); }

    /* Right Panel Animations */
    .fade-in-up { animation: fadeInUpAnim 0.5s var(--bezier-smooth) both; }
    @keyframes fadeInUpAnim { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }

    /* Welcome / Waiting State */
    .welcome-container { text-align: center; width: 100%; display: flex; flex-direction: column; align-items: center;}
    .mascot-area-new { display: flex; flex-direction: column; align-items: center; gap: 8px; margin-bottom: 24px; }
    .mascot-img-large, .loading-mascot {
        width: 180px; height: 180px; object-fit: contain; filter: drop-shadow(0 12px 24px rgba(107, 112, 92, 0.15));
        animation: floating 3s ease-in-out infinite;
    }
    @keyframes floating { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-10px); } }
    
    .mascot-label {
        background-color: var(--white); color: var(--dark-olive); padding: 6px 16px;
        border-radius: 20px; font-size: 13px; font-weight: 700; box-shadow: 0 4px 12px rgba(0,0,0,0.04);
    }
    .slogan { font-size: 20px; font-weight: 800; color: var(--text-main); margin: 0 0 40px !important; word-break: keep-all; line-height: 1.5; text-align: center; }
    
    .speech-bubble {
        background: var(--white); padding: 18px 24px; border-radius: 20px; font-size: 14.5px; line-height: 1.6; color: var(--text-desc); 
        word-break: keep-all; text-align: center; box-shadow: var(--shadow-card); width: 100%; margin-bottom: 24px; max-width: 640px;
    }
    .speech-bubble span.highlight { font-weight: 800; color: var(--text-main); }
    
    /* Strength Cards */
    .strength-list { display: flex; flex-direction: column; gap: 12px; width: 100%; max-width: 640px; margin: auto; }
    .strength-card { display: flex; align-items: center; gap: 16px; padding: 18px 24px; border-radius: 16px; background-color: var(--white); box-shadow: var(--shadow-card); }
    .strength-icon {
        width: 44px; height: 44px; background-color: var(--pastel-olive); color: var(--main-olive);
        border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 22px; flex-shrink: 0;
    }
    .strength-info { text-align: left; }
    .strength-info h3 { font-size: 15px; font-weight: 800; color: var(--text-main); margin: 0 0 4px; }
    .strength-info p { font-size: 13.5px; color: var(--text-desc); line-height: 1.4; margin: 0;}
    .trust-badge { font-size: 13px; color: var(--text-sub); text-align: center; margin-top: 24px; line-height: 1.5; font-weight: 500; }

    /* Loading Card */
    .loading-card {
        max-width: 620px; margin: 11vh auto 0; padding: 44px 34px;
        border-radius: 28px; background: rgba(255,255,255,.92);
        box-shadow: var(--shadow-base); text-align: center;
        animation: fadeInUpAnim .4s var(--bezier-smooth) both;
    }
    .loading-title { font-size: 22px; font-weight: 800; color: var(--text-main); margin-bottom: 8px; }
    .loading-dots::after { content: '.'; animation: dots 1.5s steps(5, end) infinite; }
    @keyframes dots {
        0%, 20% { color: rgba(0,0,0,0); text-shadow: .25em 0 0 rgba(0,0,0,0), .5em 0 0 rgba(0,0,0,0);}
        40% { color: inherit; text-shadow: .25em 0 0 rgba(0,0,0,0), .5em 0 0 rgba(0,0,0,0);}
        60% { text-shadow: .25em 0 0 inherit, .5em 0 0 rgba(0,0,0,0);}
        80%, 100% { text-shadow: .25em 0 0 inherit, .5em 0 0 inherit;}
    }
    .loading-desc { font-size: 15px; font-weight: 600; color: var(--main-olive); margin-bottom: 32px; }
    .housing-tip-box {
        background-color: var(--white); padding: 20px 28px; border-radius: 18px; box-shadow: var(--shadow-base);
        color: var(--dark-olive); font-weight: 500; font-size: 14.5px; line-height: 1.5; margin: 0 auto;
        text-align: left; border-left: 4px solid var(--point-yellow);
    }
    .housing-tip-box strong { color: var(--main-olive); font-size: 15px; display: block; margin-bottom: 6px; }

    /* Result Panel & Cards */
    .result-header { margin-bottom: 32px; text-align: center; }
    .result-header h2 { font-size: 26px; font-weight: 800; color: var(--text-main); margin: 0 0 8px; }
    .result-header p { font-size: 15px; color: var(--text-desc); line-height: 1.5; margin: 0; }
    
    .report-policy-card {
        background-color: var(--white); border-radius: 28px; padding: 36px;
        box-shadow: var(--shadow-base); border: 2px solid transparent !important;
        transition: all 0.3s var(--bezier-smooth); position: relative; margin-top: 14px; margin-bottom: 24px;
    }
    .report-policy-card:hover { border-color: rgba(107, 112, 92, 0.2) !important; transform: translateY(-4px); box-shadow: var(--shadow-hover); }

    .rank-badge {
        position: absolute; top: -16px; left: 36px; background-color: var(--text-main); color: var(--white);
        padding: 8px 16px; border-radius: 12px; font-size: 14px; font-weight: 800; box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    .rank-badge.top1 { background-color: var(--point-yellow); color: #4A3E00; font-size: 15px;}
    
    .card-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 24px; margin-top: 8px; gap: 20px;}
    .match-title { font-size: 24px; font-weight: 800; line-height: 1.4; margin: 0 0 12px; word-break: keep-all; }
    .match-title a { color: #2C3E50 !important; text-decoration: none !important; position: relative; display: inline-block;}
    .match-title a::after {
        content: ''; position: absolute; width: 100%; transform: scaleX(0); height: 2px; bottom: -2px; left: 0;
        background-color: #2C3E50; transform-origin: bottom right; transition: transform 0.3s var(--bezier-smooth);
    }
    .match-title a:hover::after { transform: scaleX(1); transform-origin: bottom left; }
    .match-meta { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
    .meta-tag { background-color: #F8F9FA; border: 1px solid #E9ECEF; padding: 6px 12px; border-radius: 8px; font-size: 13px; font-weight: 600; color: var(--text-desc); }
    .meta-tag.highlight { color: #3182F6; border-color: #3182F6; background: #fff; }
    
    .score-chart {
        width: 76px; height: 76px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
        background: conic-gradient(var(--main-olive) var(--score-deg), #E9ECEF 0); 
        position: relative; flex-shrink: 0; box-shadow: inset 0 2px 4px rgba(0,0,0,0.05);
    }
    .score-inner {
        width: 62px; height: 62px; background-color: var(--white); border-radius: 50%;
        display: flex; align-items: center; justify-content: center; font-size: 20px; font-weight: 800; color: #2C3E50;
    }
    .score-inner span { font-size: 12px; margin-left: 2px; margin-top: 4px;}
    
    .card-body { display: flex; flex-direction: column; gap: 16px; margin-bottom: 24px; }
    .info-section { background-color: rgba(107, 112, 92, 0.04); border-radius: 16px; padding: 20px 24px; }
    .reason-section { background-color: rgba(240, 202, 94, 0.15); }
    .check-section { background-color: #F8F9FA; border: 1px solid #E9ECEF; }
    .info-title { font-size: 15px; font-weight: 800; color: var(--dark-olive); margin-bottom: 8px; }
    .info-desc { font-size: 14.5px; color: var(--text-main); line-height: 1.6; font-weight: 500; word-break: keep-all; }
    .check-points { display: flex; flex-direction: column; gap: 8px; }
    .check-point-item { display: flex; gap: 8px; font-size: 14.5px; font-weight: 500; color: var(--text-main); line-height: 1.5; align-items: flex-start;}
    .check-point-item::before { content: '✔️'; color: var(--main-olive); font-size: 12px; margin-top: 2px;}
    
    .caution-box {
        margin-top: 14px; padding: 15px 18px; border: 1px solid rgba(240,202,94,.42);
        border-radius: 16px; background: rgba(240,202,94,.13); color: #6D571A; font-size: 14px; line-height: 1.65;
    }

    .ai-income-estimator {
        margin-top: 14px; padding: 20px; background-color: var(--pastel-olive); border-radius: 14px;
        animation: fadeInUpAnim 0.3s var(--bezier-smooth) both;
    }
    .ai-income-desc { font-size: 14.5px; font-weight: 700; color: var(--dark-olive); margin-bottom: 10px; line-height: 1.5; }

    /* Required-document grid preview inside each result card — values
       lifted 1:1 from the HTML mock's .doc-section-title/.doc-grid/
       .doc-card/.doc-header/.doc-name/.doc-agency/.doc-guide. The card is
       split top/bottom across two DOM nodes only because the button has
       to be a real st.button (see the merge-radius note below); every
       other value (padding, radii, font sizes, colors) matches the mock. */
    .doc-section-title { font-size: 16px; font-weight: 800; color: var(--text-main); margin: 20px 0 12px; }
    .doc-card {
        background: #F8F9FA; border-radius: 16px 16px 0 0; padding: 20px 20px 16px;
        border: 1px solid #E9ECEF; border-bottom: none;
    }
    .doc-header { margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center; }
    .doc-name { font-size: 16px; font-weight: 800; color: var(--text-main); }
    .doc-agency {
        font-size: 11px; font-weight: 700; background-color: var(--white); color: var(--text-sub);
        padding: 4px 8px; border-radius: 6px; border: 1px solid #E9ECEF; white-space: nowrap;
    }
    .doc-guide { font-size: 13px; color: var(--text-desc); line-height: 1.5; word-break: keep-all; }
    /* Each doc preview is st.container(key=f"doc_card_...") wrapping the
       .doc-card markup above and its real st.button — targeting the key's
       class (rather than :has() over injected markdown content, which
       proved unreliable on long pages — see the left/right panel note
       above) so the info block and button read as one continuous card. */
    div[class*="st-key-doc_card_"] button[data-testid="stBaseButton-secondary"] {
        border-radius: 0 0 16px 16px !important; border-top: none !important; margin-top: -1px;
        width: 100%;
    }
    /* Completed document — matches the HTML mock's .doc-link-btn.completed
       (greyed, non-interactive) once the drawer marks it done. */
    .doc-card.completed { opacity: .7; }
    div[class*="st-key-doc_card_"] button:disabled {
        background-color: #E9ECEF !important; color: var(--text-sub) !important;
        border-color: transparent !important; box-shadow: none !important; opacity: 1 !important;
        cursor: default !important;
    }

    /* success-cta: matches the HTML mock's gold "모든 서류 준비 완료" CTA,
       shown once every document preview in a card is marked done. This is
       a real st.button (not st.link_button, since it opens the
       congratulations modal instead of navigating directly) so the
       selector targets <button>, not <a>. */
    div[class*="st-key-success_cta_"] button {
        background: linear-gradient(135deg, var(--point-yellow), #FFD700) !important;
        border: none !important; font-weight: 800 !important;
        box-shadow: 0 8px 24px rgba(240,202,94,.4) !important; min-height: 60px !important;
    }
    /* Only `color` cascades into the label's nested wrappers — background/
       border/shadow must stay on the outer <button> only, or every nested
       div paints its own copy underneath (a "box inside a box" ripple). */
    div[class*="st-key-success_cta_"] button, div[class*="st-key-success_cta_"] button * {
        color: #4A3E00 !important; font-weight: 800 !important;
    }
    /* The label's <p> stretches to the button's full height (via the
       flex parent) but is itself block-level, so its one line of text
       sits at the top of that tall box instead of centering inside it. */
    div[class*="st-key-success_cta_"] button p {
        display: flex !important; align-items: center !important; justify-content: center !important;
        height: 100% !important; margin: 0 !important;
    }
    div[class*="st-key-success_cta_"] button:hover {
        transform: translateY(-3px); box-shadow: 0 12px 28px rgba(240,202,94,.55) !important;
    }

    /* official-link-btn: matches the HTML mock's plain outbound "공식 신청
       페이지 보러가기" pill exactly — taller and more subdued than the small
       .doc-link-btn buttons above it, with a barely-there hover instead of
       the olive fill used elsewhere. */
    div[class*="st-key-official_link_"] a {
        min-height: 56px !important; border-radius: 16px !important;
        border: 1.5px solid #D1D6DB !important; background-color: var(--white) !important;
        font-size: 16px !important; font-weight: 800 !important;
    }
    /* Only `color` cascades to the label's nested div/span/p wrappers —
       box-model properties (border/background/radius) must stay on the
       outer <a> only, or every nested wrapper paints its own copy and you
       get a "box inside a box" ripple effect. */
    div[class*="st-key-official_link_"] a, div[class*="st-key-official_link_"] a * {
        color: #2C3E50 !important; font-weight: 800 !important;
    }
    div[class*="st-key-official_link_"] { margin-top: 16px; }
    div[class*="st-key-official_link_"] a:hover {
        border-color: var(--text-main) !important; background-color: #FAFAFB !important;
    }
    div[class*="st-key-official_link_"] a:hover, div[class*="st-key-official_link_"] a:hover * {
        color: var(--text-main) !important;
    }

    /* Left slide-in guide drawer — matches the HTML mock's
       .drawer-overlay + .guide-drawer + .drawer-header/.drawer-body 1:1:
       a fixed-position overlay so the result page (and its 3 policy cards)
       never has to navigate away when a document is opened. */
    .drawer-overlay {
        position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
        background: rgba(0,0,0,.3); z-index: 9000; backdrop-filter: blur(2px);
        animation: drawerOverlayFade .3s var(--bezier-smooth) both;
    }
    @keyframes drawerOverlayFade { from { opacity: 0; } to { opacity: 1; } }

    .st-key-guide_drawer_panel {
        position: fixed !important; top: 0; left: 0; width: 400px; max-width: 92vw;
        height: 100vh; background-color: #FAFAFB; box-shadow: 8px 0 40px rgba(0,0,0,.1);
        z-index: 9999; overflow-y: auto; padding: 24px 32px !important;
        animation: drawerSlideIn .35s var(--bezier-smooth) both;
    }
    @keyframes drawerSlideIn { from { transform: translateX(-100%); } to { transform: translateX(0); } }

    .drawer-title-area { margin: 12px 0 20px; }
    .drawer-title-area h3 { font-size: 22px; font-weight: 800; color: var(--text-main); margin-bottom: 6px; }
    .drawer-title-area p { font-size: 14px; color: var(--text-sub); font-weight: 500; }

    /* 발급 사이트 팝업이 뜨지 않는 서류를 열었을 때, 드로어 패널(왼쪽 400px)
       옆의 빈 오버레이 공간 정중앙에 부엉이 박사와 안내문을 하나의 카드로
       묶어 보여줍니다. */
    .owl-empty-note {
        position: fixed; top: 50%; left: 400px; right: 0; transform: translateY(-50%); z-index: 9500;
        display: flex; justify-content: center; padding: 0 48px;
        animation: drawerOverlayFade .3s var(--bezier-smooth) both;
    }
    @media (max-width: 700px) { .owl-empty-note { display: none; } }
    .owl-empty-card {
        background-color: var(--white); border-radius: 28px; box-shadow: var(--shadow-base);
        border-left: 4px solid var(--point-yellow); padding: 28px 32px 32px; max-width: 420px;
        display: flex; flex-direction: column; align-items: center; text-align: center;
        margin: 10px auto 18px;
    }
    .owl-empty-avatar {
        width: 120px; height: 120px; object-fit: contain; margin-bottom: 12px;
        animation: floating 3s ease-in-out infinite;
    }
    .owl-empty-label { color: var(--main-olive); font-size: 15px; font-weight: 800; margin-bottom: 10px; }
    .owl-empty-text { color: var(--dark-olive); font-weight: 500; font-size: 15px; line-height: 1.6; }
    .owl-empty-text strong { color: var(--main-olive); }

    .st-key-drawer_back_btn button {
        background: transparent !important; border: none !important; box-shadow: none !important;
        font-size: 14px !important; padding: 0 !important; min-height: auto !important; justify-content: flex-start !important;
    }
    .st-key-drawer_back_btn button, .st-key-drawer_back_btn button * {
        color: var(--text-desc) !important; font-weight: 700 !important;
    }
    .st-key-drawer_back_btn button:hover, .st-key-drawer_back_btn button:hover * {
        background: transparent !important; color: var(--text-main) !important;
    }

    .check-list-title { font-size: 15px; font-weight: 800; color: var(--text-main); margin: 4px 0 12px; }

    /* Checklist items rendered as real st.checkbox widgets, restyled to
       match the mock's white .check-item cards with a custom check-box
       square instead of the native browser control. */
    .st-key-guide_drawer_panel div[data-testid="stCheckbox"] {
        padding: 18px 20px; margin-bottom: 12px;
    }
    .st-key-guide_drawer_panel input[type="checkbox"] {
        appearance: none; -webkit-appearance: none; width: 22px; height: 22px;
        border: 2px solid #D1D6DB; border-radius: 6px; background: var(--white) !important;
        position: relative; cursor: pointer; flex-shrink: 0; margin-top: 1px !important;
        transition: all .2s;
    }
    .st-key-guide_drawer_panel input[type="checkbox"]:checked {
        background-color: var(--main-olive) !important; border-color: var(--main-olive) !important;
    }
    .st-key-guide_drawer_panel input[type="checkbox"]:checked::after {
        content: '✔'; color: #fff; font-size: 13px; font-weight: 700;
        position: absolute; top: 50%; left: 50%; transform: translate(-50%, -54%);
    }

    /* btn-complete: gold CTA at the end of the drawer, matches the mock's
       "서류 발급 완료하기" button exactly. */
    .st-key-drawer_complete_btn { margin-top: 12px; margin-bottom: 12px; }
    .st-key-drawer_complete_btn button {
        background: var(--point-yellow) !important; border: none !important;
        box-shadow: 0 4px 12px rgba(240,202,94,.3) !important; min-height: 56px !important;
    }
    .st-key-drawer_complete_btn button, .st-key-drawer_complete_btn button * {
        color: #4A3E00 !important; font-weight: 800 !important;
    }
    .st-key-drawer_complete_btn button:hover {
        filter: brightness(.95); transform: translateY(-2px);
    }

    /* Success modal — matches the HTML mock's #successModalOverlay: a
       centered congratulations dialog with confetti that appears BEFORE
       actually navigating to the official application page. */
    .modal-overlay-bg {
        position: fixed; inset: 0; background: rgba(0,0,0,.5); z-index: 10000;
        animation: drawerOverlayFade .3s var(--bezier-smooth) both;
    }
    .st-key-success_modal_panel {
        position: fixed !important; top: 50%; left: 50%; transform: translate(-50%, -50%);
        width: 480px; max-width: 90vw; max-height: 85vh; overflow-y: auto;
        background: var(--white); border-radius: 24px; padding: 48px !important;
        z-index: 10001; text-align: center; box-shadow: 0 20px 40px rgba(0,0,0,.2);
        animation: modalPopIn .3s var(--bezier-smooth) both;
    }
    @keyframes modalPopIn {
        from { transform: translate(-50%, -50%) scale(.9); opacity: 0; }
        to { transform: translate(-50%, -50%) scale(1); opacity: 1; }
    }
    .st-key-success_modal_panel h2 { font-size: 26px; font-weight: 800; color: var(--text-main); margin-bottom: 16px; }
    .st-key-success_modal_panel p { font-size: 15px; color: var(--text-desc); line-height: 1.6; margin-bottom: 8px; word-break: keep-all; }
    .modal-confetti { position: relative; height: 0; overflow: visible; }
    .confetti-piece {
        position: absolute; top: -20px; width: 8px; height: 8px; opacity: 0;
        animation: confettiFall 1.8s ease-in forwards;
    }
    @keyframes confettiFall {
        0% { transform: translateY(0) rotate(0); opacity: 1; }
        100% { transform: translateY(320px) rotate(360deg); opacity: 0; }
    }

    /* Dual Guide (Service Plan) styles mapping */
    .dual-guide-title { margin-bottom: 6px; color: var(--text-main); font-size: 22px; font-weight: 800; }
    .dual-guide-desc { margin-bottom: 18px; color: var(--text-sub); font-size: 14px; font-weight: 500; line-height: 1.55; }
    .doc-preview-note {
        padding: 20px; border: 1px solid transparent; border-radius: 16px;
        background: var(--white); box-shadow: 0 2px 8px rgba(0,0,0,0.02); transition: border 0.2s;
    }
    .doc-preview-note:hover { border-color: var(--main-olive); }

    /* Document detail "drawer" (guide_col) — matches the HTML mock's
       .guide-drawer checklist: real st.checkbox widgets restyled as
       check-item cards, checked state gets a strike-through like the mock's
       `input:checked ~ .check-text`. */
    .check-list-title { font-size: 15px; font-weight: 800; color: var(--text-main); margin: 22px 0 10px; }
    div[data-testid="stCheckbox"] {
        background: var(--white); padding: 14px 18px; border-radius: 14px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.02); border: 1px solid transparent;
        margin-bottom: 8px; transition: border 0.2s, background 0.2s;
    }
    div[data-testid="stCheckbox"]:hover { border-color: var(--main-olive); }
    div[data-testid="stCheckbox"] div[data-testid="stMarkdownContainer"] p {
        font-size: 14px; font-weight: 600; color: var(--text-main); line-height: 1.5;
    }
    div[data-testid="stCheckbox"]:has(input:checked) { background: var(--pastel-olive); }
    div[data-testid="stCheckbox"]:has(input:checked) div[data-testid="stMarkdownContainer"] p {
        text-decoration: line-through; color: var(--text-sub);
    }

    .success-panel {
        margin: 24px 0; padding: 48px; border-radius: 24px;
        background: linear-gradient(135deg, rgba(240,202,94,.28), #FFFDF4);
        border: 1px solid rgba(240,202,94,.55); text-align: center;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
    }
    .success-panel h3 { margin: 0 0 16px; color: #4A3E00; font-size: 26px; font-weight: 800; }
    .success-panel p { margin: 0 0 32px; color: #6D571A; line-height: 1.6; font-size: 15px;}
    
    .error-card {
        background-color: #FFF5F5; border: 1px solid #FFE3E3; border-radius: 24px;
        padding: 40px; text-align: center; margin-bottom: 40px; box-shadow: var(--shadow-card);
    }
    .error-card h3 { color: var(--error-red); font-size: 20px; font-weight: 800; margin-bottom: 12px; }
    .error-card p { color: #6B7684; font-size: 15px; line-height: 1.6; }

    @media (max-width: 900px) {
        .block-container:has(.st-key-left_panel_body) { padding: 14px; }
        div[data-testid="stHorizontalBlock"]:has(.st-key-left_panel_body) {
            min-height: auto; flex-direction: column; flex-wrap: wrap !important;
        }
        div[data-testid="stColumn"]:has(.st-key-left_panel_body),
        div[data-testid="stColumn"]:has(.st-key-right_panel_body) {
            max-width: none !important; width: 100% !important;
        }
        .st-key-left_panel_body,
        .st-key-right_panel_body {
            min-height: auto; padding: 28px 22px; border-radius: 24px !important;
        }
        .report-policy-card { padding: 27px 22px !important; }
        .card-header { flex-direction: column; }
        .score-chart { align-self: flex-end; margin-top: -62px; }
    }
    </style>
    """
)


def render_policy_card(item: Mapping[str, Any]) -> None:
    checks_html = "".join(
        f'<div class="check-point-item">{escape(str(check))}</div>'
        for check in item["check_items"]
    )

    mock_label = (
        '<span class="meta-tag">개발용 데이터</span>'
        if item["is_mock_data"]
        else '<span class="meta-tag">실제 공고 근거</span>'
    )

    raw_score = float(item.get("match_score") or 0)
    chart_score = max(0, min(100, round(raw_score)))
    match_score = escape(f"{raw_score:g}")
    policy_name = escape(str(item["policy_name"]))
    category = escape(str(item["category"]))
    service_name = escape(str(item.get("service_name", item["policy_name"])))
    notice_id = escape(str(item.get("notice_id", "")))
    notice_region = escape(str(item.get("notice_region", "")))
    notice_score = item.get("notice_match_score")
    
    notice_meta_parts = [f"기반 서비스: {service_name}"]
    if notice_id: notice_meta_parts.append(f"공고 ID: {notice_id}")
    if notice_region: notice_meta_parts.append(f"지역: {notice_region}")
    if notice_score is not None: notice_meta_parts.append(f"공고 연결 점수: {notice_score}")
    notice_meta = escape(" · ".join(notice_meta_parts))
    
    possibility = escape(str(item["possibility"]))
    easy_description = escape(str(item["easy_description"]))
    reason = escape(str(item["reason"]))
    caution = escape(str(item["caution"]))
    link = escape(str(item["link"]), quote=True)

    rank_label = (
        "연구소 추천 1순위 🥇"
        if int(item["rank"]) == 1
        else f"추천 {item['rank']}순위"
    )
    rank_class = "top1" if int(item["rank"]) == 1 else ""

    html_block(
        f"""
        <div class="policy-card report-policy-card fade-in-up">
            <div class="rank-badge {rank_class}">{rank_label}</div>
            <div class="card-header">
                <div class="card-header-left">
                    <h3 class="match-title"><a href="{link}" target="_blank">{policy_name}</a></h3>
                    <div class="match-meta">
                        <span class="meta-tag">{category}</span>
                        <span class="meta-tag highlight">{possibility}</span>
                        {mock_label}
                    </div>
                    <div style="margin-top: 11px; color: var(--text-sub); font-size: 12px; line-height: 1.5;">{notice_meta}</div>
                </div>
                <div class="score-chart" style="--score-deg: {chart_score * 3.6}deg;">
                    <div class="score-inner">{match_score}<span>점</span></div>
                </div>
            </div>

            <div class="card-body">
                <div class="info-section">
                    <div class="info-title">📖 쉬운 설명</div>
                    <div class="info-desc">{easy_description}</div>
                </div>
                <div class="info-section reason-section">
                    <div class="info-title">🏡 추천 이유</div>
                    <div class="info-desc">{reason}</div>
                </div>
                <div class="info-section check-section">
                    <div class="info-title">🦉 이것만 더 확인해주세요</div>
                    <div class="check-points">{checks_html}</div>
                </div>
            </div>

            <div class="caution-box"><b>확인해 주세요.</b> {caution}</div>
        </div>
        """
    )
    
    policy_id = str(item["policy_id"])
    profile_data = st.session_state.get("user_profile_data", {})
    preview_notice, preview_documents = get_document_preview(policy_id, profile_data)

    def _open_service_plan(document_id: str = "") -> None:
        st.session_state.selected_policy_card = dict(item)
        if preview_notice:
            st.session_state.selected_notice_id = str(preview_notice["notice_id"])
            st.session_state.selected_record_id = str(preview_notice["record_id"])
        else:
            st.session_state.selected_notice_id = str(item.get("notice_id") or "")
            st.session_state.selected_record_id = str(item.get("record_id") or "")
        st.session_state.condition_answers = {}
        st.session_state.selected_document_id = document_id
        st.session_state.celebrated_plan = ""
        st.session_state.page = "service_plan"
        st.rerun()

    if preview_documents:
        # UI 시안의 doc-grid를 실제 서류 미리보기로 렌더링
        html_block('<div class="doc-section-title">📑 필요 서류 가이드</div>')
        preview_notice_id = str(preview_notice["notice_id"]) if preview_notice else ""
        preview_record_id = str(preview_notice["record_id"]) if preview_notice else ""
        doc_columns = st.columns(2, gap="small")
        all_done = True
        for index, doc in enumerate(preview_documents):
            document_id = str(doc["document_id"])
            doc_name = escape(str(doc.get("canonical_name") or "서류 확인 필요"))
            doc_agency = escape(str(doc.get("issuer") or "발급기관 확인"))
            doc_desc = escape(
                str(
                    doc.get("site_search_keyword")
                    or doc.get("status_label")
                    or "공고 원문에서 세부 조건을 확인하세요."
                )
            )
            is_done = bool(
                st.session_state.get(f"document_done_{preview_notice_id}_{document_id}", False)
            )
            all_done = all_done and is_done

            card_key = f"doc_card_{policy_id}_{document_id}"
            portal_url = str(doc.get("direct_url") or doc.get("portal_url") or "")
            has_real_link = _has_real_portal_link(portal_url)
            if has_real_link and not is_done:
                st.session_state["_popup_url_map"][card_key] = portal_url

            with doc_columns[index % 2]:
                with st.container(key=card_key):
                    html_block(
                        f"""
                        <div class="doc-card{' completed' if is_done else ''}">
                            <div class="doc-header">
                                <span class="doc-name">{doc_name}</span>
                                <span class="doc-agency">{doc_agency}</span>
                            </div>
                            <div class="doc-guide">{doc_desc}</div>
                        </div>
                        """
                    )
                    if is_done:
                        # 시안의 .doc-link-btn.completed처럼 발급 완료 서류는
                        # 회색 비활성 버튼으로 표시합니다.
                        st.button(
                            "✅ 발급 완료",
                            key=f"doc_preview_btn_{policy_id}_{document_id}",
                            use_container_width=True,
                            disabled=True,
                        )
                    # 클릭 즉시(1) 발급 사이트가 우측 팝업으로 열리고(render_popup_bridge가
                    # 이 컨테이너의 st-key- 클래스로 URL을 찾아 window.open),
                    # (2) 이 버튼의 Streamlit 콜백이 좌측 체크리스트 드로어를 엽니다.
                    elif st.button(
                        (
                            f"{_short_agency_label(str(doc.get('issuer') or ''))} 바로가기 ➔"
                            if has_real_link
                            else "필수서류 체크리스트 열기 ➔"
                        ),
                        key=f"doc_preview_btn_{policy_id}_{document_id}",
                        use_container_width=True,
                    ):
                        st.session_state.active_drawer = {
                            "policy_id": policy_id,
                            "document_id": document_id,
                            "notice_id": preview_notice_id,
                            "record_id": preview_record_id,
                            "document_name": str(doc.get("canonical_name") or ""),
                        }
                        st.rerun()

        if all_done and item.get("link"):
            # 시안의 .success-cta: 클릭하면 바로 이동하지 않고, 시안처럼
            # 축하 모달을 먼저 띄운 뒤 그 안의 버튼에서 실제로 이동합니다.
            with st.container(key=f"success_cta_{policy_id}"):
                if st.button(
                    "🎉 모든 서류 준비 완료! 실제 공고 신청하러 가기",
                    key=f"success_cta_btn_{policy_id}",
                    use_container_width=True,
                ):
                    st.session_state.success_modal = {
                        "policy_name": str(item.get("policy_name") or ""),
                        "link": str(item["link"]),
                    }
                    st.rerun()

        if item.get("link"):
            with st.container(key=f"official_link_{policy_id}"):
                st.link_button(
                    "공식 신청 페이지 보러가기 ➔",
                    str(item["link"]),
                    use_container_width=True,
                )
    else:
        # 연결된 실제 공고 후보가 없으면 기존처럼 공고 선택 화면으로 안내합니다.
        if st.button(
            "📑 이 공고의 필수 서류 듀얼 뷰 열기 ➔",
            key=f"open_plan_{policy_id}",
            use_container_width=True,
            type="primary",
        ):
            _open_service_plan()

    show_evidence = st.toggle(
        "검색 근거 펼치기",
        key=f"evidence_toggle_{policy_id}",
    )

    if show_evidence:
        with st.container(border=True):
            st.caption(f"검색 상태: {item['rag_status']} · 데이터: {item.get('data_mode', '')} · 출처: {item['source_name']}")
            st.caption(f"설명 생성: {item['generation_status']} · {item['llm_provider']} / {item['llm_model']}")
            if item.get("generation_error"):
                st.info(f"LLM 대신 안전 설명을 사용한 이유: {item['generation_error']}")
            
            evidence = item.get("evidence", {})
            if not evidence:
                st.warning("검색된 근거가 없습니다.")
            else:
                for section_type, label in SECTION_LABELS.items():
                    section_items = evidence.get(section_type, [])
                    st.markdown(f"**{label}**")
                    if not section_items:
                        st.write("공식 페이지에서 추가 확인이 필요합니다.")
                        continue
                    for evidence_item in section_items:
                        st.write(f"- {evidence_item['text']}")
                        source_name = evidence_item.get("source_name") or "출처 미입력"
                        source_url = evidence_item.get("source_url")
                        page_text = evidence_item.get("source_page") or "페이지 확인 필요"
                        st.caption(f"출처: {source_name} · {page_text} · 유사도 {evidence_item['similarity']}")
                        if source_url:
                            st.markdown(f"[원문 페이지 열기]({source_url})")


def render_input_page() -> None:
    left_panel, right_panel = st.columns([0.40, 0.60], gap="small")

    with left_panel:
        with st.container(key="left_panel_body"):
            html_block(
                """
                <div class="brand-header">
                    <span class="subtitle">청년 주거 독립 연구소</span>
                    <h1>Zip.지기 🦉</h1>
                    <p>몇 가지 질문만으로<br>나에게 딱 맞는 주거 정책과 서류를 찾아드려요.</p>
                </div>
                """
            )

            html_block('<div class="form-step-label">1. 희망 거주 지역</div>')
            desired_region_col, desired_sigungu_col = st.columns(2, gap="small")
            with desired_region_col:
                region = st.selectbox(
                    "희망 시·도",
                    REGIONS,
                    index=8,
                    key="input_region",
                    label_visibility="collapsed",
                    on_change=set_input_guide,
                    args=("desired_region",),
                )
            with desired_sigungu_col:
                sigungu = st.selectbox(
                    "희망 시·군·구",
                    SIGUNGU_BY_REGION[region],
                    key=f"input_sigungu_{region}",
                    label_visibility="collapsed",
                    on_change=set_input_guide,
                    args=("desired_region",),
                )

            html_block('<div class="form-step-label">2. 주민등록상 주소지</div>')
            registered_region_col, registered_sigungu_col = st.columns(2, gap="small")
            with registered_region_col:
                registered_region = st.selectbox(
                    "등록 시·도",
                    REGIONS,
                    index=8,
                    key="registered_region",
                    label_visibility="collapsed",
                    on_change=set_input_guide,
                    args=("registered_region",),
                )
            with registered_sigungu_col:
                registered_sigungu = st.selectbox(
                    "등록 시·군·구",
                    SIGUNGU_BY_REGION[registered_region],
                    key=f"registered_sigungu_{registered_region}",
                    label_visibility="collapsed",
                    on_change=set_input_guide,
                    args=("registered_region",),
                )

            html_block('<div class="form-step-label">3. 만 나이</div>')
            age = st.number_input(
                "만 나이",
                min_value=19,
                max_value=39,
                value=24,
                step=1,
                key="input_age",
                label_visibility="collapsed",
                on_change=set_input_guide,
                args=("age",),
            )

            html_block('<div class="form-step-label">4. 주택 소유 여부</div>')
            housing_owned = st.radio(
                "4. 주택 소유 여부",
                ["무주택자", "유주택자"],
                horizontal=True,
                key="input_housing_owned",
                on_change=set_input_guide,
                args=("housing",),
                label_visibility="collapsed",
            )

            html_block('<div class="form-step-label">5. 가구 월 평균 소득 (세전)</div>')
            income_type = st.radio(
                "5. 가구 월 평균 소득 (세전)",
                ["직접 입력", "잘 모르겠어요"],
                horizontal=True,
                key="input_income_type",
                on_change=set_input_guide,
                args=("income",),
                label_visibility="collapsed",
            )

            monthly_income = None
            income_context = ""
            if income_type == "직접 입력":
                monthly_income = st.number_input(
                    "월 소득 입력 (단위: 만 원)",
                    min_value=0,
                    max_value=2000,
                    value=300,
                    step=10,
                    key="input_monthly_income",
                    label_visibility="collapsed",
                    on_change=set_input_guide,
                    args=("income",),
                )
            else:
                # "잘 모르겠어요" 선택 시 AI 추정기 UI
                html_block(
                    '<div class="ai-income-estimator">'
                    '<p class="ai-income-desc">💡 소득을 정확히 모르시는군요!<br>부모님이 하시는 일을 편하게 적어주시면 AI가 대략적으로 추정해서 맞춤 정책을 찾아드릴게요. 🦉</p>'
                    '</div>'
                )
                income_context = st.text_area(
                    "소득 참고 정보",
                    placeholder="(예시: 아버지는 개인택시 운전을 하시고, 어머니는 전업주부이십니다.)",
                    key="input_income_context",
                    label_visibility="collapsed",
                    on_change=set_input_guide,
                    args=("income_context",),
                )

            html_block('<div class="form-step-label">6. 현재 세대 구성 인원 수</div>')
            household_size = st.number_input(
                "현재 세대 구성 인원 수",
                min_value=1,
                max_value=8,
                value=1,
                step=1,
                key="input_household_size",
                label_visibility="collapsed",
                on_change=set_input_guide,
                args=("household_size",),
            )

            with st.expander("추천 정확도 높이기 (현재 거주 상태 선택)", expanded=False):
                living_status = st.selectbox(
                    "현재 거주 상태",
                    [
                        "부모님과 거주", "월세", "전세", "집 구하는 중",
                        "고시원·기숙사", "쪽방·비닐하우스 등", "기타",
                    ],
                    index=1,
                    key="input_living_status",
                    on_change=set_input_guide,
                    args=("details",),
                )
                
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.button(
                (
                    "조건 수정하여 다시 분석하기"
                    if st.session_state.page == "result"
                    else "나만의 주거 리포트 분석하기"
                ),
                type="primary",
                use_container_width=True,
            )

    guide_key = str(st.session_state.get("input_guide", "default"))
    guide_text = INPUT_GUIDES.get(guide_key, INPUT_GUIDES["default"])
    
    with right_panel:
        with st.container(key="right_panel_body"):
            right_slot = st.empty()
            with right_slot.container():
                if st.session_state.page == "result":
                    render_result_panel()
                else:
                    html_block(
                        f"""
                        <div class="welcome-container fade-in-up">
                            <div class="mascot-area-new">
                                <img src="https://i.ibb.co/ksxnBZm8/owl-doctor.png"
                                     alt="부엉이 박사" class="mascot-img-large"
                                     onerror="this.onerror=null;this.src='data:image/svg+xml;charset=UTF-8,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22%3E%3Ccircle cx=%2250%22 cy=%2250%22 r=%2248%22 fill=%22%23F0CA5E%22/%3E%3Ctext x=%2250%22 y=%2265%22 font-size=%2248%22 text-anchor=%22middle%22%3E%F0%9F%A6%89%3C/text%3E%3C/svg%3E';">
                                <div class="mascot-label">수석 연구원 <strong>부엉이 박사</strong> 🦉</div>
                            </div>
                            <h2 class="slogan">지피지기 백전백승!<br>주거 정책을 알고 나를 알면,<br>청년 주거 독립도 백전백승입니다! 🏡✨</h2>
                            <div class="speech-bubble">
                                {guide_text}
                            </div>
                            <div class="strength-list">
                                <div class="strength-card">
                                    <div class="strength-icon">🏢</div>
                                    <div class="strength-info"><h3>공고 마스터 정밀 매칭</h3><p>수많은 최신 공고 중 내 조건에 맞는 트랙만 찾아냅니다.</p></div>
                                </div>
                                <div class="strength-card">
                                    <div class="strength-icon">📑</div>
                                    <div class="strength-info"><h3>필수 서류 듀얼 뷰 가이드</h3><p>발급처 사이트와 필수 체크리스트를 양옆에 띄워 편안하게 도와드려요.</p></div>
                                </div>
                                <div class="strength-card">
                                    <div class="strength-icon">💡</div>
                                    <div class="strength-info"><h3>RAG 근거 기반 쉬운 설명</h3><p>공고 원문 근거를 찾고 복잡한 행정 용어를 쉬운 말로 정리합니다.</p></div>
                                </div>
                            </div>
                            <div class="trust-badge">※ 추천 엔진과 실제 공고 근거를 분리해 분석하며, LLM은 추천 점수를 임의로 변경하지 않습니다.</div>
                        </div>
                        """
                    )

    if submitted:
        try:
            right_slot.empty()
            with right_slot.container():
                html_block(
                    """
                    <div class="loading-card">
                        <img src="https://i.ibb.co/ksxnBZm8/owl-doctor.png" alt="분석 중" class="loading-mascot"
                             onerror="this.onerror=null;this.src='data:image/svg+xml;charset=UTF-8,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22%3E%3Ccircle cx=%2250%22 cy=%2250%22 r=%2248%22 fill=%22%23F0CA5E%22/%3E%3Ctext x=%2250%22 y=%2265%22 font-size=%2248%22 text-anchor=%22middle%22%3E%F0%9F%A6%89%3C/text%3E%3C/svg%3E';">
                        <div class="loading-title">맞춤 주거 리포트를 작성하고 있어요<span class="loading-dots"></span></div>
                        <div class="loading-desc">추천 후보를 찾고 최신 공고 근거와 행정 서류를 연결하는 중입니다.</div>
                        <div class="housing-tip-box">
                            <strong>💡 부엉이 박사의 청년 주거 상식 TIP!</strong>
                            전세 계약 전에는 등기부등본의 <span class="tooltip-trigger">갑구와 을구 확인<span class="tooltip-content">갑구는 소유권(진짜 주인), 을구는 소유권 이외의 권리(근저당 등 빚)를 의미해요.</span></span>은 필수랍니다! 진짜 주인이 맞는지 꼭 체크해요.
                        </div>
                    </div>
                    """
                )

            birth_year = CURRENT_YEAR - int(age)
            profile = build_user_profile(
                region=region,
                sigungu=sigungu,
                birth_year=int(birth_year),
                housing_owned=housing_owned,
                living_status=living_status,
                household_size=household_size,
                income_type=("모름" if income_type == "잘 모르겠어요" else "직접 입력"),
                monthly_income_manwon=monthly_income,
            )

            (
                cards,
                backend,
                rag_warning,
                llm_provider,
                llm_model,
                llm_warning,
            ) = run_recommendation_pipeline(profile)

            if (registered_region, registered_sigungu) != (region, sigungu):
                address_check = (
                    f"희망 거주지({region} {sigungu})와 주민등록 주소"
                    f"({registered_region} {registered_sigungu})가 달라 지역 우선공급·"
                    "거주기간 조건을 공고에서 추가 확인해야 합니다."
                )
                for card in cards[:3]:
                    checks = list(card.get("check_items", []))
                    if address_check not in checks:
                        card["check_items"] = [address_check, *checks][:3]

            if income_type == "잘 모르겠어요" and income_context.strip():
                income_check = (
                    "입력한 가족 직업·근로 정보만으로 월 소득을 확정하지 않았습니다. "
                    "신청 전 건강보험료·소득금액증명 등 공식 자료로 확인해 주세요."
                )
                for card in cards[:3]:
                    checks = list(card.get("check_items", []))
                    if income_check not in checks:
                        card["check_items"] = [income_check, *checks][:3]

            st.session_state.user_input = build_user_summary(
                profile=profile,
                registered_region=registered_region,
                registered_sigungu=registered_sigungu,
                housing_owned=housing_owned,
                living_status=living_status,
                household_size=household_size,
                income_type=("모름" if income_type == "잘 모르겠어요" else "직접 입력"),
                monthly_income_manwon=monthly_income,
                income_context=income_context,
            )
            st.session_state.user_profile_data = {
                "sido": profile.sido,
                "sigungu": profile.sigungu,
                "birth_year": profile.birth_year,
                "age": profile.age,
                "owns_house": profile.owns_house,
                "residence_status": profile.residence_status,
                "household_size": profile.household_size,
                "income_known": profile.income_known,
                "monthly_income": profile.monthly_income,
                "income_context": income_context.strip(),
                "desired_sido": region,
                "desired_sigungu": sigungu,
                "registered_sido": registered_region,
                "registered_sigungu": registered_sigungu,
            }
            st.session_state.selected_policy_card = {}
            st.session_state.selected_notice_id = ""
            st.session_state.selected_record_id = ""
            st.session_state.condition_answers = {}
            st.session_state.selected_document_id = ""
            st.session_state.celebrated_plan = ""
            st.session_state.recommendations = cards[:3]
            st.session_state.rag_backend = backend
            st.session_state.rag_warning = rag_warning
            st.session_state.llm_provider = llm_provider
            st.session_state.llm_model = llm_model
            st.session_state.llm_warning = llm_warning
            st.session_state.page = "result"
            st.rerun()

        except Exception as exc:
            st.error(
                "추천 실행 중 오류가 발생했습니다. "
                f"{type(exc).__name__}: {exc}"
            )


def render_result_panel() -> None:
    st.session_state["_popup_url_map"] = {}

    html_block(
        """
        <div class="result-header fade-in-up">
            <h2>당신만을 위한 맞춤 주거 리포트 🎉</h2>
            <p>입력하신 조건과 매칭 점수가 가장 높은 <strong>Top 3 추천 정책</strong>과
            실제 공고 기반 서류 가이드입니다.</p>
        </div>
        """
    )

    back_col, info_col = st.columns([1, 4])
    with back_col:
        if st.button("← 처음 화면"):
            st.session_state.page = "input"
            st.session_state.recommendations = []
            st.rerun()
    with info_col:
        backend_text = st.session_state.rag_backend or "확인 중"
        provider_text = st.session_state.llm_provider or "안전 설명"
        model_text = st.session_state.llm_model or "fallback"
        st.caption(
            f"검색 방식: {backend_text} · 설명 생성기: {provider_text} / {model_text}"
        )

    if st.session_state.rag_warning:
        st.warning(st.session_state.rag_warning)

    if st.session_state.llm_warning:
        st.warning(st.session_state.llm_warning)

    recommendations = st.session_state.recommendations

    if not recommendations:
        html_block(
            """
            <div class="error-card fade-in-up">
                <h3>😭 아쉽게도 추천 가능한 정책이 없어요</h3>
                <p>현재 대부분의 청년 주거 정책은 '무주택세대구성원'을 필수 요건으로 하고 있어요.<br>
                부모님 소유의 주택이라면 세대 분리를 통해 무주택 요건을 충족할 수 있는지 팁을 확인해 보세요!</p>
            </div>
            """
        )
    else:
        for item in recommendations:
            render_policy_card(item)

    render_popup_bridge(st.session_state.get("_popup_url_map", {}))
    render_document_guide_drawer()


def _condition_boolean_value(label: str) -> Any:
    return {
        "예": True,
        "아니오": False,
        "모름": None,
    }[label]


def _condition_boolean_label(value: Any) -> str:
    if value is True:
        return "예"
    if value is False:
        return "아니오"
    return "모름"


def _render_notice_candidate(
    candidate: Mapping[str, Any],
) -> None:
    with st.container(border=True):
        st.markdown(
            f"### {candidate['announcement_title']}"
        )
        st.write(
            f"**유형:** {candidate['housing_type']}  \n"
            f"**대상:** {candidate['target_group']}  \n"
            f"**지역:** {candidate['province']} "
            f"{candidate['city_county_district']}  \n"
            f"**운영기관:** {candidate['operator'] or '확인 필요'}"
        )
        st.caption(
            f"준비계획 연결 점수: {candidate['match_score']} · "
            f"데이터 상태: {candidate['review_status']}"
        )

        if st.button(
            "이 공고로 듀얼 뷰 서류 준비 계획 만들기",
            key=f"select_notice_{candidate['notice_id']}",
            use_container_width=True,
            type="primary"
        ):
            st.session_state.selected_notice_id = candidate["notice_id"]
            st.session_state.selected_record_id = candidate["record_id"]
            st.session_state.condition_answers = {}
            st.session_state.selected_document_id = ""
            st.session_state.celebrated_plan = ""
            st.rerun()


def _render_document_detail(
    document: Mapping[str, Any],
    notice_id: str,
) -> None:
    document_id = str(document["document_id"])
    checkbox_key = f"document_done_{notice_id}_{document_id}"

    col1, col2 = st.columns([5, 1])
    with col1:
        st.markdown(f"#### {document['canonical_name']}")
        st.caption(
            f"{document['status_label']} · "
            f"{document['certainty_label']} · "
            f"발급기관: {document['issuer'] or '확인 필요'}"
        )
    with col2:
        st.checkbox(
            "발급 완료",
            key=checkbox_key,
        )

    # 시안의 드로어처럼: 발급 사이트를 화면 옆에 별도 창으로 띄워
    # 체크리스트를 계속 보면서 발급을 진행할 수 있게 합니다.
    portal_url = str(document.get("direct_url") or document.get("portal_url") or "")
    if _has_real_portal_link(portal_url):
        render_side_popup_button(portal_url, "🔗 공식 발급 사이트 열기 (사이드 창)")
    else:
        html_block(_owl_empty_card_html(document))
    if document.get("site_search_keyword"):
        st.caption(f"포털에서 발급되지 않으면 \"{document['site_search_keyword']}\"로 검색하세요.")

    st.markdown("**온라인 발급 방법**")
    st.write(document["issue_method_online"] or "온라인 발급 방법을 공식 기관에서 확인하세요.")
    st.markdown("**오프라인 발급 방법**")
    st.write(document["issue_method_offline"] or "오프라인 발급 방법을 발급기관에서 확인하세요.")

    # 시안의 좌측 드로어 체크리스트처럼: 발급 사이트에서 진행하며 하나씩
    # 체크할 수 있도록 각 유의사항을 실제 체크박스로 렌더링합니다.
    guide_rows = [
        ("본인인증", document.get("identity_required")),
        ("수수료", document.get("fee_note")),
        ("처리시간", document.get("processing_time")),
        ("발급 옵션", document.get("required_options")),
        ("출력 형식", document.get("output_format")),
        ("유효기간 주의", document.get("validity_note")),
        ("제출 주의", document.get("cautions")),
        ("개인정보 주의", document.get("privacy_warning")),
    ]
    checklist_items = [(label, value) for label, value in guide_rows if value]
    if checklist_items:
        html_block('<div class="check-list-title">✅ 발급 시 확인 체크리스트</div>')
        for label, value in checklist_items:
            st.checkbox(
                f"**{label}**: {value}",
                key=f"doc_checklist_{notice_id}_{document_id}_{label}",
            )

    if document.get("mymy_statuses"):
        st.markdown("**MyMy 이용 가능 여부**")
        for status in document["mymy_statuses"]:
            st.write(f"- {status}")

    with st.expander("공고 원문 근거 확인", expanded=False):
        for reference in document.get("source_references", []):
            page = reference.get("source_page")
            page_text = f"p.{page}" if page is not None else "페이지 확인 필요"
            st.write(
                f"- {reference.get('source_file') or '원문 파일 확인 필요'} "
                f"{page_text} · "
                f"{reference.get('review_status') or '미검수'}"
            )
            original = reference.get("original_requirement_text")
            if original:
                st.caption(original[:500])


def render_service_plan_page() -> None:
    # 이 페이지는 40:60 입력/결과 분할 레이아웃(list_col/guide_col과 무관)을 쓰지
    # 않으므로 left/right-panel-marker를 넣지 않는다. 마커가 있으면
    # ".stApp:has(.left-panel-marker) { height:100vh; overflow:hidden }" 규칙만
    # 전역으로 켜지고, 그 안에서 실제로 스크롤을 담당하는 stColumn 레벨 규칙은
    # (이 페이지의 컬럼엔 마커가 없어 매치되지 않아) 걸리지 않아 페이지 전체가
    # 뷰포트 높이에 잘려 아래 내용을 스크롤로 볼 수 없게 된다.
    card = st.session_state.selected_policy_card
    profile = st.session_state.user_profile_data

    if not card:
        st.warning("선택된 추천 서비스가 없습니다.")
        if st.button("추천 결과로 돌아가기"):
            st.session_state.page = "result"
            st.rerun()
        return

    col_back, col_title = st.columns([1, 6])
    with col_back:
        if st.button("⬅️ 결과 화면으로", use_container_width=True):
            st.session_state.page = "result"
            st.rerun()
    with col_title:
        html_block(f'<div class="dual-guide-title">{escape(str(card["policy_name"]))}</div>')
        html_block('<div class="dual-guide-desc">선택하신 정책에 대한 실제 공고 기반 듀얼 뷰 서류 준비 가이드입니다.</div>')

    planner = get_service_application_planner()

    if not st.session_state.selected_notice_id:
        result = planner.find_related_notices(
            policy_id=str(card["policy_id"]),
            user_profile=profile,
            limit=5,
        )
        st.info(result["relation_note"])
        
        candidates = result.get("candidates", [])
        if not result.get("supported") or not candidates:
            st.warning("현재 데이터에서 이 추천 서비스와 연결할 수 있는 실제 공고를 찾지 못했습니다.")
            return

        st.markdown("## 관련 실제 공고 선택 (듀얼 뷰 시작)")
        for candidate in candidates:
            _render_notice_candidate(candidate)
        return

    notice_id = st.session_state.selected_notice_id
    record_id = st.session_state.selected_record_id

    initial_plan = planner.build_plan(
        notice_id=notice_id,
        record_id=record_id,
        user_profile=profile,
        condition_answers=st.session_state.condition_answers,
    )
    notice = initial_plan["notice"]
    
    official_url = notice.get("notice_url") or notice.get("application_url")

    # 듀얼 뷰 레이아웃 (왼쪽: 조건 및 서류 리스트 / 오른쪽: 발급 가이드)
    list_col, guide_col = st.columns([0.45, 0.55], gap="large")

    with list_col:
        with st.expander("공고 공통 준비 순서 및 추가 조건 확인", expanded=False):
            st.markdown(f"**{notice['announcement_title']}**")
            for step in initial_plan["steps"]:
                st.markdown(f"**{step['step']}. {step['title']}** — {step['description']}")

            st.divider()
            answers = dict(st.session_state.condition_answers)
            for question in initial_plan["questions"]:
                key = question["condition_key"]
                widget_key = f"condition_{notice_id}_{key}"

                if question["kind"] == "boolean":
                    current_label = _condition_boolean_label(
                        answers.get(key, initial_plan["condition_answers"].get(key))
                    )
                    options = ["모름", "예", "아니오"]
                    selected = st.radio(
                        question["label"],
                        options,
                        index=options.index(current_label),
                        horizontal=True,
                        key=widget_key,
                    )
                    answers[key] = _condition_boolean_value(selected)
                else:
                    options = question.get("options", ["미해당", "모름"])
                    current = answers.get(key, "모름")
                    if current not in options:
                        current = "모름"
                    answers[key] = st.selectbox(
                        question["label"],
                        options,
                        index=options.index(current),
                        key=widget_key,
                    )

            st.session_state.condition_answers = answers

        plan = planner.build_plan(
            notice_id=notice_id,
            record_id=record_id,
            user_profile=profile,
            condition_answers=answers,
        )
        document_plan = plan["document_plan"]
        documents = document_plan["documents"]
        
        total_documents = len(documents)
        completed_documents = sum(
            bool(st.session_state.get(f"document_done_{notice_id}_{doc['document_id']}", False))
            for doc in documents
        )
        progress = completed_documents / total_documents if total_documents else 0.0
        
        st.markdown(f"### 발급할 서류 선택 ({completed_documents}/{total_documents} 완료)")
        st.progress(progress)
        
        if document_plan.get("unanswered_conditions"):
            st.caption("아직 답하지 않은 조건: " + ", ".join(document_plan["unanswered_conditions"]))

        if documents:
            valid_document_ids = [str(d["document_id"]) for d in documents]
            if st.session_state.selected_document_id not in valid_document_ids:
                st.session_state.selected_document_id = valid_document_ids[0]

            status_colors = {
                "common": ("공통 준비", "#EAF6ED", "#28773C"),
                "applicable": ("현재 조건에서 필요", "#FFF4D8", "#9A6A00"),
                "needs_answer": ("조건 확인 필요", "#FFF0EE", "#B33A2B"),
            }

            for document in documents:
                doc_id_str = str(document["document_id"])
                done = bool(st.session_state.get(f"document_done_{notice_id}_{doc_id_str}", False))
                label, background, color = status_colors.get(
                    str(document.get("status")),
                    (str(document.get("status_label") or "확인 필요"), "#F4F5F6", "#6B7684"),
                )
                selected_marker = " · 선택됨" if doc_id_str == st.session_state.selected_document_id else ""
                
                html_block(
                    f"""
                    <div class="doc-preview-note" style="margin-bottom:8px; padding: 16px;">
                        <b>{'✅' if done else '📄'} {escape(str(document['canonical_name']))}</b><br>
                        <span style="display:inline-block;margin-top:6px;padding:3px 7px;border-radius:6px;background:{background};color:{color};font-size:11px;font-weight:800;">
                            {escape(label)}{escape(selected_marker)}
                        </span>
                    </div>
                    """
                )
                if st.button(
                    "가이드 보기" if not done else "가이드 다시 보기",
                    key=f"select_doc_{notice_id}_{doc_id_str}",
                    use_container_width=True,
                ):
                    st.session_state.selected_document_id = doc_id_str
                    st.rerun()

    with guide_col:
        if documents:
            selected_document = next(
                (d for d in documents if str(d["document_id"]) == st.session_state.selected_document_id), 
                documents[0]
            )
            html_block('<div class="dual-guide-title">상세 발급 가이드 (체크리스트)</div>')
            html_block('<div class="dual-guide-desc">공식 발급 사이트를 화면 옆에 별도 창으로 열어두고, 이 체크리스트를 그대로 보면서 하나씩 확인해 발급하세요. 다 받았다면 완료 체크박스를 눌러 진행률을 올려주세요.</div>')
            
            with st.container(border=True):
                _render_document_detail(selected_document, notice_id)

            all_completed = total_documents > 0 and completed_documents == total_documents
            if all_completed:
                html_block(
                    """
                    <div class="success-panel fade-in-up">
                        <h3>모든 서류 준비 완료! 🎉</h3>
                        <p>부엉이 지기와 함께 필요한 서류를 모두 확인했어요.<br>
                        이제 공고 원문에서 접수 기간을 마지막으로 확인하고 당당하게 신청해 보세요. 🦉</p>
                    </div>
                    """
                )
                if official_url and st.button(
                    "실제 공고 신청하러 가기",
                    key=f"apply_cta_{notice_id}_{record_id}",
                    type="primary",
                    use_container_width=True,
                ):
                    st.session_state.success_modal = {
                        "policy_name": str(card.get("policy_name") or ""),
                        "link": official_url,
                    }
                    st.rerun()


if st.session_state.page in {"input", "result"}:
    render_input_page()
elif st.session_state.page == "service_plan":
    render_service_plan_page()
else:
    st.session_state.page = "input"
    st.rerun()

render_success_modal()