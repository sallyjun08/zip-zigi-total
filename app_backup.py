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
import os

import streamlit as st

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
    "무주택": False,
    "주택 소유": True,
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
}

for key, value in STATE_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


@st.cache_resource(show_spinner=False)
def get_policy_retriever(
    requested_backend: str,
) -> tuple[PolicyRetriever, str, str]:
    """
    BGE-M3 + V2 Hybrid를 우선 사용합니다.
    초기화 실패 시 semantic, keyword 순서로 안전하게 전환합니다.
    """
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
    """환경변수에 따라 Ollama, Mock 또는 Disabled 설명기를 생성합니다."""
    return create_easy_explainer_from_env()


@st.cache_resource(show_spinner=False)
def get_service_application_planner() -> ServiceApplicationPlanner:
    """실제 공고·서류 데이터 기반 신청 준비 계획 서비스를 생성합니다."""
    repository = HousingPlanRepository(
        data_dir=BASE_DIR / "data",
    )
    return ServiceApplicationPlanner(repository)


def build_user_profile(
    region: str,
    sigungu: str,
    birth_year: int,
    housing_owned: str,
    living_status: str,
    household_size: str,
    income_type: str,
    monthly_income_manwon: int | None,
) -> UserProfile:
    income_known = income_type == "직접 입력"
    monthly_income_won = (
        int(monthly_income_manwon) * 10_000
        if income_known and monthly_income_manwon is not None
        else None
    )

    return UserProfile(
        sido=region,
        sigungu=sigungu,
        birth_year=int(birth_year),
        owns_house=HOUSE_OWNERSHIP_MAP[housing_owned],
        residence_status=RESIDENCE_STATUS_MAP[living_status],
        household_size=HOUSEHOLD_SIZE_MAP[household_size],
        income_known=income_known,
        monthly_income=monthly_income_won,
    )


def build_user_summary(
    profile: UserProfile,
    housing_owned: str,
    living_status: str,
    household_size: str,
    income_type: str,
    monthly_income_manwon: int | None,
) -> dict[str, str]:
    return {
        "지역": profile.sido,
        "시군구": profile.sigungu,
        "출생연도": f"{profile.birth_year}년",
        "나이 조건": f"만 {profile.age}세 전후",
        "주택 소유 여부": housing_owned,
        "현재 거주 상태": living_status,
        "가구 인원 수": household_size,
        "월 소득": (
            "모름"
            if income_type == "모름"
            else f"{int(monthly_income_manwon or 0):,}만 원"
        ),
    }


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
        # 5개 section × Top-1 = LLM에 전달되는 핵심 Top-5 근거
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


def badge_class(possibility: str) -> str:
    if possibility == "가능성 높음":
        return "badge-high"
    if possibility == "가능성 낮음":
        return "badge-low"
    return "badge-check"


def render_nav() -> None:
    html_block(
        """
        <div class="app-nav">
            <div class="nav-brand-wrap">
                <div class="nav-owl">🦉</div>
                <div>
                    <div class="nav-kicker">청년 주거 독립 연구소</div>
                    <div class="nav-brand">Zip.지기</div>
                </div>
            </div>
            <div class="nav-pill">추천 엔진 · RAG · 쉬운 설명 · 서류 계획</div>
        </div>
        """
    )

def render_policy_card(item: Mapping[str, Any]) -> None:
    checks_html = "".join(
        f"<li>{escape(str(check))}</li>"
        for check in item["check_items"]
    )

    mock_label = (
        '<span class="mock-label">개발용 데이터</span>'
        if item["is_mock_data"]
        else '<span class="mock-label">실제 공고 근거</span>'
    )

    match_score = escape(str(item["match_score"]))
    policy_name = escape(str(item["policy_name"]))
    category = escape(str(item["category"]))
    service_name = escape(str(item.get("service_name", item["policy_name"])))
    notice_id = escape(str(item.get("notice_id", "")))
    notice_region = escape(str(item.get("notice_region", "")))
    notice_score = item.get("notice_match_score")
    notice_meta_parts = [f"기반 서비스: {service_name}"]
    if notice_id:
        notice_meta_parts.append(f"공고 ID: {notice_id}")
    if notice_region:
        notice_meta_parts.append(f"지역: {notice_region}")
    if notice_score is not None:
        notice_meta_parts.append(f"공고 연결 점수: {notice_score}")
    notice_meta = escape(" · ".join(notice_meta_parts))
    possibility = escape(str(item["possibility"]))
    easy_description = escape(str(item["easy_description"]))
    reason = escape(str(item["reason"]))
    caution = escape(str(item["caution"]))
    link = escape(str(item["link"]), quote=True)

    html_block(
        f"""
        <div class="policy-card toss-card">
            <div class="policy-top">
                <div class="policy-heading">
                    <div class="match-badge">추천 {item['rank']}순위 · 점수 {match_score}</div>
                    <div class="policy-name">{policy_name}</div>
                    <div class="policy-category">{category} {mock_label}</div>
                    <div class="policy-category">{notice_meta}</div>
                </div>
                <div class="possibility-chip {badge_class(str(item['possibility']))}">
                    {possibility}
                </div>
            </div>

            <div class="policy-highlight">
                <div class="policy-section-title">부엉이 박사의 쉬운 설명</div>
                <div class="policy-text">{easy_description}</div>
            </div>

            <div class="policy-detail-grid">
                <div class="policy-mini-card">
                    <div class="policy-section-title">왜 추천했나요?</div>
                    <div class="policy-text">{reason}</div>
                </div>
                <div class="policy-mini-card">
                    <div class="policy-section-title">추가 확인 항목</div>
                    <ul class="check-list">{checks_html}</ul>
                </div>
            </div>

            <div class="caution-box">
                <b>확인해 주세요.</b> {caution}
            </div>

            <a class="policy-link" href="{link}" target="_blank">
                공식 페이지 확인하기 ↗
            </a>
        </div>
        """
    )

    button_col, evidence_col = st.columns([1.5, 1])
    with button_col:
        if st.button(
            "📋 이 서비스 신청 준비 계획 보기",
            key=f"open_plan_{item['policy_id']}",
            use_container_width=True,
        ):
            st.session_state.selected_policy_card = dict(item)
            st.session_state.selected_notice_id = str(item.get("notice_id") or "")
            st.session_state.selected_record_id = str(item.get("record_id") or "")
            st.session_state.condition_answers = {}
            st.session_state.page = "service_plan"
            st.rerun()

    with evidence_col:
        show_evidence = st.toggle(
            "검색 근거 펼치기",
            key=f"evidence_toggle_{item['policy_id']}",
        )

    if show_evidence:
        with st.container(border=True):
            st.caption(
                f"검색 상태: {item['rag_status']} · 데이터: {item.get('data_mode', '')} · 출처: {item['source_name']}"
            )
            st.caption(
                "설명 생성: "
                f"{item['generation_status']} · "
                f"{item['llm_provider']} / {item['llm_model']}"
            )
            if item.get("generation_error"):
                st.info(
                    "LLM 대신 안전 설명을 사용한 이유: "
                    f"{item['generation_error']}"
                )
            if item.get("used_chunk_ids"):
                st.caption(
                    "설명에 사용한 chunk: "
                    + ", ".join(item["used_chunk_ids"])
                )

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
                        chunk_type = evidence_item.get("chunk_type") or section_type
                        st.caption(
                            f"출처: {source_name} · {page_text} · "
                            f"chunk_type: {chunk_type} · 유사도 {evidence_item['similarity']}"
                        )
                        if source_url:
                            st.markdown(f"[원문 페이지 열기]({source_url})")


html_block(
    """
    <style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.css');

    :root {
        --bg-workspace: #F4F3ED;
        --bg-left: #FFFFFF;
        --main-olive: #6B705C;
        --dark-olive: #4A503D;
        --light-olive: #A5A994;
        --pastel-olive: #F4F5EF;
        --point-yellow: #F0CA5E;
        --text-main: #191F28;
        --text-desc: #6B7684;
        --text-sub: #8B95A1;
        --line: #E8E9E2;
        --white: #FFFFFF;
        --shadow-base: 0 12px 30px rgba(0, 0, 0, 0.045);
        --shadow-hover: 0 20px 42px rgba(0, 0, 0, 0.075);
    }

    html, body, [class*="css"] {
        font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
    }

    .stApp {
        background: var(--bg-workspace);
        color: var(--text-main);
    }

    .block-container {
        max-width: 1440px;
        padding-top: 1.4rem;
        padding-bottom: 4rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    #MainMenu, footer, header[data-testid="stHeader"] {
        visibility: hidden;
        height: 0;
    }

    .app-nav {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 18px;
        margin-bottom: 24px;
        padding: 4px 2px;
    }

    .nav-brand-wrap {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .nav-owl {
        width: 46px;
        height: 46px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 16px;
        background: var(--point-yellow);
        font-size: 25px;
        box-shadow: 0 8px 18px rgba(240, 202, 94, 0.25);
    }

    .nav-kicker {
        color: var(--main-olive);
        font-size: 11px;
        font-weight: 900;
        letter-spacing: .4px;
    }

    .nav-brand {
        color: var(--text-main);
        font-size: 22px;
        font-weight: 900;
        letter-spacing: -.4px;
    }

    .nav-pill {
        padding: 10px 16px;
        border: 1px solid rgba(107,112,92,.14);
        border-radius: 999px;
        background: rgba(255,255,255,.72);
        color: var(--dark-olive);
        font-size: 13px;
        font-weight: 800;
    }

    .left-panel-marker, .right-panel-marker { display: none; }

    div[data-testid="stVerticalBlockBorderWrapper"]:has(.left-panel-marker) {
        min-height: calc(100vh - 110px);
        padding: 34px 34px 30px;
        border: none !important;
        border-radius: 30px;
        background: var(--white);
        box-shadow: 10px 0 34px rgba(0,0,0,.035);
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:has(.right-panel-marker) {
        min-height: calc(100vh - 110px);
        padding: 46px 42px;
        border: none !important;
        border-radius: 30px;
        background: transparent;
    }

    .brand-header { margin-bottom: 34px; }

    .brand-header .subtitle {
        display: inline-block;
        margin-bottom: 8px;
        color: var(--main-olive);
        font-size: 13px;
        font-weight: 900;
        letter-spacing: .3px;
    }

    .brand-header h1 {
        margin: 0 0 11px;
        color: var(--text-main);
        font-size: 32px;
        font-weight: 900;
        letter-spacing: -.8px;
    }

    .brand-header p {
        margin: 0;
        color: var(--text-desc);
        font-size: 15px;
        font-weight: 500;
        line-height: 1.65;
        word-break: keep-all;
    }

    .form-note {
        margin: 6px 0 18px;
        padding: 13px 15px;
        border-radius: 14px;
        background: var(--pastel-olive);
        color: var(--dark-olive);
        font-size: 13px;
        line-height: 1.55;
    }

    label, .stRadio label, .stSelectbox label, .stNumberInput label {
        color: var(--text-main) !important;
        font-weight: 800 !important;
    }

    div[data-baseweb="select"] > div,
    div[data-testid="stNumberInput"] input {
        min-height: 54px;
        border: 1.5px solid transparent !important;
        border-radius: 16px !important;
        background: #F4F5F6 !important;
        color: var(--text-main) !important;
        box-shadow: none !important;
    }

    div[data-baseweb="select"] > div:focus-within,
    div[data-testid="stNumberInput"] input:focus {
        border-color: var(--main-olive) !important;
        background: var(--white) !important;
        box-shadow: 0 0 0 4px rgba(107,112,92,.10) !important;
    }

    div[data-testid="stRadio"] > div {
        gap: 8px;
    }

    div[data-testid="stRadio"] label {
        padding: 9px 12px;
        border-radius: 12px;
        background: #F4F5F6;
        transition: .2s ease;
    }

    div[data-testid="stRadio"] label:has(input:checked) {
        background: var(--pastel-olive);
        box-shadow: inset 0 0 0 1.5px var(--main-olive);
    }

    div.stButton > button,
    div[data-testid="stLinkButton"] a {
        min-height: 54px;
        border: none !important;
        border-radius: 16px !important;
        background: var(--main-olive) !important;
        color: white !important;
        font-size: 16px !important;
        font-weight: 900 !important;
        box-shadow: none !important;
        transition: transform .2s ease, background .2s ease;
    }

    div.stButton > button:hover,
    div[data-testid="stLinkButton"] a:hover {
        background: var(--dark-olive) !important;
        transform: translateY(-1px);
    }

    div.stButton > button:active { transform: scale(.985); }

    .speech-bubble {
        position: relative;
        max-width: 670px;
        margin: 0 auto 22px;
        padding: 29px 34px;
        border-radius: 30px 30px 30px 7px;
        background: white;
        color: var(--text-main);
        box-shadow: var(--shadow-base);
        font-size: 16px;
        line-height: 1.75;
        word-break: keep-all;
    }

    .mascot-area {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 18px;
        margin: 0 auto 32px;
    }

    .mascot-circle {
        width: 116px;
        height: 116px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
        background: var(--point-yellow);
        font-size: 61px;
        filter: drop-shadow(0 9px 16px rgba(0,0,0,.08));
    }

    .mascot-name {
        padding: 8px 15px;
        border-radius: 999px;
        background: rgba(107,112,92,.09);
        color: var(--dark-olive);
        font-size: 14px;
        font-weight: 900;
    }

    .research-log-header {
        max-width: 670px;
        margin: 0 auto 14px;
        padding-left: 5px;
        color: var(--text-main);
        font-size: 20px;
        font-weight: 900;
    }

    .strength-list {
        display: grid;
        grid-template-columns: 1fr;
        gap: 13px;
        max-width: 670px;
        margin: 0 auto;
    }

    .strength-card {
        display: flex;
        align-items: center;
        gap: 18px;
        padding: 20px 22px;
        border-radius: 22px;
        background: white;
        box-shadow: var(--shadow-base);
    }

    .strength-icon {
        width: 50px;
        height: 50px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex: 0 0 auto;
        border-radius: 16px;
        background: var(--pastel-olive);
        font-size: 23px;
    }

    .strength-info h3 {
        margin: 0 0 4px;
        color: var(--text-main);
        font-size: 16px;
        font-weight: 900;
    }

    .strength-info p {
        margin: 0;
        color: var(--text-desc);
        font-size: 14px;
        line-height: 1.55;
    }

    .trust-badge {
        margin: 25px auto 0;
        color: #969D9F;
        text-align: center;
        font-size: 12px;
        line-height: 1.6;
    }

    .result-hero {
        padding: 34px 38px;
        border-radius: 28px;
        background: linear-gradient(135deg, var(--dark-olive), var(--main-olive));
        color: white;
        box-shadow: 0 20px 48px rgba(74,80,61,.20);
        margin-bottom: 24px;
    }

    .result-kicker {
        margin-bottom: 9px;
        color: #F3E8B1;
        font-size: 13px;
        font-weight: 900;
    }

    .result-title {
        font-size: 34px;
        font-weight: 900;
        letter-spacing: -.8px;
        margin-bottom: 10px;
    }

    .result-desc {
        max-width: 900px;
        line-height: 1.7;
        opacity: .94;
    }

    .section-title {
        margin: 30px 0 8px;
        color: var(--text-main);
        font-size: 26px;
        font-weight: 900;
        letter-spacing: -.5px;
    }

    .section-desc {
        margin-bottom: 20px;
        color: var(--text-desc);
        line-height: 1.65;
    }

    .summary-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 13px;
        margin-bottom: 34px;
    }

    .summary-card {
        min-height: 98px;
        padding: 18px;
        border: 1px solid rgba(107,112,92,.10);
        border-radius: 20px;
        background: white;
        box-shadow: var(--shadow-base);
    }

    .summary-label {
        margin-bottom: 8px;
        color: var(--text-sub);
        font-size: 12px;
        font-weight: 900;
    }

    .summary-value {
        color: var(--dark-olive);
        font-size: 18px;
        font-weight: 900;
        word-break: keep-all;
    }

    .toss-card,
    .policy-card {
        padding: 30px;
        border: 1px solid rgba(0,0,0,.025);
        border-radius: 28px;
        background: white;
        box-shadow: var(--shadow-base);
    }

    .policy-card { margin: 0 0 12px; }

    .policy-top {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 20px;
        margin-bottom: 20px;
    }

    .match-badge {
        display: inline-flex;
        padding: 7px 12px;
        margin-bottom: 13px;
        border-radius: 11px;
        background: rgba(107,112,92,.09);
        color: var(--dark-olive);
        font-size: 13px;
        font-weight: 900;
    }

    .policy-name {
        color: var(--text-main);
        font-size: 25px;
        font-weight: 900;
        line-height: 1.38;
        word-break: keep-all;
    }

    .policy-category {
        margin-top: 9px;
        color: var(--text-desc);
        font-size: 14px;
        font-weight: 700;
    }

    .mock-label {
        display: inline-block;
        margin-left: 7px;
        padding: 4px 8px;
        border-radius: 999px;
        background: #F1F2EF;
        color: #74796B;
        font-size: 11px;
        font-weight: 900;
    }

    .possibility-chip,
    .badge-high, .badge-check, .badge-low {
        display: inline-flex;
        align-items: center;
        flex: 0 0 auto;
        padding: 8px 12px;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 900;
    }

    .badge-high { background: #EAF6ED; color: #28773C; }
    .badge-check { background: #FFF4D8; color: #9A6A00; }
    .badge-low { background: #FFF0EE; color: #B33A2B; }

    .policy-highlight {
        padding: 21px 23px;
        border-radius: 18px;
        background: var(--pastel-olive);
    }

    .policy-detail-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 14px;
        margin-top: 14px;
    }

    .policy-mini-card {
        padding: 20px 21px;
        border: 1px solid var(--line);
        border-radius: 18px;
        background: #FBFBF9;
    }

    .policy-section-title {
        margin-bottom: 8px;
        color: var(--dark-olive);
        font-size: 14px;
        font-weight: 900;
    }

    .policy-text, .check-list {
        color: #424A4F;
        font-size: 15px;
        line-height: 1.72;
    }

    .check-list { margin: 0; padding-left: 19px; }

    .caution-box {
        margin-top: 14px;
        padding: 15px 18px;
        border: 1px solid rgba(240,202,94,.42);
        border-radius: 16px;
        background: rgba(240,202,94,.13);
        color: #6D571A;
        font-size: 14px;
        line-height: 1.65;
    }

    .policy-link {
        display: inline-flex;
        margin-top: 16px;
        padding: 11px 15px;
        border-radius: 13px;
        background: #F4F5F1;
        color: var(--dark-olive) !important;
        font-weight: 900;
        text-decoration: none !important;
    }

    .notice {
        margin-top: 24px;
        padding: 20px 22px;
        border: 1px solid rgba(240,202,94,.50);
        border-radius: 19px;
        background: #FFF9E9;
        color: #735B16;
        line-height: 1.7;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: rgba(107,112,92,.13) !important;
        border-radius: 20px !important;
        background: rgba(255,255,255,.82);
    }

    div[data-testid="stExpander"] {
        border: 1px solid rgba(107,112,92,.13) !important;
        border-radius: 16px !important;
        background: rgba(255,255,255,.78);
    }

    div[data-testid="stAlert"] {
        border-radius: 16px;
    }

    @media (max-width: 1050px) {
        .summary-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .policy-detail-grid { grid-template-columns: 1fr; }
        .nav-pill { display: none; }
    }

    @media (max-width: 760px) {
        .block-container { padding-left: 1rem; padding-right: 1rem; }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.left-panel-marker),
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.right-panel-marker) {
            min-height: auto;
            padding: 25px 20px;
        }
        .result-title { font-size: 28px; }
        .summary-grid { grid-template-columns: 1fr; }
        .policy-top { flex-direction: column; }
        .speech-bubble { padding: 24px; }
    }
    </style>
    """
)


def render_input_page() -> None:
    left_panel, right_panel = st.columns([0.43, 0.57], gap="small")

    with left_panel:
        with st.container(border=True):
            html_block('<div class="left-panel-marker"></div>')
            html_block(
                """
                <div class="brand-header">
                    <span class="subtitle">청년 주거 독립 연구소</span>
                    <h1>Zip.지기 🦉</h1>
                    <p>몇 가지 질문만으로 나에게 맞는 주거 정책을 찾고,<br>
                    신청에 필요한 서류 준비 순서까지 안내해 드려요.</p>
                </div>
                """
            )

            html_block(
                """
                <div class="form-note">
                    6개 핵심 질문과 1개 선택 질문을 사용합니다.
                    월 소득을 몰라도 추천을 진행할 수 있어요.
                </div>
                """
            )

            st.markdown("#### 1·2. 희망 거주 지역")
            region_col, sigungu_col = st.columns(2, gap="small")
            with region_col:
                region = st.selectbox(
                    "시·도",
                    REGIONS,
                    index=8,
                    key="input_region",
                    label_visibility="collapsed",
                )
            with sigungu_col:
                sigungu = st.selectbox(
                    "시·군·구",
                    SIGUNGU_BY_REGION[region],
                    key=f"input_sigungu_{region}",
                    label_visibility="collapsed",
                )

            st.markdown("#### 3. 출생연도")
            birth_year = st.number_input(
                "출생연도",
                min_value=1950,
                max_value=CURRENT_YEAR,
                value=2002,
                step=1,
                label_visibility="collapsed",
            )
            st.caption(
                f"현재 연도 기준 만 {CURRENT_YEAR - int(birth_year)}세 전후로 계산합니다."
            )

            housing_owned = st.radio(
                "4. 주택 소유 여부",
                ["무주택", "주택 소유", "잘 모르겠음"],
                horizontal=True,
            )

            living_status = st.selectbox(
                "5. 현재 거주 상태",
                [
                    "부모님과 거주", "월세", "전세", "집 구하는 중",
                    "고시원·기숙사", "쪽방·비닐하우스 등", "기타",
                ],
                index=1,
            )

            household_size = st.selectbox(
                "6. 가구 인원 수",
                ["1인 가구", "2인 가구", "3인 가구", "4인 이상"],
            )

            income_type = st.radio(
                "+1. 가구 월 평균 소득",
                ["모름", "직접 입력"],
                horizontal=True,
            )

            monthly_income = None
            if income_type == "직접 입력":
                monthly_income = st.number_input(
                    "월 소득 입력",
                    min_value=0,
                    max_value=2000,
                    value=200,
                    step=10,
                    help="세전 기준이며 단위는 만 원입니다.",
                )

            st.markdown("")
            submitted = st.button(
                "나만의 주거 리포트 분석하기",
                type="primary",
                use_container_width=True,
            )

    with right_panel:
        with st.container(border=True):
            html_block('<div class="right-panel-marker"></div>')
            html_block(
                """
                <div class="speech-bubble">
                    반가워요! 주거 독립을 연구하는 부엉이 박사입니다. 🦉<br><br>
                    복잡한 정책과 행정 서류 때문에 막막하셨죠? 왼쪽에서 몇 가지만
                    알려주시면, 추천 엔진이 정책 후보를 찾고 RAG가 근거를 검색한 뒤
                    LLM이 쉬운 말로 설명해 드릴게요. RAG 결과는 실제 공고 CSV와 원문 청크만 사용하며, 추천받은 서비스는 실제 공고와
                    연결해 <strong>신청 순서와 필수 서류 준비 계획</strong>까지 만들 수 있습니다.
                </div>
                <div class="mascot-area">
                    <div class="mascot-circle">🦉</div>
                    <span class="mascot-name">수석 연구원 부엉이 박사</span>
                </div>
                <div class="research-log-header">부엉이 박사의 연구 일지 📝</div>
                <div class="strength-list">
                    <div class="strength-card">
                        <div class="strength-icon">🏢</div>
                        <div class="strength-info">
                            <h3>조건 기반 정책 매칭</h3>
                            <p>지역, 출생연도, 주택 소유 여부, 거주 상태와 가구 조건으로 후보를 좁힙니다.</p>
                        </div>
                    </div>
                    <div class="strength-card">
                        <div class="strength-icon">🔎</div>
                        <div class="strength-info">
                            <h3>RAG 근거 검색</h3>
                            <p>추천된 정책 문서 안에서 지원 대상, 소득, 혜택과 신청 방법의 근거를 찾습니다.</p>
                        </div>
                    </div>
                    <div class="strength-card">
                        <div class="strength-icon">📑</div>
                        <div class="strength-info">
                            <h3>실제 서류 준비 계획</h3>
                            <p>관련 공고를 선택하면 조건별 필요 서류, 발급처와 제출 단계를 체크리스트로 안내합니다.</p>
                        </div>
                    </div>
                </div>
                <div class="trust-badge">
                    ※ LLM은 추천 점수나 자격을 임의로 변경하지 않고, 추천 엔진과 문서 근거를 쉽게 설명합니다.<br>
                    최종 신청 가능 여부는 반드시 공식 공고에서 확인해야 합니다.
                </div>
                """
            )

    if submitted:
        try:
            profile = build_user_profile(
                region=region,
                sigungu=sigungu,
                birth_year=int(birth_year),
                housing_owned=housing_owned,
                living_status=living_status,
                household_size=household_size,
                income_type=income_type,
                monthly_income_manwon=monthly_income,
            )

            with st.spinner(
                "부엉이 박사가 정책 후보와 문서 근거를 분석하고 있습니다..."
            ):
                (
                    cards,
                    backend,
                    rag_warning,
                    llm_provider,
                    llm_model,
                    llm_warning,
                ) = run_recommendation_pipeline(profile)

            st.session_state.user_input = build_user_summary(
                profile=profile,
                housing_owned=housing_owned,
                living_status=living_status,
                household_size=household_size,
                income_type=income_type,
                monthly_income_manwon=monthly_income,
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
            }
            st.session_state.selected_policy_card = {}
            st.session_state.selected_notice_id = ""
            st.session_state.selected_record_id = ""
            st.session_state.condition_answers = {}
            st.session_state.recommendations = cards
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


def render_result_page() -> None:
    render_nav()

    html_block(
        """
        <div class="result-hero">
            <div class="result-kicker">ZIP.지기 분석 완료</div>
            <div class="result-title">당신만을 위한 주거 리포트가 도착했어요 🎉</div>
            <div class="result-desc">
                규칙 기반 추천 엔진이 서비스 계열을 선정하고, RAG가 실제 모집공고의
                notice_id 범위 안에서 원문 근거를 찾은 뒤 LLM이 쉬운 한국어로 정리했습니다. 아래 결과는 신청
                가능성을 돕는 안내이며 최종 자격 판정은 아닙니다.
            </div>
        </div>
        """
    )

    back_col, info_col = st.columns([1, 4])
    with back_col:
        if st.button("← 조건 다시 입력", use_container_width=True):
            st.session_state.page = "input"
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

    html_block('<div class="section-title">입력 정보 요약</div>')
    html_block(
        '<div class="section-desc">분석에 사용한 조건입니다. 잘못된 값이 있으면 다시 입력해 주세요.</div>'
    )

    summary_html = ""
    for key, value in st.session_state.user_input.items():
        summary_html += (
            '<div class="summary-card">'
            f'<div class="summary-label">{escape(str(key))}</div>'
            f'<div class="summary-value">{escape(str(value))}</div>'
            '</div>'
        )
    html_block(f'<div class="summary-grid">{summary_html}</div>')

    html_block('<div class="section-title">추천된 주거 혜택</div>')
    html_block(
        """
        <div class="section-desc">
            추천 점수 순으로 정렬했습니다. 각 정책에서 쉬운 설명과 추천 이유를 확인하고,
            실제 신청 준비가 필요한 정책은 「신청 준비 계획 보기」를 눌러 관련 공고와
            서류 체크리스트로 이어갈 수 있습니다.
        </div>
        """
    )

    recommendations = st.session_state.recommendations

    if not recommendations:
        st.warning(
            "현재 입력 조건으로 하드 필터를 통과한 정책이 없습니다. "
            "조건을 다시 확인하거나 추천 정책 데이터를 확장해야 합니다."
        )
    else:
        for item in recommendations:
            render_policy_card(item)
            st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    html_block(
        """
        <div class="notice">
            <b>중요 안내</b><br>
            실제 RAG 데이터는 data/housing_supply_notice_master.csv와
            data/housing_rag_chunks_review_ready.csv에서 읽습니다. 개발용 표식이 발견된
            공고는 기본적으로 결과에서 제외됩니다. 최종 신청 자격,
            접수 기간과 제출 서류는 반드시 공식 공고에서 다시 확인해 주세요.
        </div>
        """
    )



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
            "이 공고로 준비 계획 만들기",
            key=f"select_notice_{candidate['notice_id']}",
            use_container_width=True,
        ):
            st.session_state.selected_notice_id = (
                candidate["notice_id"]
            )
            st.session_state.selected_record_id = (
                candidate["record_id"]
            )
            st.session_state.condition_answers = {}
            st.rerun()


def _render_document_detail(
    document: Mapping[str, Any],
    notice_id: str,
) -> None:
    document_id = str(document["document_id"])
    checkbox_key = (
        f"document_done_{notice_id}_{document_id}"
    )

    col1, col2 = st.columns([5, 1])
    with col1:
        st.markdown(
            f"#### {document['canonical_name']}"
        )
        st.caption(
            f"{document['status_label']} · "
            f"{document['certainty_label']} · "
            f"발급기관: {document['issuer'] or '확인 필요'}"
        )
    with col2:
        st.checkbox(
            "준비 완료",
            key=checkbox_key,
        )

    with st.expander(
        f"{document['canonical_name']} 발급 방법과 공고 근거",
        expanded=False,
    ):
        st.markdown("**온라인 발급 방법**")
        st.write(
            document["issue_method_online"]
            or "온라인 발급 방법을 공식 기관에서 확인하세요."
        )

        st.markdown("**오프라인 발급 방법**")
        st.write(
            document["issue_method_offline"]
            or "오프라인 발급 방법을 발급기관에서 확인하세요."
        )

        if document.get("site_search_keyword"):
            st.info(
                "공식 포털 검색어: "
                f"{document['site_search_keyword']}"
            )

        portal_url = (
            document.get("direct_url")
            or document.get("portal_url")
        )
        if portal_url:
            st.markdown(
                f"[공식 발급 포털 열기]({portal_url})"
            )

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
        for label, value in guide_rows:
            if value:
                st.markdown(f"**{label}**")
                st.write(value)

        if document.get("submission_stages"):
            st.markdown("**제출 단계**")
            for stage in document["submission_stages"]:
                st.write(f"- {stage}")

        if document.get("mymy_statuses"):
            st.markdown("**MyMy 이용 가능 여부**")
            for status in document["mymy_statuses"]:
                if status == "N":
                    explanation = "MyMy 제출 대상이 아닌 것으로 추출됨"
                elif status == "조건부":
                    explanation = "일부 신청자 또는 일부 서류만 가능할 수 있음"
                else:
                    explanation = "공고문에서 이용 가능 여부를 추가 확인해야 함"
                st.write(f"- {status}: {explanation}")

        st.markdown("**공고 원문 근거**")
        for reference in document.get(
            "source_references",
            [],
        ):
            page = reference.get("source_page")
            page_text = (
                f"p.{page}"
                if page is not None
                else "페이지 확인 필요"
            )
            st.write(
                f"- {reference.get('source_file') or '원문 파일 확인 필요'} "
                f"{page_text} · "
                f"{reference.get('review_status') or '미검수'}"
            )
            original = reference.get(
                "original_requirement_text"
            )
            if original:
                st.caption(original[:500])


def render_service_plan_page() -> None:
    render_nav()

    card = st.session_state.selected_policy_card
    profile = st.session_state.user_profile_data

    if not card:
        st.warning(
            "선택된 추천 서비스가 없습니다. 추천 결과로 돌아가 다시 선택해 주세요."
        )
        if st.button("추천 결과로 돌아가기"):
            st.session_state.page = "result"
            st.rerun()
        return

    top_left, top_right = st.columns([5, 1])
    with top_left:
        html_block(
            f"""
            <div class="result-hero">
                <div class="result-title">신청 준비 계획</div>
                <div class="result-desc">
                    추천 서비스 <b>{escape(str(card['policy_name']))}</b>와 관련된
                    실제 공고를 선택하고, 조건별 서류와 발급 방법을 확인합니다.
                </div>
            </div>
            """
        )
    with top_right:
        if st.button(
            "추천 결과로",
            use_container_width=True,
        ):
            st.session_state.page = "result"
            st.rerun()

    planner = get_service_application_planner()

    if not st.session_state.selected_notice_id:
        result = planner.find_related_notices(
            policy_id=str(card["policy_id"]),
            user_profile=profile,
            limit=5,
        )

        st.info(result["relation_note"])
        st.warning(
            "이 실제 공고 데이터는 추천 결과를 다시 판단하기 위한 것이 아니라, "
            "추천받은 서비스의 신청 준비 절차와 서류 가이드를 만들기 위해 사용합니다."
        )

        candidates = result.get("candidates", [])
        if not result.get("supported") or not candidates:
            st.warning(
                "현재 데이터에서 이 추천 서비스와 연결할 수 있는 실제 주택공급 공고를 "
                "찾지 못했습니다. 이 서비스는 기존 공식 페이지에서 조건과 서류를 "
                "추가 확인해야 합니다."
            )
            return

        st.markdown("## 관련 실제 공고 선택")
        st.caption(
            "지역·대상계층·주택유형을 기준으로 준비 계획에 활용할 공고를 정렬했습니다."
        )

        for candidate in candidates:
            _render_notice_candidate(candidate)
        return

    notice_id = st.session_state.selected_notice_id
    record_id = st.session_state.selected_record_id

    # 질문 목록과 초기값 확인
    initial_plan = planner.build_plan(
        notice_id=notice_id,
        record_id=record_id,
        user_profile=profile,
        condition_answers=st.session_state.condition_answers,
    )
    notice = initial_plan["notice"]

    st.markdown(f"## {notice['announcement_title']}")
    st.write(
        f"**운영기관:** {notice['operator'] or '확인 필요'}  \n"
        f"**주택유형:** {notice['housing_type']}  \n"
        f"**대상계층:** {notice['target_group']}  \n"
        f"**지역:** {notice['province']} {notice['city_county_district']}"
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button(
            "다른 공고 선택",
            use_container_width=True,
        ):
            st.session_state.selected_notice_id = ""
            st.session_state.selected_record_id = ""
            st.session_state.condition_answers = {}
            st.rerun()
    with col2:
        official_url = (
            notice.get("notice_url")
            or notice.get("application_url")
        )
        if official_url:
            st.link_button(
                "공식 신청 페이지 열기",
                official_url,
                use_container_width=True,
            )

    if notice.get("needs_review"):
        st.warning(
            "이 공고 데이터는 자동 추출 1차본입니다. "
            "표시된 조건과 서류는 원문 페이지와 반드시 대조해야 합니다."
        )

    st.markdown("## 준비 순서")
    for step in initial_plan["steps"]:
        st.markdown(
            f"**{step['step']}. {step['title']}** — "
            f"{step['description']}"
        )

    # 조건 질문
    answers = dict(
        st.session_state.condition_answers
    )
    with st.expander(
        "공고별 추가 조건 확인",
        expanded=True,
    ):
        st.caption(
            "답변에 따라 조건부 서류가 추가되거나 제외됩니다. "
            "모르는 항목은 '모름'으로 두어도 됩니다."
        )

        for question in initial_plan["questions"]:
            key = question["condition_key"]
            widget_key = f"condition_{notice_id}_{key}"

            if question["kind"] == "boolean":
                current_label = _condition_boolean_label(
                    answers.get(
                        key,
                        initial_plan[
                            "condition_answers"
                        ].get(key),
                    )
                )
                options = ["모름", "예", "아니오"]
                selected = st.radio(
                    question["label"],
                    options,
                    index=options.index(current_label),
                    horizontal=True,
                    key=widget_key,
                )
                answers[key] = _condition_boolean_value(
                    selected
                )
            else:
                options = question.get(
                    "options",
                    ["미해당", "모름"],
                )
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

    st.markdown("## 서류 준비 현황")
    total_documents = len(documents)
    completed_documents = sum(
        bool(
            st.session_state.get(
                f"document_done_{notice_id}_{doc['document_id']}",
                False,
            )
        )
        for doc in documents
    )
    progress = (
        completed_documents / total_documents
        if total_documents
        else 0.0
    )
    st.progress(progress)
    st.write(
        f"{completed_documents} / {total_documents}개 준비 완료"
    )

    st.info(document_plan["review_warning"])

    if document_plan.get("unanswered_conditions"):
        st.warning(
            "아직 답하지 않은 조건: "
            + ", ".join(
                document_plan["unanswered_conditions"]
            )
        )

    group_config = [
        (
            "common",
            "공통 준비 후보",
            "공고에서 공통적으로 요구하는 것으로 추출된 서류입니다.",
        ),
        (
            "applicable",
            "현재 답변에서 준비 후보",
            "입력한 추가 조건과 일치하여 필요할 수 있는 서류입니다.",
        ),
        (
            "needs_answer",
            "조건 확인 후 결정",
            "아직 모르는 조건에 따라 필요 여부가 달라지는 서류입니다.",
        ),
    ]

    for status, title, description in group_config:
        group = [
            document
            for document in documents
            if document["status"] == status
        ]
        if not group:
            continue

        st.markdown(f"### {title} ({len(group)}개)")
        st.caption(description)
        for document in group:
            with st.container(border=True):
                _render_document_detail(
                    document,
                    notice_id,
                )

    st.markdown("## 제출 단계별 확인")
    for stage, document_ids in document_plan[
        "stage_groups"
    ].items():
        document_names = [
            document["canonical_name"]
            for document in documents
            if document["document_id"]
            in document_ids
        ]
        with st.expander(
            f"{stage} ({len(document_names)}개)",
            expanded=False,
        ):
            for name in document_names:
                st.write(f"- {name}")

    if plan.get("guidance_chunks"):
        with st.expander(
            "공고 원문에서 추출한 신청·제출 안내",
            expanded=False,
        ):
            for chunk in plan["guidance_chunks"]:
                st.markdown(
                    f"**{chunk['chunk_type']}**"
                )
                st.write(chunk["text"])
                page_start = chunk.get(
                    "source_page_start"
                )
                page_end = chunk.get("source_page_end")
                page_text = (
                    f"p.{page_start}"
                    if page_start == page_end
                    else f"p.{page_start}~{page_end}"
                )
                st.caption(
                    f"{chunk['source_file']} {page_text} · "
                    f"{chunk['review_status']}"
                )


if st.session_state.page == "input":
    render_input_page()
elif st.session_state.page == "result":
    render_result_page()
elif st.session_state.page == "service_plan":
    render_service_plan_page()
else:
    st.session_state.page = "input"
    st.rerun()
