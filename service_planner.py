# -*- coding: utf-8 -*-
"""
추천 서비스 신청 준비 계획 생성기
================================

이 모듈은 실제 주택공급 공고 데이터로 새로운 정책 추천을 하지 않습니다.
기존 추천 엔진이 고른 서비스와 관련된 실제 공고를 연결하고,
사용자가 해당 서비스를 신청하기 위해 확인할 조건과 준비 서류를 안내합니다.

데이터 흐름
-----------
추천된 policy_id
    -> 관련 실제 공고 후보 조회
    -> 사용자가 공고 선택
    -> notice_id로 서류 규칙 조회
    -> 조건 질문 응답 평가
    -> document_id로 발급 가이드 조인
    -> 단계별 신청 준비 계획 생성

주의
----
현재 공고와 서류 규칙 데이터는 자동 추출 1차본이므로,
review_status가 '검수완료'가 아닌 항목은 확정 필수 서류가 아니라
'필요할 수 있는 서류'로 표현해야 합니다.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
import json
import math

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "data"

NOTICE_FILE = "housing_supply_notice_master.csv"
DOCUMENT_FILE = "housing_document_master.csv"
RULE_FILE = "housing_notice_document_rules_review_ready.csv"
CHUNK_FILE = "housing_rag_chunks_review_ready.csv"


SERVICE_NOTICE_MAP: dict[str, dict[str, Any]] = {
    "JEONSE_LOAN": {
        "housing_types": ["청년 전세임대"],
        "target_keywords": ["청년", "대학생", "취업준비생"],
        "relation_type": "related_notice",
        "relation_note": (
            "추천된 전세자금 서비스와 동일한 대출 상품 공고는 아닙니다. "
            "이 데이터에서는 관련 주거대안인 청년 전세임대 공고의 준비 절차를 안내합니다."
        ),
    },
    "LH_MAEIP": {
        "housing_types": [
            "청년 매입임대",
            "기숙사형 청년주택",
            "청년·신혼부부 매입임대리츠",
        ],
        "target_keywords": ["청년", "대학생", "취업준비생"],
        "relation_type": "direct_family",
        "relation_note": (
            "추천된 청년 매입임대 계열과 관련된 실제 모집공고를 연결했습니다."
        ),
    },
    "HAPPY_HOUSE": {
        "housing_types": ["행복주택"],
        "target_keywords": ["청년", "대학생", "취업준비생", "사회초년생"],
        "relation_type": "direct_family",
        "relation_note": (
            "추천된 행복주택 계열과 관련된 실제 모집공고를 연결했습니다."
        ),
    },
    "VULNERABLE_HOUSING_RELOCATION": {
        "housing_types": ["영구임대", "일반 매입임대", "국민임대"],
        "target_keywords": ["취약", "저소득", "무주택"],
        "relation_type": "related_notice",
        "relation_note": (
            "이주지원 사업 자체 공고가 아니라, 관련 공공임대·매입임대 공고의 "
            "준비 절차를 안내합니다."
        ),
    },
}


CONDITION_QUESTIONS: dict[str, dict[str, Any]] = {
    "applicant_type_student": {
        "label": "현재 대학생이거나 입학·복학 예정인가요?",
        "kind": "boolean",
    },
    "applicant_type_jobseeker": {
        "label": "현재 취업준비생 자격으로 신청하나요?",
        "kind": "boolean",
    },
    "applicant_type_hanparent": {
        "label": "한부모가족 자격으로 신청하나요?",
        "kind": "boolean",
    },
    "pregnancy_claim": {
        "label": "임신 사실을 자격 조건으로 제출하나요?",
        "kind": "boolean",
    },
    "adopted_child": {
        "label": "입양한 자녀와 관련된 자격을 적용받나요?",
        "kind": "boolean",
    },
    "spouse_separate_household": {
        "label": "배우자와 주민등록상 세대가 분리되어 있나요?",
        "kind": "boolean",
    },
    "proxy_application": {
        "label": "본인이 아닌 대리인이 신청하나요?",
        "kind": "boolean",
    },
    "foreign_family_member": {
        "label": "가구원 중 외국인 또는 국내거소신고자가 있나요?",
        "kind": "boolean",
    },
    "housing_vulnerable": {
        "label": "고시원·쪽방 등 주거취약시설에 거주하고 있나요?",
        "kind": "boolean",
    },
    "address_changed_after_notice": {
        "label": "공고일 이후 주소가 변경되었나요?",
        "kind": "boolean",
    },
    "member_moved_after_notice": {
        "label": "공고일 이후 세대원이 전입하거나 전출했나요?",
        "kind": "boolean",
    },
    "subscription_account_evidence": {
        "label": "청약통장 가입 또는 순위 증빙을 제출해야 하나요?",
        "kind": "boolean",
    },
    "priority_supply": {
        "label": "우선공급 또는 특별공급 유형으로 신청하나요?",
        "kind": "boolean",
    },
    "employment_evidence_required": {
        "label": "근로 또는 재직 사실을 자격 조건으로 증명해야 하나요?",
        "kind": "boolean",
    },
    "applicant_type": {
        "label": "어떤 신청자 유형으로 신청하나요?",
        "kind": "choice",
        "options": ["미해당", "신혼부부", "예비신혼부부", "모름"],
    },
}


CONDITION_ANSWER_LABELS = {
    True: "예",
    False: "아니오",
    None: "모름",
}


class ServicePlannerError(RuntimeError):
    """서비스 준비 계획 데이터 또는 처리 오류."""


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = _safe_str(value).lower()
    return text in {"true", "1", "y", "yes", "예"}


def _profile_to_dict(profile: Any) -> dict[str, Any]:
    if profile is None:
        return {}
    if isinstance(profile, Mapping):
        return dict(profile)
    if is_dataclass(profile):
        return asdict(profile)

    fields = (
        "sido",
        "sigungu",
        "birth_year",
        "age",
        "owns_house",
        "residence_status",
        "household_size",
        "income_known",
        "monthly_income",
    )
    return {
        field: getattr(profile, field)
        for field in fields
        if hasattr(profile, field)
    }


def _parse_json_dict(value: Any) -> dict[str, Any]:
    text = _safe_str(value)
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _split_regions(value: Any) -> list[str]:
    text = _safe_str(value)
    if not text:
        return []
    return [item.strip() for item in text.split("|") if item.strip()]


def _contains_any(text: Any, keywords: Sequence[str]) -> bool:
    normalized = _safe_str(text)
    return any(keyword in normalized for keyword in keywords)


def _age_from_profile(profile: Mapping[str, Any]) -> Optional[int]:
    age = profile.get("age")
    if isinstance(age, (int, float)) and not pd.isna(age):
        return int(age)

    birth_year = profile.get("birth_year")
    if isinstance(birth_year, (int, float)) and not pd.isna(birth_year):
        from datetime import date
        return date.today().year - int(birth_year)

    return None


class HousingPlanRepository:
    """실제 공고·서류 규칙·발급 가이드 CSV를 메모리에 로드합니다."""

    def __init__(self, data_dir: Path | str = DEFAULT_DATA_DIR) -> None:
        self.data_dir = Path(data_dir)

        required = [
            NOTICE_FILE,
            DOCUMENT_FILE,
            RULE_FILE,
            CHUNK_FILE,
        ]
        missing = [
            filename
            for filename in required
            if not (self.data_dir / filename).exists()
        ]
        if missing:
            raise ServicePlannerError(
                f"필수 데이터 파일이 없습니다: {missing}"
            )

        self.notices = pd.read_csv(
            self.data_dir / NOTICE_FILE,
            low_memory=False,
        )
        self.documents = pd.read_csv(
            self.data_dir / DOCUMENT_FILE,
            low_memory=False,
        )
        self.rules = pd.read_csv(
            self.data_dir / RULE_FILE,
            low_memory=False,
        )
        self.chunks = pd.read_csv(
            self.data_dir / CHUNK_FILE,
            low_memory=False,
        )

        self._normalize()

    def _normalize(self) -> None:
        for frame in (
            self.notices,
            self.rules,
            self.chunks,
        ):
            if "is_latest" in frame:
                frame["is_latest"] = frame["is_latest"].map(_safe_bool)

        for frame, columns in (
            (
                self.notices,
                [
                    "record_id",
                    "notice_id",
                    "announcement_title",
                    "housing_type",
                    "target_group",
                    "province",
                    "city_county_district",
                    "application_url",
                    "notice_url",
                    "source_file",
                ],
            ),
            (
                self.rules,
                [
                    "rule_id",
                    "notice_id",
                    "record_id",
                    "document_id",
                    "document_name",
                    "review_status",
                    "requirement_type",
                    "submission_stage",
                    "source_file",
                ],
            ),
            (
                self.documents,
                [
                    "document_id",
                    "canonical_name",
                    "issuer",
                    "portal_url",
                    "direct_url",
                ],
            ),
            (
                self.chunks,
                [
                    "chunk_id",
                    "notice_id",
                    "chunk_type",
                    "retrieval_text",
                    "source_file",
                    "review_status",
                ],
            ),
        ):
            for column in columns:
                if column in frame:
                    frame[column] = frame[column].fillna("").astype(str).str.strip()

    def find_notice_candidates(
        self,
        *,
        policy_id: str,
        user_profile: Any,
        limit: int = 5,
    ) -> dict[str, Any]:
        mapping = SERVICE_NOTICE_MAP.get(policy_id)
        if not mapping:
            return {
                "supported": False,
                "policy_id": policy_id,
                "relation_note": (
                    "현재 실제 공고 데이터에는 이 추천 서비스와 직접 연결할 수 있는 "
                    "주택공급 공고 유형이 없습니다."
                ),
                "candidates": [],
            }

        profile = _profile_to_dict(user_profile)
        user_sido = _safe_str(profile.get("sido"))
        user_sigungu = _safe_str(profile.get("sigungu"))
        user_age = _age_from_profile(profile)

        candidates = self.notices[
            self.notices["is_latest"]
            & self.notices["housing_type"].isin(mapping["housing_types"])
        ].copy()

        if mapping.get("target_keywords"):
            target_mask = candidates["target_group"].apply(
                lambda value: _contains_any(
                    value,
                    mapping["target_keywords"],
                )
            )
            candidates = candidates[target_mask].copy()

        if candidates.empty:
            return {
                "supported": True,
                "policy_id": policy_id,
                "relation_type": mapping["relation_type"],
                "relation_note": mapping["relation_note"],
                "candidates": [],
            }

        def score_row(row: pd.Series) -> float:
            score = 60.0

            provinces = _split_regions(row.get("province"))
            district_text = _safe_str(row.get("city_county_district"))

            if "전국" in provinces or _safe_str(row.get("province")) == "전국":
                score += 25.0
            elif user_sido and user_sido in provinces:
                score += 40.0
            else:
                score -= 15.0

            if user_sigungu:
                if user_sigungu in district_text:
                    score += 20.0
                elif "전국" in district_text:
                    score += 8.0
                elif "여러 지역" in district_text and user_sido in provinces:
                    score += 10.0

            age_min = row.get("age_min")
            age_max = row.get("age_max")
            if user_age is not None:
                has_min = pd.notna(age_min)
                has_max = pd.notna(age_max)
                if has_min and user_age < float(age_min):
                    score -= 50.0
                elif has_max and user_age > float(age_max):
                    score -= 50.0
                elif has_min or has_max:
                    score += 15.0

            target_group = _safe_str(row.get("target_group"))
            if "19~39세" in target_group or "청년·사회초년생" in target_group:
                score += 10.0
            elif "대학생" in target_group or "취업준비생" in target_group:
                score += 2.0

            if _safe_bool(row.get("needs_review")):
                score -= 3.0

            return round(score, 2)

        candidates["_match_score"] = candidates.apply(
            score_row,
            axis=1,
        )

        # 하나의 notice_id에 대상계층 행이 여러 개면 가장 적합한 record를 대표로 사용합니다.
        candidates = (
            candidates.sort_values(
                ["_match_score", "notice_id"],
                ascending=[False, True],
            )
            .drop_duplicates("notice_id", keep="first")
            .head(limit)
        )

        records: list[dict[str, Any]] = []
        for _, row in candidates.iterrows():
            records.append(
                {
                    "notice_id": _safe_str(row.get("notice_id")),
                    "record_id": _safe_str(row.get("record_id")),
                    "announcement_title": _safe_str(
                        row.get("announcement_title")
                    ),
                    "operator": _safe_str(row.get("operator")),
                    "housing_type": _safe_str(row.get("housing_type")),
                    "target_group": _safe_str(row.get("target_group")),
                    "supply_track": _safe_str(row.get("supply_track")),
                    "province": _safe_str(row.get("province")),
                    "city_county_district": _safe_str(
                        row.get("city_county_district")
                    ),
                    "application_method": _safe_str(
                        row.get("application_method")
                    ),
                    "application_url": _safe_str(
                        row.get("application_url")
                    ),
                    "notice_url": _safe_str(row.get("notice_url")),
                    "notice_status": _safe_str(
                        row.get("notice_status")
                    ),
                    "source_file": _safe_str(row.get("source_file")),
                    "verified_at": _safe_str(row.get("verified_at")),
                    "review_status": (
                        "미검수"
                        if _safe_bool(row.get("needs_review"))
                        else "검수 상태 확인"
                    ),
                    "match_score": float(row["_match_score"]),
                }
            )

        return {
            "supported": True,
            "policy_id": policy_id,
            "relation_type": mapping["relation_type"],
            "relation_note": mapping["relation_note"],
            "candidates": records,
        }

    def get_notice_summary(
        self,
        notice_id: str,
        record_id: str = "",
    ) -> dict[str, Any]:
        rows = self.notices[
            (self.notices["notice_id"] == notice_id)
            & self.notices["is_latest"]
        ]
        if rows.empty:
            raise ServicePlannerError(
                f"공고를 찾을 수 없습니다: {notice_id}"
            )

        selected = rows
        if record_id:
            matched = rows[rows["record_id"] == record_id]
            if not matched.empty:
                selected = matched

        row = selected.iloc[0]

        return {
            "notice_id": notice_id,
            "record_id": _safe_str(row.get("record_id")),
            "announcement_title": _safe_str(
                row.get("announcement_title")
            ),
            "operator": _safe_str(row.get("operator")),
            "housing_type": _safe_str(row.get("housing_type")),
            "target_group": _safe_str(row.get("target_group")),
            "supply_track": _safe_str(row.get("supply_track")),
            "province": _safe_str(row.get("province")),
            "city_county_district": _safe_str(
                row.get("city_county_district")
            ),
            "housing_ownership_condition": _safe_str(
                row.get("housing_ownership_condition")
            ),
            "income_rule_text": _safe_str(
                row.get("income_rule_text")
            ),
            "asset_rule_text": _safe_str(
                row.get("asset_rule_text")
            ),
            "application_method": _safe_str(
                row.get("application_method")
            ),
            "application_start": _safe_str(
                row.get("application_start")
            ),
            "application_end": _safe_str(
                row.get("application_end")
            ),
            "document_submission_method": _safe_str(
                row.get("document_submission_method")
            ),
            "application_url": _safe_str(
                row.get("application_url")
            ),
            "notice_url": _safe_str(row.get("notice_url")),
            "source_file": _safe_str(row.get("source_file")),
            "source_pages": _safe_str(row.get("source_pages")),
            "verified_at": _safe_str(row.get("verified_at")),
            "data_quality": _safe_str(row.get("data_quality")),
            "needs_review": _safe_bool(row.get("needs_review")),
            "notes": _safe_str(row.get("notes")),
        }

    def get_condition_questions(
        self,
        *,
        notice_id: str,
        record_id: str = "",
    ) -> list[dict[str, Any]]:
        rules = self.rules[
            (self.rules["notice_id"] == notice_id)
            & self.rules["is_latest"]
            & (self.rules["review_status"] != "제외")
        ].copy()

        if record_id:
            # record_id가 비어 있는 공통 규칙도 함께 유지합니다.
            rules = rules[
                (rules["record_id"] == "")
                | (rules["record_id"] == record_id)
            ]

        condition_keys: dict[str, set[str]] = {}
        for value in rules["condition_json"]:
            condition = _parse_json_dict(value)
            for key, expected in condition.items():
                condition_keys.setdefault(key, set()).add(
                    _safe_str(expected)
                )

        questions: list[dict[str, Any]] = []
        for key in sorted(condition_keys):
            config = CONDITION_QUESTIONS.get(
                key,
                {
                    "label": f"{key} 조건에 해당하나요?",
                    "kind": "boolean",
                },
            )
            question = {
                "condition_key": key,
                "label": config["label"],
                "kind": config["kind"],
                "expected_values": sorted(
                    value
                    for value in condition_keys[key]
                    if value
                ),
            }
            if config.get("options"):
                question["options"] = list(config["options"])
            questions.append(question)

        return questions

    def get_initial_condition_answers(
        self,
        user_profile: Any,
    ) -> dict[str, Any]:
        profile = _profile_to_dict(user_profile)
        residence_status = _safe_str(
            profile.get("residence_status")
        )

        return {
            "housing_vulnerable": (
                True
                if residence_status == "주거취약시설"
                else None
            ),
        }

    def _evaluate_condition(
        self,
        condition: Mapping[str, Any],
        answers: Mapping[str, Any],
    ) -> str:
        if not condition:
            return "common"

        has_unknown = False

        for key, expected in condition.items():
            answer = answers.get(key)

            if answer in (None, "", "모름"):
                has_unknown = True
                continue

            if isinstance(expected, bool):
                if bool(answer) is not expected:
                    return "not_applicable"
            else:
                if _safe_str(answer) != _safe_str(expected):
                    return "not_applicable"

        return "needs_answer" if has_unknown else "applicable"

    def build_document_plan(
        self,
        *,
        notice_id: str,
        record_id: str = "",
        condition_answers: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        answers = dict(condition_answers or {})

        rules = self.rules[
            (self.rules["notice_id"] == notice_id)
            & self.rules["is_latest"]
            & (self.rules["review_status"] != "제외")
        ].copy()

        if record_id:
            rules = rules[
                (rules["record_id"] == "")
                | (rules["record_id"] == record_id)
            ]

        if rules.empty:
            return {
                "notice_id": notice_id,
                "documents": [],
                "stage_groups": {},
                "unanswered_conditions": [],
                "review_warning": (
                    "이 공고에 연결된 서류 규칙을 찾지 못했습니다."
                ),
            }

        rows: list[dict[str, Any]] = []

        for _, rule in rules.iterrows():
            condition = _parse_json_dict(
                rule.get("condition_json")
            )
            requirement_type = _safe_str(
                rule.get("requirement_type")
            ).lower()
            if not condition and requirement_type in {
                "conditional", "조건부", "해당자", "applicable",
            }:
                # 원본 데이터가 '해당자' 여부만 제공하고 세부 조건을 제공하지 않는 경우,
                # 공통 필수로 오인하지 않고 '조건 확인 후 결정'로 유지합니다.
                evaluation = "needs_answer"
            else:
                evaluation = self._evaluate_condition(
                    condition,
                    answers,
                )
            if evaluation == "not_applicable":
                continue

            rows.append(
                {
                    "document_id": _safe_str(
                        rule.get("document_id")
                    ),
                    "document_name_rule": _safe_str(
                        rule.get("document_name")
                    ),
                    "evaluation": evaluation,
                    "condition": condition,
                    "requirement_type": _safe_str(
                        rule.get("requirement_type")
                    ),
                    "required_for": _safe_str(
                        rule.get("required_for")
                    ),
                    "trigger_reason": _safe_str(
                        rule.get("trigger_reason")
                    ),
                    "submission_stage": _safe_str(
                        rule.get("submission_stage")
                    ) or "제출 단계 확인 필요",
                    "submission_channel": _safe_str(
                        rule.get("submission_channel")
                    ),
                    "mymy_eligible": _safe_str(
                        rule.get("mymy_eligible")
                    ),
                    "source_file": _safe_str(
                        rule.get("source_file")
                    ),
                    "source_page": (
                        int(rule["source_page"])
                        if pd.notna(rule.get("source_page"))
                        else None
                    ),
                    "review_status": _safe_str(
                        rule.get("review_status")
                    ) or "미검수",
                    "original_requirement_text": _safe_str(
                        rule.get("original_requirement_text")
                    ),
                }
            )

        if not rows:
            return {
                "notice_id": notice_id,
                "documents": [],
                "stage_groups": {},
                "unanswered_conditions": [],
                "review_warning": (
                    "현재 답변 조건에서 표시할 서류가 없습니다."
                ),
            }

        rule_frame = pd.DataFrame(rows)
        merged = rule_frame.merge(
            self.documents,
            on="document_id",
            how="left",
            suffixes=("_rule", "_master"),
        )

        priority = {
            "common": 3,
            "applicable": 2,
            "needs_answer": 1,
        }

        documents: list[dict[str, Any]] = []
        unanswered: set[str] = set()

        for document_id, group in merged.groupby(
            "document_id",
            dropna=False,
        ):
            group = group.copy()
            group["_priority"] = group["evaluation"].map(
                priority
            )
            group = group.sort_values(
                "_priority",
                ascending=False,
            )
            best = group.iloc[0]

            evaluations = set(group["evaluation"])
            if "common" in evaluations:
                status = "common"
            elif "applicable" in evaluations:
                status = "applicable"
            else:
                status = "needs_answer"

            conditions: list[dict[str, Any]] = []
            source_references: list[dict[str, Any]] = []
            review_statuses: set[str] = set()

            for _, row in group.iterrows():
                condition = row.get("condition")
                if isinstance(condition, dict) and condition:
                    conditions.append(condition)
                    if row["evaluation"] == "needs_answer":
                        unanswered.update(condition.keys())

                reference = {
                    "source_file": _safe_str(
                        row.get("source_file")
                    ),
                    "source_page": (
                        int(row["source_page"])
                        if pd.notna(row.get("source_page"))
                        else None
                    ),
                    "review_status": _safe_str(
                        row.get("review_status")
                    ) or "미검수",
                    "original_requirement_text": _safe_str(
                        row.get("original_requirement_text")
                    ),
                }
                if reference not in source_references:
                    source_references.append(reference)

                review_statuses.add(
                    _safe_str(row.get("review_status"))
                    or "미검수"
                )

            all_verified = (
                review_statuses
                and review_statuses == {"검수완료"}
            )
            certainty_label = (
                "필수 또는 조건부 필수"
                if all_verified
                else "필요할 수 있음"
            )

            portal_url = _safe_str(best.get("portal_url"))
            direct_url = _safe_str(best.get("direct_url"))

            documents.append(
                {
                    "document_id": _safe_str(document_id),
                    "canonical_name": (
                        _safe_str(best.get("canonical_name"))
                        or _safe_str(best.get("document_name_rule"))
                    ),
                    "document_category": _safe_str(
                        best.get("document_category")
                    ),
                    "status": status,
                    "status_label": {
                        "common": "공통 준비 후보",
                        "applicable": "현재 답변에서 준비 후보",
                        "needs_answer": "조건 확인 후 결정",
                    }[status],
                    "certainty_label": certainty_label,
                    "issuer": _safe_str(best.get("issuer")),
                    "issue_method_online": _safe_str(
                        best.get("issue_method_online")
                    ),
                    "issue_method_offline": _safe_str(
                        best.get("issue_method_offline")
                    ),
                    "portal_url": portal_url,
                    "direct_url": direct_url,
                    "site_search_keyword": _safe_str(
                        best.get("site_search_keyword")
                    ),
                    "identity_required": _safe_str(
                        best.get("identity_required")
                    ),
                    "fee_note": _safe_str(best.get("fee_note")),
                    "processing_time": _safe_str(
                        best.get("processing_time")
                    ),
                    "required_options": _safe_str(
                        best.get("required_options")
                    ),
                    "output_format": _safe_str(
                        best.get("output_format")
                    ),
                    "validity_note": _safe_str(
                        best.get("validity_note")
                    ),
                    "cautions": _safe_str(
                        best.get("cautions")
                    ),
                    "privacy_warning": _safe_str(
                        best.get("privacy_warning")
                    ),
                    "submission_stages": sorted(
                        {
                            _safe_str(value)
                            for value in group["submission_stage"]
                            if _safe_str(value)
                        }
                    ),
                    "submission_channels": sorted(
                        {
                            _safe_str(value)
                            for value in group["submission_channel"]
                            if _safe_str(value)
                        }
                    ),
                    "mymy_statuses": sorted(
                        {
                            _safe_str(value)
                            for value in group["mymy_eligible"]
                            if _safe_str(value)
                        }
                    ),
                    "conditions": conditions,
                    "review_statuses": sorted(review_statuses),
                    "source_references": source_references[:5],
                }
            )

        status_order = {
            "common": 0,
            "applicable": 1,
            "needs_answer": 2,
        }
        documents.sort(
            key=lambda item: (
                status_order[item["status"]],
                item["canonical_name"],
            )
        )

        stage_groups: dict[str, list[str]] = {}
        for document in documents:
            for stage in (
                document["submission_stages"]
                or ["제출 단계 확인 필요"]
            ):
                stage_groups.setdefault(stage, []).append(
                    document["document_id"]
                )

        return {
            "notice_id": notice_id,
            "record_id": record_id,
            "documents": documents,
            "stage_groups": stage_groups,
            "unanswered_conditions": sorted(unanswered),
            "review_warning": (
                "현재 서류 규칙은 자동 추출 1차본입니다. "
                "검수완료가 아닌 항목은 확정 필수가 아니라 "
                "원문 확인이 필요한 준비 후보로 표시합니다."
            ),
        }

    def get_notice_guidance_chunks(
        self,
        *,
        notice_id: str,
        limit_per_type: int = 2,
    ) -> list[dict[str, Any]]:
        allowed_types = [
            "application_method",
            "submission_method",
            "documents_common",
            "documents_conditional",
            "caution",
            "schedule",
        ]

        chunks = self.chunks[
            (self.chunks["notice_id"] == notice_id)
            & self.chunks["is_latest"]
            & self.chunks["chunk_type"].isin(allowed_types)
            & (self.chunks["review_status"] != "제외")
        ].copy()

        if chunks.empty:
            return []

        selected: list[dict[str, Any]] = []
        for chunk_type in allowed_types:
            subset = chunks[
                chunks["chunk_type"] == chunk_type
            ].head(limit_per_type)

            for _, row in subset.iterrows():
                selected.append(
                    {
                        "chunk_id": _safe_str(
                            row.get("chunk_id")
                        ),
                        "chunk_type": _safe_str(
                            row.get("chunk_type")
                        ),
                        "title": _safe_str(
                            row.get("chunk_title")
                        ),
                        "text": _safe_str(
                            row.get("retrieval_text")
                        ),
                        "source_file": _safe_str(
                            row.get("source_file")
                        ),
                        "source_page_start": (
                            int(row["source_page_start"])
                            if pd.notna(
                                row.get("source_page_start")
                            )
                            else None
                        ),
                        "source_page_end": (
                            int(row["source_page_end"])
                            if pd.notna(
                                row.get("source_page_end")
                            )
                            else None
                        ),
                        "review_status": _safe_str(
                            row.get("review_status")
                        ) or "미검수",
                    }
                )

        return selected


class ServiceApplicationPlanner:
    """앱에서 사용할 고수준 신청 준비 계획 서비스."""

    def __init__(
        self,
        repository: HousingPlanRepository,
    ) -> None:
        self.repository = repository

    def find_related_notices(
        self,
        *,
        policy_id: str,
        user_profile: Any,
        limit: int = 5,
    ) -> dict[str, Any]:
        return self.repository.find_notice_candidates(
            policy_id=policy_id,
            user_profile=user_profile,
            limit=limit,
        )

    def build_plan(
        self,
        *,
        notice_id: str,
        record_id: str,
        user_profile: Any,
        condition_answers: Optional[
            Mapping[str, Any]
        ] = None,
    ) -> dict[str, Any]:
        summary = self.repository.get_notice_summary(
            notice_id,
            record_id,
        )
        questions = self.repository.get_condition_questions(
            notice_id=notice_id,
            record_id=record_id,
        )
        initial_answers = (
            self.repository.get_initial_condition_answers(
                user_profile
            )
        )
        merged_answers = {
            **initial_answers,
            **dict(condition_answers or {}),
        }
        document_plan = self.repository.build_document_plan(
            notice_id=notice_id,
            record_id=record_id,
            condition_answers=merged_answers,
        )
        guidance_chunks = (
            self.repository.get_notice_guidance_chunks(
                notice_id=notice_id
            )
        )

        return {
            "notice": summary,
            "questions": questions,
            "condition_answers": merged_answers,
            "document_plan": document_plan,
            "guidance_chunks": guidance_chunks,
            "steps": [
                {
                    "step": 1,
                    "title": "공고와 신청 기간 확인",
                    "description": (
                        "공고 제목, 접수 상태, 신청 페이지와 원문을 먼저 확인합니다."
                    ),
                },
                {
                    "step": 2,
                    "title": "추가 조건 답변",
                    "description": (
                        "대학생·취업준비생·대리신청 등 공고별 조건을 확인합니다."
                    ),
                },
                {
                    "step": 3,
                    "title": "공통 서류 발급",
                    "description": (
                        "등본, 가족관계증명서 등 공통 준비 후보부터 발급합니다."
                    ),
                },
                {
                    "step": 4,
                    "title": "조건부 서류 준비",
                    "description": (
                        "현재 답변과 일치하는 조건부 서류를 준비합니다."
                    ),
                },
                {
                    "step": 5,
                    "title": "제출 단계와 채널 확인",
                    "description": (
                        "청약신청 시, 서류대상자 선정 후, 계약 시 등 제출 시점을 구분합니다."
                    ),
                },
            ],
        }
