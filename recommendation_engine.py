# -*- coding: utf-8 -*-
"""
Policy Bridge 규칙 기반 추천 엔진

1. 나이·주택 소유·지역·거주 상태로 하드 필터링
2. 소득 적합도와 지역 정밀도로 추천 우선순위 계산
3. MatchResult 목록을 점수 내림차순으로 반환

현재 정책과 금액은 개발용 임시 데이터입니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import List, Optional
import math


MEDIAN_INCOME_2026 = {
    1: 2_564_238,
    2: 4_199_292,
    3: 5_359_036,
    4: 6_494_738,
    5: 7_556_719,
    6: 8_555_952,
}


def get_median_income(household_size: int) -> int:
    if household_size < 1:
        household_size = 1
    if household_size in MEDIAN_INCOME_2026:
        return MEDIAN_INCOME_2026[household_size]
    if household_size > 6:
        increase = MEDIAN_INCOME_2026[6] - MEDIAN_INCOME_2026[5]
        return MEDIAN_INCOME_2026[6] + increase * (household_size - 6)
    return MEDIAN_INCOME_2026[1]


RESIDENCE_STATUS_OPTIONS = [
    "부모님과 거주",
    "월세",
    "전세",
    "집 구하는 중",
    "고시원-기숙사",
    "주거취약시설",
    "기타",
]


@dataclass(frozen=True)
class UserProfile:
    sido: str
    sigungu: str
    birth_year: int
    owns_house: Optional[bool]
    residence_status: str
    household_size: int
    income_known: bool
    monthly_income: Optional[int] = None

    def __post_init__(self) -> None:
        current_year = date.today().year
        if not (1900 <= self.birth_year <= current_year):
            raise ValueError("출생연도가 올바르지 않습니다.")
        if self.household_size < 1:
            raise ValueError("가구원 수는 1명 이상이어야 합니다.")
        if self.residence_status not in RESIDENCE_STATUS_OPTIONS:
            raise ValueError(f"지원하지 않는 거주 상태입니다: {self.residence_status}")
        if self.income_known and self.monthly_income is None:
            raise ValueError("소득을 안다고 선택한 경우 월 소득이 필요합니다.")
        if self.monthly_income is not None and self.monthly_income < 0:
            raise ValueError("월 소득은 0원 이상이어야 합니다.")

    @property
    def age(self) -> int:
        return date.today().year - self.birth_year


@dataclass(frozen=True)
class Policy:
    policy_id: str
    name: str
    description: str
    min_age: int
    max_age: int
    regions: List[str]
    sigungu_focus: Optional[List[str]] = None
    require_no_house: bool = True
    eligible_residence: Optional[List[str]] = None
    income_limit_percent: Optional[float] = None
    income_limit_amount: Optional[int] = None
    apply_deep_link: str = ""


@dataclass(frozen=True)
class MatchResult:
    policy: Policy
    role_base_score: float
    income_note: str


POLICY_DB: List[Policy] = [
    Policy(
        policy_id="RENT_SUPPORT",
        name="청년월세 특별지원",
        description="무주택 청년의 월세 부담을 줄이기 위한 지원 정책입니다.",
        min_age=19,
        max_age=39,
        regions=["전국"],
        eligible_residence=["월세", "고시원-기숙사"],
        income_limit_percent=60,
        apply_deep_link="https://www.gov.kr",
    ),
    Policy(
        policy_id="JEONSE_LOAN",
        name="청년전용 버팀목전세자금대출",
        description="무주택 청년의 전세보증금 마련을 돕는 저리 대출 정책입니다.",
        min_age=19,
        max_age=34,
        regions=["전국"],
        eligible_residence=[
            "월세", "전세", "집 구하는 중", "고시원-기숙사", "주거취약시설", "기타"
        ],
        income_limit_amount=5_000_000,
        apply_deep_link="https://nhuf.molit.go.kr",
    ),
    Policy(
        policy_id="LH_MAEIP",
        name="LH 청년 매입임대주택",
        description="LH가 매입한 주택을 청년에게 비교적 낮은 임대료로 공급합니다.",
        min_age=19,
        max_age=39,
        regions=["전국"],
        eligible_residence=["고시원-기숙사", "기타", "월세"],
        income_limit_percent=100,
        apply_deep_link="https://apply.lh.or.kr",
    ),
    Policy(
        policy_id="HAPPY_HOUSE",
        name="행복주택 청년계층",
        description="청년에게 시세보다 낮은 조건으로 공급되는 공공임대주택입니다.",
        min_age=19,
        max_age=39,
        regions=["전국"],
        eligible_residence=None,
        income_limit_percent=100,
        apply_deep_link="https://www.myhome.go.kr",
    ),
    Policy(
        policy_id="YOUTH_SAVINGS_JEONGYAK",
        name="청년 우대형 주택청약종합저축",
        description="청년의 내 집 마련을 돕는 청약·금융 상품입니다.",
        min_age=19,
        max_age=34,
        regions=["전국"],
        eligible_residence=None,
        income_limit_amount=3_000_000,
        apply_deep_link="https://www.gov.kr",
    ),
    Policy(
        policy_id="VULNERABLE_HOUSING_RELOCATION",
        name="주거취약계층 주거지원사업",
        description="고시원·쪽방 등 비적정 주거지 거주자의 이주를 돕는 정책입니다.",
        min_age=0,
        max_age=120,
        regions=["전국"],
        eligible_residence=["고시원-기숙사", "주거취약시설"],
        income_limit_percent=60,
        apply_deep_link="https://apply.lh.or.kr",
    ),
]


def passes_hard_filter(user: UserProfile, policy: Policy) -> bool:
    if not (policy.min_age <= user.age <= policy.max_age):
        return False
    if policy.require_no_house and user.owns_house is True:
        return False
    if policy.regions != ["전국"] and user.sido not in policy.regions:
        return False
    if (
        policy.eligible_residence is not None
        and user.residence_status not in policy.eligible_residence
    ):
        return False
    return True


def _income_fit_score(user: UserProfile, policy: Policy) -> tuple[float, str]:
    if not user.income_known:
        return 31.6, "월 소득이 입력되지 않아 소득·재산 기준을 추가로 확인해야 합니다."

    income = user.monthly_income or 0

    if policy.income_limit_amount is not None:
        limit = policy.income_limit_amount
        if income <= limit:
            margin = (limit - income) / max(limit, 1)
            return (
                10.5 + 73.7 * min(margin, 1.0),
                "입력한 월 소득이 임시 소득 기준 안에 있습니다.",
            )
        overage = (income - limit) / max(limit, 1)
        return (
            max(0.0, 10.5 - overage * 21.1),
            "입력한 월 소득이 임시 기준을 초과할 가능성이 있습니다.",
        )

    if policy.income_limit_percent is not None:
        median = get_median_income(user.household_size)
        income_percent = income / median * 100
        limit = policy.income_limit_percent

        if income_percent <= limit:
            margin = (limit - income_percent) / max(limit, 1)
            return (
                10.5 + 73.7 * min(margin, 1.0),
                f"입력 소득은 기준 중위소득 약 {income_percent:.0f}%로 계산됩니다.",
            )

        overage = (income_percent - limit) / max(limit, 1)
        return (
            max(0.0, 10.5 - overage * 21.1),
            f"입력 소득은 기준 중위소득 약 {income_percent:.0f}%이며 정책 기준을 추가 확인해야 합니다.",
        )

    return 63.2, "현재 임시 데이터에는 별도의 소득 기준이 없습니다."


SIDO_CENTROIDS = {
    "서울특별시": (37.5665, 126.9780),
    "부산광역시": (35.1796, 129.0756),
    "대구광역시": (35.8714, 128.6014),
    "인천광역시": (37.4563, 126.7052),
    "광주광역시": (35.1595, 126.8526),
    "대전광역시": (36.3504, 127.3845),
    "울산광역시": (35.5384, 129.3114),
    "세종특별자치시": (36.4800, 127.2890),
    "경기도": (37.4138, 127.5183),
    "강원특별자치도": (37.8228, 128.1555),
    "충청북도": (36.6357, 127.4917),
    "충청남도": (36.5184, 126.8000),
    "전북특별자치도": (35.7175, 127.1530),
    "전라남도": (34.8161, 126.4630),
    "경상북도": (36.4919, 128.8889),
    "경상남도": (35.4606, 128.2132),
    "제주특별자치도": (33.4996, 126.5312),
}

KOREA_CENTER = (36.4800, 127.2890)
MAX_DISTANCE_KM = 500.0


def _haversine_km(
    point1: tuple[float, float],
    point2: tuple[float, float],
) -> float:
    lat1, lon1 = map(math.radians, point1)
    lat2, lon2 = map(math.radians, point2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    value = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 2 * 6371.0 * math.asin(math.sqrt(value))


def _region_precision_score(user: UserProfile, policy: Policy) -> float:
    user_point = SIDO_CENTROIDS.get(user.sido, KOREA_CENTER)

    if policy.sigungu_focus and user.sigungu in policy.sigungu_focus:
        distance = 0.0
    elif policy.regions == ["전국"]:
        distance = _haversine_km(user_point, KOREA_CENTER)
    else:
        target_points = [
            SIDO_CENTROIDS[region]
            for region in policy.regions
            if region in SIDO_CENTROIDS
        ]
        if not target_points:
            distance = _haversine_km(user_point, KOREA_CENTER)
        else:
            target_center = (
                sum(point[0] for point in target_points) / len(target_points),
                sum(point[1] for point in target_points) / len(target_points),
            )
            distance = _haversine_km(user_point, target_center)

    proximity = max(0.0, 1.0 - distance / MAX_DISTANCE_KM)
    return round(15.8 * proximity, 2)


def calculate_match_score(user: UserProfile, policy: Policy) -> MatchResult:
    income_score, income_note = _income_fit_score(user, policy)
    region_score = _region_precision_score(user, policy)
    total = round(min(income_score + region_score, 100.0), 1)
    return MatchResult(policy=policy, role_base_score=total, income_note=income_note)


def recommend_policies(
    user: UserProfile,
    policy_db: List[Policy] = POLICY_DB,
) -> List[MatchResult]:
    candidates = [policy for policy in policy_db if passes_hard_filter(user, policy)]
    results = [calculate_match_score(user, policy) for policy in candidates]
    results.sort(key=lambda result: result.role_base_score, reverse=True)
    return results


if __name__ == "__main__":
    demo_user = UserProfile(
        sido="경기도",
        sigungu="수원시",
        birth_year=2002,
        owns_house=False,
        residence_status="월세",
        household_size=1,
        income_known=False,
        monthly_income=None,
    )
    for rank, result in enumerate(recommend_policies(demo_user), start=1):
        print(rank, result.policy.policy_id, result.role_base_score, result.income_note)
