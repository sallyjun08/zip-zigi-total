# -*- coding: utf-8 -*-
"""
Policy Bridge 원본 5종 CSV -> 운영용 4종 CSV 변환기
==================================================

원본 파일은 읽기 전용으로 사용하며 수정/이동/이름 변경하지 않는다.
변환 결과는 별도의 output 디렉터리에 생성한다.

원본 5종
- 공고_모음.csv
- 소득.csv
- 신청자격_판정_로직.csv
- 용어_사전.csv
- 행정_서류_발급.csv

운영용 4종
- housing_supply_notice_master.csv
- housing_rag_chunks_review_ready.csv
- housing_document_master.csv
- housing_notice_document_rules_review_ready.csv
"""

from __future__ import annotations

import argparse
import csv
from datetime import date, datetime
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Iterable, Mapping, Sequence

RAW_FILES = {
    "notices": "공고_모음.csv",
    "income": "소득.csv",
    "eligibility": "신청자격_판정_로직.csv",
    "terms": "용어_사전.csv",
    "documents": "행정_서류_발급.csv",
}

OUTPUT_FILES = {
    "notices": "housing_supply_notice_master.csv",
    "chunks": "housing_rag_chunks_review_ready.csv",
    "documents": "housing_document_master.csv",
    "rules": "housing_notice_document_rules_review_ready.csv",
}

NOTICE_FIELDS = [
    "record_id", "notice_id", "announcement_title", "operator",
    "housing_type", "supply_family", "target_group", "supply_track",
    "province", "city_county_district", "age_min", "age_max",
    "age_requirement_text", "housing_ownership_condition",
    "income_rule_text", "asset_rule_text", "application_method",
    "application_start", "application_end", "application_period_text",
    "document_submission_method", "application_url", "notice_url",
    "contact", "rent_summary", "selection_result_note",
    "income_rule_type", "source_file", "source_pages", "verified_at",
    "data_quality", "notice_status", "needs_review", "notes", "is_latest",
]

CHUNK_FIELDS = [
    "chunk_id", "notice_id", "record_id", "chunk_type", "chunk_title",
    "retrieval_text", "source_name", "source_url", "source_file",
    "source_page_start", "source_page_end", "review_status", "is_latest",
    "announcement_title", "housing_type", "target_group", "supply_track",
    "province", "city_county_district",
]

DOCUMENT_FIELDS = [
    "document_id", "canonical_name", "document_category", "issuer",
    "issue_method_online", "issue_method_offline", "portal_url", "direct_url",
    "site_search_keyword", "identity_required", "fee_note", "processing_time",
    "required_options", "output_format", "validity_note", "cautions",
    "privacy_warning",
]

RULE_FIELDS = [
    "rule_id", "notice_id", "record_id", "document_id", "document_name",
    "requirement_type", "required_for", "trigger_reason", "condition_json",
    "submission_stage", "submission_channel", "mymy_eligible", "source_file",
    "source_page", "review_status", "original_requirement_text", "is_latest",
]


def _safe(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _clean_text(value: Any) -> str:
    return " ".join(_safe(value).replace("\x00", " ").split())


def _truthy(value: Any) -> bool:
    return _safe(value).lower() in {"true", "1", "y", "yes", "예", "t"}


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> tuple[list[dict[str, str]], str]:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    raise ValueError("CSV 헤더가 없습니다.")
                fieldnames = [str(name).lstrip("\ufeff").strip() for name in reader.fieldnames]
                rows: list[dict[str, str]] = []
                for raw in reader:
                    row = {
                        fieldnames[index]: _safe(raw.get(original_name, ""))
                        for index, original_name in enumerate(reader.fieldnames)
                    }
                    rows.append(row)
                return rows, encoding
        except UnicodeDecodeError as exc:
            last_error = exc
    raise RuntimeError(f"CSV 인코딩을 읽지 못했습니다: {path}: {last_error}")


def _require_columns(name: str, rows: Sequence[Mapping[str, Any]], required: set[str]) -> None:
    if not rows:
        raise RuntimeError(f"{name}: 데이터 행이 없습니다.")
    missing = required - set(rows[0].keys())
    if missing:
        raise RuntimeError(f"{name}: 필수 컬럼 누락: {sorted(missing)}")


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    temp_path.replace(path)


def _format_won(value: Any) -> str:
    text = _safe(value)
    if not text:
        return ""
    try:
        number = int(float(text))
    except ValueError:
        return text
    return f"{number:,}원"


def _parse_period(value: str) -> tuple[str, str]:
    text = _clean_text(value)
    if not text:
        return "", ""
    matches = re.findall(r"20\d{2}[./-]\d{1,2}[./-]\d{1,2}(?:\s+\d{1,2}:\d{2})?", text)
    if not matches:
        return "", ""
    return matches[0], matches[1] if len(matches) >= 2 else ""


def _income_text(row: Mapping[str, Any]) -> str:
    parts: list[str] = []
    if _safe(row.get("판정방식")):
        parts.append(f"판정방식: {_clean_text(row.get('판정방식'))}")
    if _safe(row.get("적용기준")):
        parts.append(f"적용기준: {_clean_text(row.get('적용기준'))}")
    limits: list[str] = []
    for size in range(1, 9):
        value = _format_won(row.get(f"상한_{size}인"))
        if value:
            limits.append(f"{size}인 가구 {value}")
    if limits:
        parts.append("월평균소득 상한: " + ", ".join(limits))
    if _safe(row.get("비고")):
        parts.append(f"비고: {_clean_text(row.get('비고'))}")
    return ". ".join(parts)


def _asset_text(row: Mapping[str, Any]) -> str:
    parts: list[str] = []
    total = _format_won(row.get("asset_total_krw"))
    vehicle = _format_won(row.get("asset_vehicle_krw"))
    if total:
        parts.append(f"총자산 기준 {total}")
    if vehicle:
        parts.append(f"자동차가액 기준 {vehicle}")
    welfare = _clean_text(row.get("welfare_requirement"))
    if welfare:
        parts.append(f"복지자격: {welfare}")
    return ". ".join(parts)


def _split_how(value: str) -> tuple[str, str]:
    text = _clean_text(value)
    if not text:
        return "", ""
    parts = [part.strip() for part in re.split(r"\s+/\s+", text, maxsplit=1)]
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[1]


def _infer_issuer(name: str, how: str, url: str) -> str:
    text = f"{name} {how} {url}"
    mapping = [
        (("efamily.scourt.go.kr", "가족관계", "혼인관계", "입양관계"), "대법원 전자가족관계등록시스템"),
        (("nhis.or.kr", "건강보험"), "국민건강보험공단"),
        (("nps.or.kr", "국민연금"), "국민연금공단"),
        (("hometax", "소득금액증명", "사실증명"), "국세청 홈택스"),
        (("gov.kr", "plus.gov.kr", "정부24", "주민등록"), "정부24·행정복지센터"),
        (("대학교", "대학", "재학", "졸업", "제적"), "학교 또는 교육기관"),
        (("은행", "청약통장"), "금융기관"),
    ]
    for keywords, issuer in mapping:
        if any(keyword in text for keyword in keywords):
            return issuer
    return "해당 발급기관"


def _document_category(name: str) -> str:
    categories = [
        (("주민등록", "가족관계", "혼인관계", "입양관계"), "신원·가족관계"),
        (("소득", "세금", "건강보험", "국민연금", "급여", "재직"), "소득·근로"),
        (("수급", "차상위", "한부모", "장애", "국가유공"), "복지자격"),
        (("재학", "졸업", "제적", "학생"), "학업·청년자격"),
        (("임대차", "주택", "거주"), "주거·계약"),
        (("위임장", "인감", "신분증"), "본인확인·대리신청"),
    ]
    for keywords, category in categories:
        if any(keyword in name for keyword in keywords):
            return category
    return "기타 제출서류"


def _infer_condition(name: str, tip: str) -> dict[str, Any]:
    text = f"{name} {tip}"
    mappings: list[tuple[tuple[str, ...], str, Any]] = [
        (("재학증명", "학생증"), "applicant_type_student", True),
        (("졸업", "제적", "중퇴"), "applicant_type_jobseeker", True),
        (("한부모"), "applicant_type_hanparent", True),
        (("임신"), "pregnancy_claim", True),
        (("입양"), "adopted_child", True),
        (("외국인", "국내거소"), "foreign_family_member", True),
        (("위임장", "인감증명"), "proxy_application", True),
        (("청약통장", "청약저축"), "subscription_account_evidence", True),
        (("재직", "근로", "건강보험 자격득실", "국민연금 가입"), "employment_evidence_required", True),
    ]
    for keywords, key, expected in mappings:
        if isinstance(keywords, str):
            keywords = (keywords,)
        if any(keyword in text for keyword in keywords):
            return {key: expected}
    return {}


def _notice_metadata(notice: Mapping[str, Any], eligibility: Mapping[str, Any]) -> dict[str, str]:
    return {
        "announcement_title": _clean_text(notice.get("title")),
        "housing_type": _clean_text(notice.get("housing_type")),
        "target_group": _clean_text(eligibility.get("target_group")),
        "supply_track": _clean_text(eligibility.get("target_group")),
        "province": _clean_text(eligibility.get("province") or notice.get("province")),
        "city_county_district": _clean_text(eligibility.get("city") or notice.get("city")),
    }


def convert(source_dir: Path, output_dir: Path) -> dict[str, Any]:
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    raw_paths = {key: source_dir / filename for key, filename in RAW_FILES.items()}
    missing = [str(path) for path in raw_paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError("원본 파일이 없습니다:\n- " + "\n- ".join(missing))

    before_hashes = {key: _sha256(path) for key, path in raw_paths.items()}
    loaded: dict[str, list[dict[str, str]]] = {}
    encodings: dict[str, str] = {}
    for key, path in raw_paths.items():
        rows, encoding = _read_csv(path)
        loaded[key] = rows
        encodings[key] = encoding

    _require_columns("공고_모음.csv", loaded["notices"], {
        "notice_id", "title", "operator", "housing_type", "province", "city",
        "rent_summary", "apply_method", "apply_period", "result_note", "contact",
        "income_rule_type",
    })
    _require_columns("소득.csv", loaded["income"], {
        "rule_id", "notice_id", "target_group", "판정방식", "적용기준", "비고",
    })
    _require_columns("신청자격_판정_로직.csv", loaded["eligibility"], {
        "rule_id", "notice_id", "age_min", "age_max", "age_req_text", "ownership",
        "housing size", "special_note", "welfare_requirement", "asset_total_krw",
        "asset_vehicle_krw", "target_group", "needs_review_age",
    })
    _require_columns("용어_사전.csv", loaded["terms"], {
        "term_id", "term", "definition", "source_url",
    })
    _require_columns("행정_서류_발급.csv", loaded["documents"], {
        "notice_id", "doc_id", "name", "req", "how", "url", "tip",
    })

    notice_by_id = {row["notice_id"]: row for row in loaded["notices"]}
    income_by_rule = {row["rule_id"]: row for row in loaded["income"]}
    eligibility_by_rule = {row["rule_id"]: row for row in loaded["eligibility"]}

    notice_ids = set(notice_by_id)
    for table_name, rows in (("소득", loaded["income"]), ("신청자격", loaded["eligibility"]), ("서류", loaded["documents"])):
        unknown = sorted({_safe(row.get("notice_id")) for row in rows} - notice_ids)
        if unknown:
            raise RuntimeError(f"{table_name} 데이터에 공고 마스터에 없는 notice_id가 있습니다: {unknown[:10]}")
    if set(income_by_rule) != set(eligibility_by_rule):
        raise RuntimeError("소득.csv와 신청자격_판정_로직.csv의 rule_id 집합이 일치하지 않습니다.")

    generated_at = date.today().isoformat()
    notice_rows: list[dict[str, Any]] = []
    for rule_id in sorted(eligibility_by_rule):
        eligibility = eligibility_by_rule[rule_id]
        income = income_by_rule[rule_id]
        notice = notice_by_id[eligibility["notice_id"]]
        start, end = _parse_period(notice.get("apply_period", ""))
        income_rule_text = _income_text(income)
        asset_rule_text = _asset_text(eligibility)
        notes = ". ".join(filter(None, [
            _clean_text(eligibility.get("special_note")),
            _clean_text(notice.get("result_note")),
            "공고 원문 URL은 원본 데이터에 포함되지 않아 별도 공식 확인이 필요합니다.",
        ]))
        notice_rows.append({
            "record_id": rule_id,
            "notice_id": eligibility["notice_id"],
            "announcement_title": _clean_text(notice.get("title")),
            "operator": _clean_text(notice.get("operator")),
            "housing_type": _clean_text(notice.get("housing_type")),
            "supply_family": _clean_text(notice.get("housing_type")),
            "target_group": _clean_text(eligibility.get("target_group") or income.get("target_group")),
            "supply_track": _clean_text(eligibility.get("housing size") or eligibility.get("target_group")),
            "province": _clean_text(eligibility.get("province") or notice.get("province")),
            "city_county_district": _clean_text(eligibility.get("city") or notice.get("city")),
            "age_min": _safe(eligibility.get("age_min")),
            "age_max": _safe(eligibility.get("age_max")),
            "age_requirement_text": _clean_text(eligibility.get("age_req_text")),
            "housing_ownership_condition": _clean_text(eligibility.get("ownership")),
            "income_rule_text": income_rule_text,
            "asset_rule_text": asset_rule_text,
            "application_method": _clean_text(notice.get("apply_method")),
            "application_start": start,
            "application_end": end,
            "application_period_text": _clean_text(notice.get("apply_period")),
            "document_submission_method": _clean_text(notice.get("apply_method")),
            "application_url": "",
            "notice_url": "",
            "contact": _clean_text(notice.get("contact")),
            "rent_summary": _clean_text(notice.get("rent_summary")),
            "selection_result_note": _clean_text(notice.get("result_note")),
            "income_rule_type": _clean_text(notice.get("income_rule_type")),
            "source_file": "공고_모음.csv | 신청자격_판정_로직.csv | 소득.csv",
            "source_pages": "",
            "verified_at": generated_at,
            "data_quality": "원본 5종 CSV 구조화 변환",
            "notice_status": "원본 데이터 기반",
            "needs_review": "TRUE" if _truthy(eligibility.get("needs_review_age")) else "FALSE",
            "notes": notes,
            "is_latest": "TRUE",
        })

    document_groups: dict[str, dict[str, str]] = {}
    for row in loaded["documents"]:
        doc_id = _safe(row.get("doc_id"))
        if not doc_id:
            continue
        if doc_id in document_groups:
            existing = document_groups[doc_id]
            for key in ("name", "how", "url"):
                if _safe(existing.get(key)) != _safe(row.get(key)):
                    raise RuntimeError(f"동일 doc_id의 기본 정보가 다릅니다: {doc_id} / {key}")
        else:
            document_groups[doc_id] = row

    document_master_rows: list[dict[str, Any]] = []
    for doc_id in sorted(document_groups):
        row = document_groups[doc_id]
        name = _clean_text(row.get("name"))
        how = _clean_text(row.get("how"))
        url = _safe(row.get("url"))
        online, offline = _split_how(how)
        document_master_rows.append({
            "document_id": doc_id,
            "canonical_name": name,
            "document_category": _document_category(name),
            "issuer": _infer_issuer(name, how, url),
            "issue_method_online": online,
            "issue_method_offline": offline,
            "portal_url": url,
            "direct_url": url,
            "site_search_keyword": name,
            "identity_required": "온라인 발급 시 본인인증이 필요할 수 있습니다.",
            "fee_note": "발급 채널에 따라 수수료를 확인하세요.",
            "processing_time": "온라인 즉시 발급 가능 여부를 공식 포털에서 확인하세요.",
            "required_options": "공고에서 요구한 상세·주민번호 공개·주소변동 등 표시 옵션을 확인하세요.",
            "output_format": "전자문서 또는 출력물",
            "validity_note": "공고일 이후 발급분 등 유효기간 조건을 원문에서 확인하세요.",
            "cautions": "공고별 제출 대상과 발급 옵션이 다를 수 있습니다.",
            "privacy_warning": "주민등록번호 등 개인정보가 포함된 문서는 제출처와 보관에 주의하세요.",
        })

    rule_rows: list[dict[str, Any]] = []
    for index, row in enumerate(loaded["documents"], start=1):
        req = _clean_text(row.get("req"))
        condition = _infer_condition(_clean_text(row.get("name")), _clean_text(row.get("tip"))) if req != "공통" else {}
        notice = notice_by_id[row["notice_id"]]
        rule_rows.append({
            "rule_id": f"{row['notice_id']}-DOC-{index:04d}",
            "notice_id": row["notice_id"],
            "record_id": "",
            "document_id": row["doc_id"],
            "document_name": _clean_text(row.get("name")),
            "requirement_type": "common" if req == "공통" else "conditional",
            "required_for": req or "확인 필요",
            "trigger_reason": "원본 데이터에서 공통으로 표시" if req == "공통" else "원본 데이터에서 해당자 서류로 표시",
            "condition_json": json.dumps(condition, ensure_ascii=False, separators=(",", ":")),
            "submission_stage": "신청·서류제출 단계",
            "submission_channel": _clean_text(notice.get("apply_method")),
            "mymy_eligible": "확인 필요",
            "source_file": "행정_서류_발급.csv",
            "source_page": "",
            "review_status": "원본구조화",
            "original_requirement_text": _clean_text(row.get("tip")),
            "is_latest": "TRUE",
        })

    chunk_rows: list[dict[str, Any]] = []
    chunk_ids: set[str] = set()

    def add_chunk(
        *,
        chunk_id: str,
        notice_id: str,
        record_id: str,
        chunk_type: str,
        chunk_title: str,
        text: str,
        source_file: str,
        source_name: str,
        source_url: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        normalized_text = _clean_text(text)
        if not normalized_text:
            return
        if chunk_id in chunk_ids:
            raise RuntimeError(f"중복 chunk_id 생성: {chunk_id}")
        chunk_ids.add(chunk_id)
        meta = dict(metadata or {})
        chunk_rows.append({
            "chunk_id": chunk_id,
            "notice_id": notice_id,
            "record_id": record_id,
            "chunk_type": chunk_type,
            "chunk_title": chunk_title,
            "retrieval_text": normalized_text,
            "source_name": source_name,
            "source_url": source_url,
            "source_file": source_file,
            "source_page_start": "",
            "source_page_end": "",
            "review_status": "원본구조화",
            "is_latest": "TRUE",
            "announcement_title": meta.get("announcement_title", ""),
            "housing_type": meta.get("housing_type", ""),
            "target_group": meta.get("target_group", ""),
            "supply_track": meta.get("supply_track", ""),
            "province": meta.get("province", ""),
            "city_county_district": meta.get("city_county_district", ""),
        })

    for notice_id in sorted(notice_by_id):
        notice = notice_by_id[notice_id]
        first_eligibility = next(row for row in loaded["eligibility"] if row["notice_id"] == notice_id)
        meta = _notice_metadata(notice, first_eligibility)
        base_prefix = notice_id
        add_chunk(
            chunk_id=f"{base_prefix}-benefit",
            notice_id=notice_id,
            record_id="",
            chunk_type="rent_conditions",
            chunk_title="임대조건과 지원 내용",
            text=f"공고명: {notice.get('title')}. 주택유형: {notice.get('housing_type')}. 임대조건·지원내용: {notice.get('rent_summary')}",
            source_file="공고_모음.csv",
            source_name="공고 모음 원본 데이터",
            metadata=meta,
        )
        add_chunk(
            chunk_id=f"{base_prefix}-application",
            notice_id=notice_id,
            record_id="",
            chunk_type="application_method",
            chunk_title="신청 방법",
            text=f"공고명: {notice.get('title')}. 신청방법: {notice.get('apply_method')}",
            source_file="공고_모음.csv",
            source_name="공고 모음 원본 데이터",
            metadata=meta,
        )
        add_chunk(
            chunk_id=f"{base_prefix}-schedule",
            notice_id=notice_id,
            record_id="",
            chunk_type="schedule",
            chunk_title="신청 기간과 결과 안내",
            text=f"신청기간: {notice.get('apply_period')}. 선정·결과 안내: {notice.get('result_note')}",
            source_file="공고_모음.csv",
            source_name="공고 모음 원본 데이터",
            metadata=meta,
        )
        add_chunk(
            chunk_id=f"{base_prefix}-contact",
            notice_id=notice_id,
            record_id="",
            chunk_type="contact",
            chunk_title="문의처",
            text=f"문의처: {notice.get('contact')}",
            source_file="공고_모음.csv",
            source_name="공고 모음 원본 데이터",
            metadata=meta,
        )

    for rule_id in sorted(eligibility_by_rule):
        eligibility = eligibility_by_rule[rule_id]
        income = income_by_rule[rule_id]
        notice = notice_by_id[eligibility["notice_id"]]
        meta = _notice_metadata(notice, eligibility)
        age_text = _clean_text(eligibility.get("age_req_text"))
        eligibility_text = ". ".join(filter(None, [
            f"공고명: {_clean_text(notice.get('title'))}",
            f"대상계층: {_clean_text(eligibility.get('target_group'))}",
            f"지역: {_clean_text(eligibility.get('province'))} {_clean_text(eligibility.get('city'))}",
            f"연령조건: {age_text}" if age_text else "",
            f"주택소유조건: {_clean_text(eligibility.get('ownership'))}",
            f"공급형: {_clean_text(eligibility.get('housing size'))}",
            f"소득판정 대상: {_clean_text(eligibility.get('income_subject'))}" if _safe(eligibility.get("income_subject")) else "",
            f"복지자격조건: {_clean_text(eligibility.get('welfare_requirement'))}" if _safe(eligibility.get("welfare_requirement")) else "",
        ]))
        add_chunk(
            chunk_id=f"{rule_id}-eligibility",
            notice_id=eligibility["notice_id"],
            record_id=rule_id,
            chunk_type="eligibility_common",
            chunk_title="신청 자격",
            text=eligibility_text,
            source_file="신청자격_판정_로직.csv",
            source_name="신청자격 판정 원본 데이터",
            metadata=meta,
        )
        add_chunk(
            chunk_id=f"{rule_id}-income",
            notice_id=eligibility["notice_id"],
            record_id=rule_id,
            chunk_type="income_asset",
            chunk_title="소득·자산 조건",
            text=". ".join(filter(None, [
                f"대상계층: {_clean_text(eligibility.get('target_group'))}",
                _income_text(income),
                _asset_text(eligibility),
            ])),
            source_file="소득.csv | 신청자격_판정_로직.csv",
            source_name="소득·자산 원본 데이터",
            metadata=meta,
        )
        caution_text = ". ".join(filter(None, [
            _clean_text(eligibility.get("special_note")),
            "연령 조건 원문 재확인 필요" if _truthy(eligibility.get("needs_review_age")) else "",
            _clean_text(income.get("비고")),
        ]))
        add_chunk(
            chunk_id=f"{rule_id}-caution",
            notice_id=eligibility["notice_id"],
            record_id=rule_id,
            chunk_type="caution",
            chunk_title="추가 확인 사항",
            text=caution_text,
            source_file="신청자격_판정_로직.csv | 소득.csv",
            source_name="자격·소득 원본 데이터",
            metadata=meta,
        )

    docs_by_notice: dict[str, list[dict[str, str]]] = {}
    for row in loaded["documents"]:
        docs_by_notice.setdefault(row["notice_id"], []).append(row)
        notice = notice_by_id[row["notice_id"]]
        first_eligibility = next(item for item in loaded["eligibility"] if item["notice_id"] == row["notice_id"])
        meta = _notice_metadata(notice, first_eligibility)
        chunk_type = "documents_common" if _clean_text(row.get("req")) == "공통" else "documents_conditional"
        add_chunk(
            chunk_id=f"{row['notice_id']}-doc-{row['doc_id']}",
            notice_id=row["notice_id"],
            record_id="",
            chunk_type=chunk_type,
            chunk_title=f"제출서류: {_clean_text(row.get('name'))}",
            text=". ".join(filter(None, [
                f"서류명: {_clean_text(row.get('name'))}",
                f"구분: {_clean_text(row.get('req'))}",
                f"발급방법: {_clean_text(row.get('how'))}",
                f"공고 요구사항: {_clean_text(row.get('tip'))}",
            ])),
            source_file="행정_서류_발급.csv",
            source_name="행정 서류 발급 원본 데이터",
            source_url=_safe(row.get("url")),
            metadata=meta,
        )

    # 용어사전은 공고·자격·소득·서류 텍스트에 실제로 등장하는 용어만 notice별로 연결한다.
    terms_by_term: dict[str, dict[str, str]] = {}
    for row in loaded["terms"]:
        term = _clean_text(row.get("term"))
        if len(term) < 2:
            continue
        current = terms_by_term.get(term)
        if current is None or len(_safe(row.get("definition"))) > len(_safe(current.get("definition"))):
            terms_by_term[term] = row

    combined_by_notice: dict[str, str] = {notice_id: "" for notice_id in notice_by_id}
    for table_key in ("notices", "income", "eligibility", "documents"):
        for row in loaded[table_key]:
            notice_id = _safe(row.get("notice_id"))
            if notice_id in combined_by_notice:
                combined_by_notice[notice_id] += " " + " ".join(_safe(value) for value in row.values())

    for notice_id, combined_text in combined_by_notice.items():
        notice = notice_by_id[notice_id]
        first_eligibility = next(item for item in loaded["eligibility"] if item["notice_id"] == notice_id)
        meta = _notice_metadata(notice, first_eligibility)
        for term, row in sorted(terms_by_term.items()):
            if term not in combined_text:
                continue
            safe_term_id = re.sub(r"[^A-Za-z0-9_-]+", "-", _safe(row.get("term_id")))
            add_chunk(
                chunk_id=f"{notice_id}-term-{safe_term_id}",
                notice_id=notice_id,
                record_id="",
                chunk_type="definitions",
                chunk_title=f"용어 설명: {term}",
                text=f"용어 '{term}'의 뜻: {_clean_text(row.get('definition'))}",
                source_file="용어_사전.csv",
                source_name="마이홈 용어사전 원본 데이터",
                source_url=_safe(row.get("source_url")),
                metadata=meta,
            )

    if output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir.mkdir(parents=True)

    _write_csv(output_dir / OUTPUT_FILES["notices"], NOTICE_FIELDS, notice_rows)
    _write_csv(output_dir / OUTPUT_FILES["chunks"], CHUNK_FIELDS, chunk_rows)
    _write_csv(output_dir / OUTPUT_FILES["documents"], DOCUMENT_FIELDS, document_master_rows)
    _write_csv(output_dir / OUTPUT_FILES["rules"], RULE_FIELDS, rule_rows)
    _write_csv(
        output_dir / "housing_terms.csv",
        ["term_id", "order_no", "initial", "category", "term", "definition", "source_url"],
        loaded["terms"],
    )

    after_hashes = {key: _sha256(path) for key, path in raw_paths.items()}
    if before_hashes != after_hashes:
        raise RuntimeError("원본 파일 해시가 변했습니다. 변환을 중단합니다.")

    output_hashes = {
        filename: _sha256(output_dir / filename)
        for filename in list(OUTPUT_FILES.values()) + ["housing_terms.csv"]
    }
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "source_files_unchanged": True,
        "source_encodings": encodings,
        "source_sha256": before_hashes,
        "output_sha256": output_hashes,
        "counts": {
            "raw_notices": len(loaded["notices"]),
            "raw_income_rules": len(loaded["income"]),
            "raw_eligibility_rules": len(loaded["eligibility"]),
            "raw_terms": len(loaded["terms"]),
            "raw_document_rules": len(loaded["documents"]),
            "notice_master_rows": len(notice_rows),
            "rag_chunks": len(chunk_rows),
            "document_master_rows": len(document_master_rows),
            "document_rule_rows": len(rule_rows),
        },
        "notes": [
            "원본 5개 CSV는 읽기 전용으로 사용했고 수정하지 않았습니다.",
            "공고 원문 URL과 PDF 페이지 정보는 원본 데이터에 없으므로 생성하지 않았습니다.",
            "해당자 서류 중 명확한 조건만 condition_json으로 추론했고 나머지는 조건 확인 후보로 유지합니다.",
        ],
    }
    manifest_path = output_dir / "raw_data_conversion_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="원본 5종 CSV를 운영용 4종 CSV로 변환합니다.")
    parser.add_argument("--source-dir", required=True, help="원본 5종 CSV가 있는 폴더")
    parser.add_argument("--output-dir", default="", help="생성 결과 폴더. 기본값: <source-dir>/generated")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir) if args.output_dir else source_dir / "generated"
    try:
        manifest = convert(source_dir, output_dir)
    except Exception as exc:
        print(f"[FAIL] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    counts = manifest["counts"]
    print("[PASS] 원본 5종 CSV는 변경하지 않았습니다.")
    print(f"[PASS] 공고 마스터: {counts['notice_master_rows']:,}행")
    print(f"[PASS] RAG 청크: {counts['rag_chunks']:,}행")
    print(f"[PASS] 서류 마스터: {counts['document_master_rows']:,}행")
    print(f"[PASS] 공고-서류 규칙: {counts['document_rule_rows']:,}행")
    print(f"[INFO] 생성 위치: {manifest['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
