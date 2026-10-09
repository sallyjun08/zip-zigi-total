# -*- coding: utf-8 -*-
"""
Policy Bridge 실제 공고 RAG Retriever
=====================================

기본 운영 모드
--------------
- 데이터: data/housing_supply_notice_master.csv +
          data/housing_rag_chunks_review_ready.csv
- 검색: BAAI/bge-m3 Dense + BM25 V2 Hybrid
- 범위: 추천 정책 계열에 연결된 실제 notice_id 내부만 검색
- 출력: 실제 공고명, notice_id, 원문 파일·페이지, 공식 링크 포함

호환 모드
---------
기존 data/policy_chunks.json도 data_mode="legacy"로 사용할 수 있지만,
앱의 기본값은 실제 CSV 모드(real)다.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, is_dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence
import json
import math
import os
import re

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "data"
DEFAULT_LEGACY_DATA_PATH = DEFAULT_DATA_DIR / "policy_chunks.json"
DEFAULT_REAL_CHUNK_PATH = DEFAULT_DATA_DIR / "housing_rag_chunks_review_ready.csv"
DEFAULT_NOTICE_PATH = DEFAULT_DATA_DIR / "housing_supply_notice_master.csv"
DEFAULT_CACHE_DIR = BASE_DIR / "vector_store"

DEFAULT_MODEL_NAME = os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-m3")
DEFAULT_DATA_MODE = os.getenv("RAG_DATA_MODE", "real").strip().lower()
DEFAULT_HYBRID_DENSE_WEIGHT = float(os.getenv("RAG_HYBRID_DENSE_WEIGHT", "0.70"))
DEFAULT_HYBRID_KEYWORD_WEIGHT = float(os.getenv("RAG_HYBRID_KEYWORD_WEIGHT", "0.30"))
DEFAULT_DEVICE = os.getenv("RAG_DEVICE", "").strip() or None
DEFAULT_BATCH_SIZE = int(os.getenv("RAG_BATCH_SIZE", "16"))
DEFAULT_NOTICE_CANDIDATES = int(os.getenv("RAG_NOTICE_CANDIDATES", "3"))

SECTION_QUERIES = {
    "eligibility": "지원 대상, 연령, 거주지역, 무주택, 우선순위와 신청 자격 조건",
    "income": "소득 기준, 자산 기준, 자동차 기준과 가구원 수에 따른 경제 조건",
    "benefit": "임대료, 보증금, 공급 주택, 지원 내용과 실제 주거 혜택",
    "application": "신청 기간, 신청 기관, 접수 방법, 선정 절차와 서류 제출 방법",
    "caution": "제외 대상, 중복 제한, 정정 사항, 자격 완화와 추가 확인 유의사항",
}

SECTION_LABELS = {
    "eligibility": "지원 대상",
    "income": "소득·재산 조건",
    "benefit": "지원 혜택",
    "application": "신청 방법",
    "caution": "추가 확인 사항",
}

# 실제 4,060개 청크의 chunk_type 분포에 맞춘 구조화 라우팅이다.
SECTION_CHUNK_TYPES = {
    "eligibility": {
        "priority_rule", "residence_period", "eligibility_common",
        "eligibility_target", "housing_ownership", "eligibility_relaxation",
        "definitions",
    },
    "income": {"income_asset"},
    "benefit": {"rent_conditions", "unit_supply"},
    "application": {
        "application_method", "selection_process", "schedule", "contact",
        "submission_method", "submission_mymy", "documents_common",
        "documents_conditional",
    },
    "caution": {
        "caution", "correction", "eligibility_relaxation",
        "documents_conditional", "submission_mymy",
    },
}

# 일반 서비스 추천 결과를 실제 주택공급 공고 계열에 연결한다.
# 직접 같은 상품이 아닌 경우 relation_type="related_notice"로 명확히 표시한다.
DEFAULT_POLICY_NOTICE_MAP: dict[str, dict[str, Any]] = {
    "JEONSE_LOAN": {
        "housing_types": ["청년 전세임대", "전세임대"],
        "target_keywords": ["청년", "대학생", "취업준비생"],
        "relation_type": "related_notice",
        "relation_note": "전세자금 대출과 동일 상품은 아니며 관련 청년 전세임대 공고를 연결합니다.",
    },
    "LH_MAEIP": {
        "housing_types": [
            "청년 매입임대", "기숙사형 청년주택",
            "청년·신혼부부 매입임대리츠", "매입임대",
        ],
        "target_keywords": ["청년", "대학생", "취업준비생"],
        "relation_type": "direct_family",
        "relation_note": "청년 매입임대 계열 실제 모집공고를 연결합니다.",
    },
    "HAPPY_HOUSE": {
        "housing_types": ["행복주택"],
        "target_keywords": ["청년", "대학생", "취업준비생", "사회초년생"],
        "relation_type": "direct_family",
        "relation_note": "행복주택 청년계층 실제 모집공고를 연결합니다.",
    },
    "VULNERABLE_HOUSING_RELOCATION": {
        "housing_types": ["영구임대", "일반 매입임대", "국민임대", "매입임대"],
        "target_keywords": ["취약", "저소득", "무주택"],
        "relation_type": "related_notice",
        "relation_note": "이주지원과 관련된 공공임대·매입임대 공고를 연결합니다.",
    },
}

LEGACY_REQUIRED_FIELDS = {
    "chunk_id", "policy_id", "policy_name", "section_type", "section_title",
    "text", "source_name", "source_url", "updated_at", "is_mock",
}
REAL_CHUNK_REQUIRED_FIELDS = {"chunk_id", "notice_id", "retrieval_text"}
NOTICE_REQUIRED_FIELDS = {"notice_id", "announcement_title", "housing_type"}


class RetrieverConfigurationError(RuntimeError):
    """검색기 설정 또는 데이터 형식이 올바르지 않을 때 발생한다."""


class PolicyNotFoundError(KeyError):
    """요청한 policy_id를 실제 공고 계열에 연결할 수 없을 때 발생한다."""


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _boolish(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    text = _safe_str(value).lower()
    if not text:
        return default
    return text in {"true", "1", "y", "yes", "예", "t"}


def _read_attr_or_key(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def profile_to_dict(profile: Any) -> dict[str, Any]:
    if profile is None:
        return {}
    if isinstance(profile, Mapping):
        return dict(profile)
    if is_dataclass(profile):
        return asdict(profile)
    fields = (
        "sido", "sigungu", "birth_year", "age", "owns_house",
        "residence_status", "household_size", "income_known", "monthly_income",
    )
    return {field: getattr(profile, field) for field in fields if hasattr(profile, field)}


def build_user_context(profile: Any) -> str:
    data = profile_to_dict(profile)
    sido = data.get("sido") or data.get("region") or "지역 미입력"
    sigungu = data.get("sigungu") or data.get("district") or "시군구 미입력"

    age = data.get("age")
    if age is None:
        birth_year = data.get("birth_year")
        if isinstance(birth_year, int):
            from datetime import date
            age = date.today().year - birth_year
    age_text = f"만 {age}세 전후" if age is not None else "나이 미입력"

    owns_house = data.get("owns_house")
    house_text = "무주택" if owns_house is False else "주택 소유" if owns_house is True else "주택 소유 여부 모름"
    residence_status = data.get("residence_status") or data.get("living_status") or "거주 상태 미입력"
    household_size = data.get("household_size")
    household_text = f"{household_size}인 가구" if isinstance(household_size, int) else "가구원 수 미입력"

    if data.get("income_known") and isinstance(data.get("monthly_income"), (int, float)):
        income_text = f"월 소득 약 {int(data['monthly_income']):,}원"
    else:
        income_text = "월 소득 모름"

    return (
        f"사용자는 {sido} {sigungu} 거주, {age_text}, {house_text}, "
        f"현재 거주 상태는 {residence_status}, {household_text}, {income_text}이다."
    )


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise RetrieverConfigurationError(f"CSV 파일을 찾을 수 없습니다: {path}")
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise RetrieverConfigurationError(f"CSV 인코딩을 읽지 못했습니다: {path}: {last_error}")


def _first_nonempty(values: Iterable[Any]) -> str:
    for value in values:
        text = _safe_str(value)
        if text:
            return text
    return ""


def _join_unique(values: Iterable[Any]) -> str:
    seen: list[str] = []
    for value in values:
        for token in _safe_str(value).split("|"):
            token = token.strip()
            if token and token not in seen:
                seen.append(token)
    return "|".join(seen)


def _looks_mock(mapping: Mapping[str, Any]) -> bool:
    for key in ("is_mock", "is_temporary"):
        if key in mapping and _boolish(mapping.get(key)):
            return True
    searchable = " ".join(
        _safe_str(mapping.get(key))
        for key in (
            "data_quality", "announcement_title", "source_file", "source_name",
            "notice_status", "notes", "retrieval_text", "text",
        )
    ).lower()
    markers = ("개발용", "샘플", "임시 데이터", "temporary", "synthetic", "mock")
    return any(marker in searchable for marker in markers)


def _page_text(start: Any, end: Any) -> str:
    start_text = _safe_str(start)
    end_text = _safe_str(end)
    if not start_text:
        return ""
    if not end_text or end_text == start_text:
        return f"p.{start_text}"
    return f"p.{start_text}~{end_text}"


def _normalize_legacy_chunk(chunk: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(chunk)
    normalized["text"] = " ".join(_safe_str(normalized.get("text")).split())
    normalized["section_type"] = _safe_str(normalized.get("section_type"))
    normalized["policy_id"] = _safe_str(normalized.get("policy_id"))
    normalized["chunk_id"] = _safe_str(normalized.get("chunk_id"))
    normalized["is_mock"] = bool(normalized.get("is_mock", False))
    normalized.setdefault("notice_id", "")
    normalized.setdefault("chunk_type", normalized["section_type"])
    normalized.setdefault("source_file", "")
    normalized.setdefault("source_page_start", "")
    normalized.setdefault("source_page_end", "")
    return normalized


def load_legacy_chunks(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise RetrieverConfigurationError(f"정책 chunk JSON을 찾을 수 없습니다: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RetrieverConfigurationError(f"정책 chunk JSON 형식이 올바르지 않습니다: {exc}") from exc
    chunks = payload.get("chunks") if isinstance(payload, dict) else payload
    if not isinstance(chunks, list):
        raise RetrieverConfigurationError("policy_chunks.json은 list 또는 {'chunks': [...]} 형식이어야 합니다.")

    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(chunks):
        if not isinstance(raw, Mapping):
            raise RetrieverConfigurationError(f"{index}번째 chunk가 객체가 아닙니다.")
        missing = LEGACY_REQUIRED_FIELDS - set(raw)
        if missing:
            raise RetrieverConfigurationError(f"{index}번째 chunk 필수 필드 누락: {sorted(missing)}")
        chunk = _normalize_legacy_chunk(raw)
        if not chunk["chunk_id"] or chunk["chunk_id"] in seen:
            raise RetrieverConfigurationError(f"비어 있거나 중복된 chunk_id: {chunk['chunk_id']}")
        if chunk["section_type"] not in SECTION_QUERIES:
            raise RetrieverConfigurationError(f"허용되지 않은 section_type: {chunk['section_type']}")
        seen.add(chunk["chunk_id"])
        normalized.append(chunk)
    if not normalized:
        raise RetrieverConfigurationError("검색할 legacy 정책 chunk가 없습니다.")
    return normalized


def load_policy_notice_map(data_dir: Path) -> dict[str, dict[str, Any]]:
    mapping = json.loads(json.dumps(DEFAULT_POLICY_NOTICE_MAP, ensure_ascii=False))
    custom_path = data_dir / "policy_notice_map.json"
    if not custom_path.exists():
        return mapping
    try:
        custom = json.loads(custom_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise RetrieverConfigurationError(f"policy_notice_map.json 형식 오류: {exc}") from exc
    if not isinstance(custom, Mapping):
        raise RetrieverConfigurationError("policy_notice_map.json 최상위는 객체여야 합니다.")
    for policy_id, rule in custom.items():
        if not isinstance(rule, Mapping):
            raise RetrieverConfigurationError(f"{policy_id} 매핑은 객체여야 합니다.")
        mapping[str(policy_id)] = dict(rule)
    return mapping


def load_real_corpus(
    chunk_path: Path,
    notice_path: Path,
) -> tuple[list[dict[str, Any]], pd.DataFrame, dict[str, dict[str, Any]]]:
    chunk_df = _read_csv(chunk_path)
    notice_df = _read_csv(notice_path)

    missing_chunks = REAL_CHUNK_REQUIRED_FIELDS - set(chunk_df.columns)
    missing_notices = NOTICE_REQUIRED_FIELDS - set(notice_df.columns)
    if missing_chunks:
        raise RetrieverConfigurationError(f"실제 RAG CSV 필수 컬럼 누락: {sorted(missing_chunks)}")
    if missing_notices:
        raise RetrieverConfigurationError(f"공고 마스터 필수 컬럼 누락: {sorted(missing_notices)}")

    chunk_df = chunk_df.copy()
    notice_df = notice_df.copy()
    for frame in (chunk_df, notice_df):
        frame.columns = [str(column).lstrip("\ufeff").strip() for column in frame.columns]

    if "is_latest" in chunk_df.columns:
        chunk_df = chunk_df[chunk_df["is_latest"].map(lambda value: _boolish(value, default=True))]
    if "review_status" in chunk_df.columns:
        chunk_df = chunk_df[chunk_df["review_status"].fillna("").astype(str).str.strip() != "제외"]
    chunk_df = chunk_df[chunk_df["retrieval_text"].fillna("").astype(str).str.strip() != ""]

    if "is_latest" in notice_df.columns:
        notice_df = notice_df[notice_df["is_latest"].map(lambda value: _boolish(value, default=True))]
    if notice_df.empty or chunk_df.empty:
        raise RetrieverConfigurationError("최신 실제 공고 또는 RAG 청크가 비어 있습니다.")

    notice_lookup: dict[str, dict[str, Any]] = {}
    for notice_id, group in notice_df.groupby("notice_id", dropna=False):
        notice_id_text = _safe_str(notice_id)
        if not notice_id_text:
            continue
        record = {
            "notice_id": notice_id_text,
            "record_id": _first_nonempty(group.get("record_id", [])),
            "announcement_title": _first_nonempty(group["announcement_title"]),
            "operator": _first_nonempty(group.get("operator", [])),
            "housing_type": _first_nonempty(group["housing_type"]),
            "supply_family": _first_nonempty(group.get("supply_family", [])),
            "target_group": _join_unique(group.get("target_group", [])),
            "supply_track": _join_unique(group.get("supply_track", [])),
            "province": _join_unique(group.get("province", [])),
            "city_county_district": _join_unique(group.get("city_county_district", [])),
            "application_url": _first_nonempty(group.get("application_url", [])),
            "notice_url": _first_nonempty(group.get("notice_url", [])),
            "source_file": _first_nonempty(group.get("source_file", [])),
            "verified_at": _first_nonempty(group.get("verified_at", [])),
            "data_quality": _first_nonempty(group.get("data_quality", [])),
            "notice_status": _first_nonempty(group.get("notice_status", [])),
            "notes": _first_nonempty(group.get("notes", [])),
        }
        record["is_mock"] = _looks_mock(record)
        notice_lookup[notice_id_text] = record

    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _, row in chunk_df.iterrows():
        raw = row.to_dict()
        chunk_id = _safe_str(raw.get("chunk_id"))
        notice_id = _safe_str(raw.get("notice_id"))
        text = " ".join(_safe_str(raw.get("retrieval_text")).split())
        if not chunk_id or not notice_id or not text:
            continue
        if chunk_id in seen:
            raise RetrieverConfigurationError(f"중복 chunk_id가 있습니다: {chunk_id}")
        seen.add(chunk_id)
        notice = notice_lookup.get(notice_id, {})
        title = _safe_str(raw.get("announcement_title")) or _safe_str(notice.get("announcement_title"))
        source_file = _safe_str(raw.get("source_file")) or _safe_str(notice.get("source_file"))
        source_page_start = raw.get("source_page_start", raw.get("source_page", ""))
        source_page_end = raw.get("source_page_end", source_page_start)
        source_url = _safe_str(notice.get("notice_url")) or _safe_str(notice.get("application_url"))
        operator = _safe_str(notice.get("operator"))
        source_name = operator or source_file or "주택공급 공고 원문"
        if title and title not in source_name:
            source_name = f"{source_name} · {title}"
        chunk_type = _safe_str(raw.get("chunk_type")) or "unknown"
        chunk = {
            "chunk_id": chunk_id,
            "notice_id": notice_id,
            "policy_id": "",
            "policy_name": title,
            "section_type": "",
            "section_title": _safe_str(raw.get("chunk_title")) or chunk_type,
            "chunk_type": chunk_type,
            "text": text,
            "source_name": source_name,
            "source_url": source_url,
            "updated_at": _safe_str(notice.get("verified_at")),
            "is_mock": bool(_looks_mock(raw) or notice.get("is_mock", False)),
            "source_file": source_file,
            "source_page_start": _safe_str(source_page_start),
            "source_page_end": _safe_str(source_page_end),
            "review_status": _safe_str(raw.get("review_status")) or "미검수",
            "announcement_title": title,
            "operator": operator,
            "housing_type": _safe_str(raw.get("housing_type")) or _safe_str(notice.get("housing_type")),
            "target_group": _safe_str(raw.get("target_group")) or _safe_str(notice.get("target_group")),
            "supply_track": _safe_str(raw.get("supply_track")) or _safe_str(notice.get("supply_track")),
            "province": _safe_str(raw.get("province")) or _safe_str(notice.get("province")),
            "city_county_district": _safe_str(raw.get("city_county_district")) or _safe_str(notice.get("city_county_district")),
        }
        normalized.append(chunk)

    if not normalized:
        raise RetrieverConfigurationError("실제 RAG CSV에서 유효한 청크를 만들지 못했습니다.")
    return normalized, notice_df.reset_index(drop=True), notice_lookup


def _fingerprint(chunks: Sequence[Mapping[str, Any]], model_name: str, data_mode: str) -> str:
    source = {
        "model_name": model_name,
        "data_mode": data_mode,
        "chunks": [
            {
                "chunk_id": c["chunk_id"],
                "notice_id": c.get("notice_id", ""),
                "policy_id": c.get("policy_id", ""),
                "chunk_type": c.get("chunk_type", ""),
                "text": c["text"],
            }
            for c in chunks
        ],
    }
    encoded = json.dumps(source, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


class SemanticBackend:
    def __init__(
        self,
        chunks: Sequence[Mapping[str, Any]],
        model_name: str,
        cache_dir: Path,
        data_mode: str,
        device: Optional[str] = DEFAULT_DEVICE,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RetrieverConfigurationError(
                "semantic/hybrid 검색에는 sentence-transformers가 필요합니다. "
                "pip install sentence-transformers numpy"
            ) from exc
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = max(1, int(batch_size))
        kwargs = {"device": device} if device else {}
        self._model = SentenceTransformer(model_name, **kwargs)

        fingerprint = _fingerprint(chunks, model_name, data_mode)
        prefix = "real_housing" if data_mode == "real_csv" else "legacy_policy"
        self._cache_file = self.cache_dir / f"{prefix}_embeddings.npz"
        self._manifest_file = self.cache_dir / f"{prefix}_embeddings_manifest.json"
        self.document_embeddings = self._load_cache(fingerprint, len(chunks))
        if self.document_embeddings is None:
            texts = [str(chunk["text"]) for chunk in chunks]
            self.document_embeddings = np.asarray(
                self._model.encode(
                    texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=len(texts) >= 100,
                    batch_size=self.batch_size,
                ),
                dtype=np.float32,
            )
            np.savez_compressed(self._cache_file, embeddings=self.document_embeddings)
            self._manifest_file.write_text(
                json.dumps(
                    {
                        "fingerprint": fingerprint,
                        "model_name": model_name,
                        "data_mode": data_mode,
                        "chunk_count": len(chunks),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

    def _load_cache(self, fingerprint: str, chunk_count: int) -> Optional[np.ndarray]:
        if not self._cache_file.exists() or not self._manifest_file.exists():
            return None
        try:
            manifest = json.loads(self._manifest_file.read_text(encoding="utf-8"))
            if manifest.get("fingerprint") != fingerprint or manifest.get("chunk_count") != chunk_count:
                return None
            embeddings = np.asarray(np.load(self._cache_file)["embeddings"], dtype=np.float32)
            if embeddings.ndim != 2 or embeddings.shape[0] != chunk_count:
                return None
            return embeddings
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            return None

    def score(self, query: str, candidate_indices: Sequence[int]) -> np.ndarray:
        query_embedding = np.asarray(
            self._model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=self.batch_size,
            )[0],
            dtype=np.float32,
        )
        return self.document_embeddings[list(candidate_indices)] @ query_embedding


_TOKEN_RE = re.compile(r"[가-힣A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text)]


class KeywordBackend:
    def __init__(self, chunks: Sequence[Mapping[str, Any]]) -> None:
        self.documents = [_tokenize(str(chunk["text"])) for chunk in chunks]
        self.document_count = len(self.documents)
        document_frequency: Counter[str] = Counter()
        for tokens in self.documents:
            document_frequency.update(set(tokens))
        self.idf = {
            token: math.log((1 + self.document_count) / (1 + frequency)) + 1.0
            for token, frequency in document_frequency.items()
        }
        self.document_vectors = [self._vectorize(tokens) for tokens in self.documents]

    def _vectorize(self, tokens: Sequence[str]) -> dict[str, float]:
        counts = Counter(tokens)
        if not counts:
            return {}
        vector = {token: count * self.idf.get(token, 1.0) for token, count in counts.items()}
        norm = math.sqrt(sum(value * value for value in vector.values())) or 1.0
        return {token: value / norm for token, value in vector.items()}

    def score(self, query: str, candidate_indices: Sequence[int]) -> np.ndarray:
        query_vector = self._vectorize(_tokenize(query))
        return np.asarray(
            [
                sum(query_value * self.document_vectors[index].get(token, 0.0) for token, query_value in query_vector.items())
                for index in candidate_indices
            ],
            dtype=np.float32,
        )


class BM25Backend:
    def __init__(self, chunks: Sequence[Mapping[str, Any]], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.documents = [_tokenize(str(chunk["text"])) for chunk in chunks]
        self.document_count = len(self.documents)
        self.k1 = float(k1)
        self.b = float(b)
        self.document_lengths = np.asarray([len(tokens) for tokens in self.documents], dtype=np.float32)
        self.average_document_length = float(self.document_lengths.mean() if self.document_count else 0.0) or 1.0
        self.term_frequencies = [Counter(tokens) for tokens in self.documents]
        document_frequency: Counter[str] = Counter()
        for tokens in self.documents:
            document_frequency.update(set(tokens))
        self.idf = {
            token: math.log(1.0 + (self.document_count - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }

    def score(self, query: str, candidate_indices: Sequence[int]) -> np.ndarray:
        query_tokens = _tokenize(query)
        scores: list[float] = []
        for index in candidate_indices:
            tf = self.term_frequencies[index]
            length = float(self.document_lengths[index])
            score = 0.0
            for token in query_tokens:
                frequency = tf.get(token, 0)
                if frequency <= 0:
                    continue
                denominator = frequency + self.k1 * (1.0 - self.b + self.b * length / self.average_document_length)
                score += self.idf.get(token, 0.0) * (frequency * (self.k1 + 1.0) / max(denominator, 1e-12))
            scores.append(score)
        return np.asarray(scores, dtype=np.float32)


def _minmax_normalize(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    if scores.size == 0:
        return scores
    minimum = float(scores.min())
    maximum = float(scores.max())
    if math.isclose(maximum, minimum):
        return np.ones_like(scores) if maximum > 0 else np.zeros_like(scores)
    return (scores - minimum) / (maximum - minimum)


class HybridBackend:
    """BGE-M3 Dense + BM25 V2 Hybrid."""

    def __init__(
        self,
        chunks: Sequence[Mapping[str, Any]],
        model_name: str,
        cache_dir: Path,
        data_mode: str = "real_csv",
        *,
        dense_weight: float = DEFAULT_HYBRID_DENSE_WEIGHT,
        keyword_weight: float = DEFAULT_HYBRID_KEYWORD_WEIGHT,
        device: Optional[str] = DEFAULT_DEVICE,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        if dense_weight < 0 or keyword_weight < 0:
            raise RetrieverConfigurationError("Hybrid 가중치는 0 이상이어야 합니다.")
        total = dense_weight + keyword_weight
        if total <= 0:
            raise RetrieverConfigurationError("Hybrid 가중치 합은 0보다 커야 합니다.")
        self.dense_weight = float(dense_weight) / total
        self.keyword_weight = float(keyword_weight) / total
        self.semantic = SemanticBackend(chunks, model_name, cache_dir, data_mode, device, batch_size)
        self.keyword = BM25Backend(chunks)
        self.last_components: dict[str, np.ndarray] = {}

    def score(self, query: str, candidate_indices: Sequence[int]) -> np.ndarray:
        dense_raw = self.semantic.score(query, candidate_indices)
        keyword_raw = self.keyword.score(query, candidate_indices)
        dense = _minmax_normalize(dense_raw)
        keyword = _minmax_normalize(keyword_raw)
        hybrid = self.dense_weight * dense + self.keyword_weight * keyword
        self.last_components = {
            "dense_raw": dense_raw,
            "keyword_raw": keyword_raw,
            "dense": dense,
            "keyword": keyword,
            "hybrid": hybrid,
        }
        return np.asarray(hybrid, dtype=np.float32)


def _text_matches_any(value: Any, keywords: Sequence[str]) -> bool:
    text = _safe_str(value)
    return any(keyword in text for keyword in keywords)


def _split_multi(value: Any) -> list[str]:
    return [item.strip() for item in _safe_str(value).split("|") if item.strip()]


class PolicyRetriever:
    """
    실제 CSV와 legacy JSON을 모두 지원하는 검색기.

    data_path:
      - 실제 모드: data 폴더 또는 housing_rag_chunks_review_ready.csv
      - legacy 모드: policy_chunks.json
    data_mode: "real", "legacy", "auto"
    """

    def __init__(
        self,
        data_path: Path | str = DEFAULT_DATA_DIR,
        backend: str = "hybrid",
        model_name: str = DEFAULT_MODEL_NAME,
        cache_dir: Path | str = DEFAULT_CACHE_DIR,
        data_mode: str = DEFAULT_DATA_MODE,
        notice_path: Path | str | None = None,
        hybrid_dense_weight: float = DEFAULT_HYBRID_DENSE_WEIGHT,
        hybrid_keyword_weight: float = DEFAULT_HYBRID_KEYWORD_WEIGHT,
        device: Optional[str] = DEFAULT_DEVICE,
        batch_size: int = DEFAULT_BATCH_SIZE,
        notice_candidate_limit: int = DEFAULT_NOTICE_CANDIDATES,
    ) -> None:
        raw_path = Path(data_path)
        self.backend_name = backend.lower().strip()
        requested_mode = data_mode.lower().strip()
        if requested_mode not in {"real", "legacy", "auto"}:
            raise RetrieverConfigurationError("data_mode는 real, legacy 또는 auto여야 합니다.")

        if raw_path.is_dir() or raw_path.suffix == "":
            self.data_dir = raw_path
            real_chunk_path = self.data_dir / "housing_rag_chunks_review_ready.csv"
            legacy_path = self.data_dir / "policy_chunks.json"
            resolved_notice_path = Path(notice_path) if notice_path else self.data_dir / "housing_supply_notice_master.csv"
        elif raw_path.suffix.lower() == ".csv":
            self.data_dir = raw_path.parent
            real_chunk_path = raw_path
            legacy_path = self.data_dir / "policy_chunks.json"
            resolved_notice_path = Path(notice_path) if notice_path else self.data_dir / "housing_supply_notice_master.csv"
        else:
            self.data_dir = raw_path.parent
            legacy_path = raw_path
            real_chunk_path = self.data_dir / "housing_rag_chunks_review_ready.csv"
            resolved_notice_path = Path(notice_path) if notice_path else self.data_dir / "housing_supply_notice_master.csv"

        if requested_mode == "auto":
            requested_mode = "real" if real_chunk_path.exists() and resolved_notice_path.exists() else "legacy"
        self.data_mode = "real_csv" if requested_mode == "real" else "legacy_json"
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.notice_candidate_limit = max(1, int(notice_candidate_limit))
        self.notices = pd.DataFrame()
        self.notice_lookup: dict[str, dict[str, Any]] = {}
        self.policy_notice_map = load_policy_notice_map(self.data_dir)

        if requested_mode == "real":
            self.chunks, self.notices, self.notice_lookup = load_real_corpus(real_chunk_path, resolved_notice_path)
            self.data_path = real_chunk_path
        else:
            self.chunks = load_legacy_chunks(legacy_path)
            self.data_path = legacy_path

        self._policy_indices: dict[str, list[int]] = {}
        self._policy_section_indices: dict[tuple[str, str], list[int]] = {}
        self._notice_indices: dict[str, list[int]] = {}
        self._notice_chunk_type_indices: dict[tuple[str, str], list[int]] = {}
        for index, chunk in enumerate(self.chunks):
            policy_id = _safe_str(chunk.get("policy_id"))
            notice_id = _safe_str(chunk.get("notice_id"))
            section_type = _safe_str(chunk.get("section_type"))
            chunk_type = _safe_str(chunk.get("chunk_type"))
            if policy_id:
                self._policy_indices.setdefault(policy_id, []).append(index)
                self._policy_section_indices.setdefault((policy_id, section_type), []).append(index)
            if notice_id:
                self._notice_indices.setdefault(notice_id, []).append(index)
                self._notice_chunk_type_indices.setdefault((notice_id, chunk_type), []).append(index)

        if self.backend_name == "hybrid":
            self.backend = HybridBackend(
                self.chunks,
                self.model_name,
                self.cache_dir,
                self.data_mode,
                dense_weight=hybrid_dense_weight,
                keyword_weight=hybrid_keyword_weight,
                device=device,
                batch_size=batch_size,
            )
        elif self.backend_name == "semantic":
            self.backend = SemanticBackend(
                self.chunks, self.model_name, self.cache_dir, self.data_mode, device, batch_size
            )
        elif self.backend_name == "keyword":
            self.backend = KeywordBackend(self.chunks)
        else:
            raise RetrieverConfigurationError("backend는 hybrid, semantic 또는 keyword여야 합니다.")

    @property
    def policy_ids(self) -> list[str]:
        if self.data_mode == "real_csv":
            return sorted(self.policy_notice_map)
        return sorted(self._policy_indices)

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    def _select_notice_candidates(self, policy_id: str, user_profile: Any) -> tuple[list[dict[str, Any]], str]:
        rule = self.policy_notice_map.get(policy_id)
        if not rule:
            return [], "현재 실제 주택공급 공고 DB에는 이 서비스와 연결할 공고 계열이 없습니다."
        if self.notices.empty:
            return [], "공고 마스터가 비어 있습니다."

        candidates = self.notices.copy()
        if "is_latest" in candidates.columns:
            candidates = candidates[candidates["is_latest"].map(lambda value: _boolish(value, default=True))]
        housing_types = [str(value) for value in rule.get("housing_types", []) if str(value).strip()]
        if housing_types:
            candidates = candidates[
                candidates.get("housing_type", pd.Series(index=candidates.index, dtype=str)).apply(
                    lambda value: _text_matches_any(value, housing_types)
                )
            ]
        target_keywords = [str(value) for value in rule.get("target_keywords", []) if str(value).strip()]
        if target_keywords and not candidates.empty and "target_group" in candidates.columns:
            target_mask = candidates["target_group"].apply(lambda value: _text_matches_any(value, target_keywords))
            if target_mask.any():
                candidates = candidates[target_mask]
        if candidates.empty:
            return [], _safe_str(rule.get("relation_note")) or "연결 가능한 실제 공고가 없습니다."

        profile = profile_to_dict(user_profile)
        sido = _safe_str(profile.get("sido"))
        sigungu = _safe_str(profile.get("sigungu"))
        age = profile.get("age")
        if age is None and isinstance(profile.get("birth_year"), int):
            from datetime import date
            age = date.today().year - int(profile["birth_year"])
        owns_house = profile.get("owns_house")

        def score_row(row: pd.Series) -> float:
            score = 50.0
            provinces = _split_multi(row.get("province"))
            districts = _safe_str(row.get("city_county_district"))
            if "전국" in provinces or _safe_str(row.get("province")) == "전국":
                score += 25.0
            elif sido and sido in provinces:
                score += 40.0
            elif sido:
                score -= 15.0
            if sigungu:
                if sigungu in districts:
                    score += 20.0
                elif "전국" in districts:
                    score += 8.0
                elif "여러 지역" in districts and sido in provinces:
                    score += 10.0
            if target_keywords and _text_matches_any(row.get("target_group"), target_keywords):
                score += 10.0
            if age is not None:
                min_age = row.get("age_min")
                max_age = row.get("age_max")
                if pd.notna(min_age) and float(age) < float(min_age):
                    score -= 60.0
                elif pd.notna(max_age) and float(age) > float(max_age):
                    score -= 60.0
                elif pd.notna(min_age) or pd.notna(max_age):
                    score += 15.0
            ownership_text = _safe_str(row.get("housing_ownership_condition"))
            if owns_house is True and "무주택" in ownership_text:
                score -= 100.0
            elif owns_house is False and "무주택" in ownership_text:
                score += 10.0
            if _looks_mock(row.to_dict()):
                score -= 200.0
            return round(score, 3)

        candidates = candidates.copy()
        candidates["_rag_match_score"] = candidates.apply(score_row, axis=1)
        candidates = (
            candidates.sort_values(["_rag_match_score", "notice_id"], ascending=[False, True])
            .drop_duplicates("notice_id", keep="first")
            .head(self.notice_candidate_limit)
        )
        output: list[dict[str, Any]] = []
        for _, row in candidates.iterrows():
            notice_id = _safe_str(row.get("notice_id"))
            lookup = self.notice_lookup.get(notice_id, {})
            record = {
                "notice_id": notice_id,
                "record_id": _safe_str(row.get("record_id")) or _safe_str(lookup.get("record_id")),
                "announcement_title": _safe_str(row.get("announcement_title")) or _safe_str(lookup.get("announcement_title")),
                "operator": _safe_str(row.get("operator")) or _safe_str(lookup.get("operator")),
                "housing_type": _safe_str(row.get("housing_type")) or _safe_str(lookup.get("housing_type")),
                "target_group": _safe_str(row.get("target_group")) or _safe_str(lookup.get("target_group")),
                "supply_track": _safe_str(row.get("supply_track")) or _safe_str(lookup.get("supply_track")),
                "province": _safe_str(row.get("province")) or _safe_str(lookup.get("province")),
                "city_county_district": _safe_str(row.get("city_county_district")) or _safe_str(lookup.get("city_county_district")),
                "notice_url": _safe_str(row.get("notice_url")) or _safe_str(lookup.get("notice_url")),
                "application_url": _safe_str(row.get("application_url")) or _safe_str(lookup.get("application_url")),
                "source_file": _safe_str(row.get("source_file")) or _safe_str(lookup.get("source_file")),
                "verified_at": _safe_str(row.get("verified_at")) or _safe_str(lookup.get("verified_at")),
                "match_score": float(row["_rag_match_score"]),
                "relation_type": _safe_str(rule.get("relation_type")),
                "relation_note": _safe_str(rule.get("relation_note")),
                "is_mock": bool(_looks_mock(row.to_dict()) or lookup.get("is_mock", False)),
            }
            output.append(record)
        return output, _safe_str(rule.get("relation_note"))

    def _real_candidate_indices(self, notice_ids: Sequence[str], section_type: str) -> list[int]:
        allowed_types = SECTION_CHUNK_TYPES[section_type]
        indices: list[int] = []
        for notice_id in notice_ids:
            for chunk_type in allowed_types:
                indices.extend(self._notice_chunk_type_indices.get((notice_id, chunk_type), []))
        if indices:
            return list(dict.fromkeys(indices))
        # 해당 section 추출이 비어 있을 때만 선택 공고 전체로 fallback한다.
        for notice_id in notice_ids:
            indices.extend(self._notice_indices.get(notice_id, []))
        return list(dict.fromkeys(indices))

    def _component_scores(self, position: int) -> dict[str, float]:
        if self.backend_name != "hybrid":
            return {}
        components = getattr(self.backend, "last_components", {})
        return {
            "dense_score": round(float(components["dense"][position]), 4),
            "keyword_score": round(float(components["keyword"][position]), 4),
        }

    def retrieve_policy_evidence(
        self,
        policy_id: str,
        user_profile: Any,
        sections: Optional[Iterable[str]] = None,
        top_k: int = 1,
        min_score: float = 0.20,
    ) -> dict[str, Any]:
        policy_id = str(policy_id).strip()
        requested_sections = list(sections or SECTION_QUERIES.keys())
        unknown_sections = set(requested_sections) - set(SECTION_QUERIES)
        if unknown_sections:
            raise ValueError(f"지원하지 않는 section: {sorted(unknown_sections)}")
        if top_k < 1:
            raise ValueError("top_k는 1 이상이어야 합니다.")

        if self.data_mode == "real_csv":
            candidates, relation_note = self._select_notice_candidates(policy_id, user_profile)
            if not candidates:
                return {
                    "policy_id": policy_id,
                    "policy_name": policy_id,
                    "retrieval_status": "not_supported",
                    "backend": self.backend_name,
                    "embedding_model": self.model_name if self.backend_name in {"hybrid", "semantic"} else "",
                    "data_mode": self.data_mode,
                    "is_mock_data": False,
                    "requested_sections": requested_sections,
                    "user_context": build_user_context(user_profile),
                    "evidence": {section: [] for section in requested_sections},
                    "source_urls": [],
                    "matched_notices": [],
                    "selected_notice_ids": [],
                    "relation_note": relation_note,
                }
            notice_ids = [candidate["notice_id"] for candidate in candidates if candidate["notice_id"] in self._notice_indices]
            policy_name = candidates[0]["announcement_title"] or policy_id
            user_context = build_user_context(user_profile)
            notice_context = "; ".join(
                f"{item['announcement_title']}({item['notice_id']}, {item['province']} {item['city_county_district']})"
                for item in candidates
            )
            evidence: dict[str, list[dict[str, Any]]] = {}
            found_count = 0
            for section_type in requested_sections:
                candidate_indices = self._real_candidate_indices(notice_ids, section_type)
                query = (
                    f"추천 서비스 계열: {policy_id}. 선택된 실제 공고: {notice_context}. "
                    f"찾을 정보: {SECTION_QUERIES[section_type]}. {user_context}"
                )
                if not candidate_indices:
                    evidence[section_type] = []
                    continue
                scores = self.backend.score(query, candidate_indices)
                ranked_positions = np.argsort(scores)[::-1]
                section_results: list[dict[str, Any]] = []
                seen_texts: set[str] = set()
                for position in ranked_positions:
                    score = float(scores[position])
                    if score < min_score:
                        continue
                    chunk = self.chunks[candidate_indices[int(position)]]
                    if chunk["text"] in seen_texts:
                        continue
                    seen_texts.add(chunk["text"])
                    page = _page_text(chunk.get("source_page_start"), chunk.get("source_page_end"))
                    section_results.append(
                        {
                            "chunk_id": chunk["chunk_id"],
                            "policy_id": policy_id,
                            "policy_name": policy_name,
                            "notice_id": chunk.get("notice_id", ""),
                            "notice_title": chunk.get("announcement_title", ""),
                            "section_type": section_type,
                            "chunk_type": chunk.get("chunk_type", ""),
                            "section_title": chunk.get("section_title", ""),
                            "text": chunk["text"],
                            "similarity": round(score, 4),
                            "source_name": chunk.get("source_name", ""),
                            "source_url": chunk.get("source_url", ""),
                            "source_file": chunk.get("source_file", ""),
                            "source_page": page,
                            "source_page_start": chunk.get("source_page_start", ""),
                            "source_page_end": chunk.get("source_page_end", ""),
                            "updated_at": chunk.get("updated_at", ""),
                            "review_status": chunk.get("review_status", ""),
                            "is_mock": chunk.get("is_mock", False),
                            **self._component_scores(int(position)),
                        }
                    )
                    if len(section_results) >= top_k:
                        break
                evidence[section_type] = section_results
                if section_results:
                    found_count += 1

            status = "success" if found_count == len(requested_sections) else "partial" if found_count else "not_found"
            source_urls = sorted(
                {
                    item["source_url"]
                    for values in evidence.values()
                    for item in values
                    if item.get("source_url")
                }
            )
            is_mock = any(candidate.get("is_mock", False) for candidate in candidates) or any(
                item.get("is_mock", False) for values in evidence.values() for item in values
            )
            return {
                "policy_id": policy_id,
                "policy_name": policy_name,
                "retrieval_status": status,
                "backend": self.backend_name,
                "embedding_model": self.model_name if self.backend_name in {"hybrid", "semantic"} else "",
                "hybrid_weights": (
                    {"dense": round(float(self.backend.dense_weight), 4), "keyword": round(float(self.backend.keyword_weight), 4)}
                    if self.backend_name == "hybrid" else {}
                ),
                "data_mode": self.data_mode,
                "requested_sections": requested_sections,
                "is_mock_data": is_mock,
                "user_context": user_context,
                "evidence": evidence,
                "source_urls": source_urls,
                "matched_notices": candidates,
                "selected_notice_ids": notice_ids,
                "relation_note": relation_note,
            }

        # Legacy JSON 모드
        if policy_id not in self._policy_indices:
            raise PolicyNotFoundError(f"legacy 문서에 없는 policy_id: {policy_id}")
        first_chunk = self.chunks[self._policy_indices[policy_id][0]]
        policy_name = first_chunk["policy_name"]
        user_context = build_user_context(user_profile)
        evidence: dict[str, list[dict[str, Any]]] = {}
        found_count = 0
        for section_type in requested_sections:
            candidate_indices = self._policy_section_indices.get((policy_id, section_type), []) or self._policy_indices[policy_id]
            query = f"정책명: {policy_name}. 찾을 정보: {SECTION_QUERIES[section_type]}. {user_context}"
            scores = self.backend.score(query, candidate_indices)
            section_results: list[dict[str, Any]] = []
            for position in np.argsort(scores)[::-1]:
                score = float(scores[position])
                if score < min_score:
                    continue
                chunk = self.chunks[candidate_indices[int(position)]]
                section_results.append(
                    {
                        "chunk_id": chunk["chunk_id"],
                        "policy_id": policy_id,
                        "policy_name": policy_name,
                        "section_type": section_type,
                        "chunk_type": chunk.get("chunk_type", section_type),
                        "section_title": chunk["section_title"],
                        "text": chunk["text"],
                        "similarity": round(score, 4),
                        "source_name": chunk["source_name"],
                        "source_url": chunk["source_url"],
                        "updated_at": chunk["updated_at"],
                        "is_mock": chunk["is_mock"],
                        **self._component_scores(int(position)),
                    }
                )
                if len(section_results) >= top_k:
                    break
            evidence[section_type] = section_results
            if section_results:
                found_count += 1
        status = "success" if found_count == len(requested_sections) else "partial" if found_count else "not_found"
        return {
            "policy_id": policy_id,
            "policy_name": policy_name,
            "retrieval_status": status,
            "backend": self.backend_name,
            "embedding_model": self.model_name if self.backend_name in {"hybrid", "semantic"} else "",
            "data_mode": self.data_mode,
            "requested_sections": requested_sections,
            "is_mock_data": any(self.chunks[index]["is_mock"] for index in self._policy_indices[policy_id]),
            "user_context": user_context,
            "evidence": evidence,
            "source_urls": sorted({item["source_url"] for values in evidence.values() for item in values if item["source_url"]}),
            "matched_notices": [],
            "selected_notice_ids": [],
            "relation_note": "",
        }

    def retrieve_for_matches(
        self,
        matches: Sequence[Any],
        user_profile: Any,
        max_policies: Optional[int] = None,
        top_k_per_section: int = 1,
        min_score: float = 0.20,
    ) -> list[dict[str, Any]]:
        selected = list(matches)[:max_policies] if max_policies is not None else list(matches)
        results: list[dict[str, Any]] = []
        for match in selected:
            policy = _read_attr_or_key(match, "policy")
            if policy is None:
                raise ValueError("match에 policy가 없습니다.")
            policy_id = _read_attr_or_key(policy, "policy_id")
            if not policy_id:
                raise ValueError("match.policy에 policy_id가 없습니다.")
            result = self.retrieve_policy_evidence(
                str(policy_id), user_profile, top_k=top_k_per_section, min_score=min_score
            )
            result["recommendation_score"] = _read_attr_or_key(match, "role_base_score")
            result["recommendation_income_note"] = _read_attr_or_key(match, "income_note")
            results.append(result)
        return results

    @staticmethod
    def format_llm_context(retrieval_result: Mapping[str, Any]) -> str:
        lines = [
            f"[정책 ID] {retrieval_result.get('policy_id', '')}",
            f"[정책/공고명] {retrieval_result.get('policy_name', '')}",
            f"[데이터 모드] {retrieval_result.get('data_mode', '')}",
            f"[사용자 조건] {retrieval_result.get('user_context', '')}",
        ]
        notices = retrieval_result.get("matched_notices", [])
        if notices:
            lines.append("\n[연결된 실제 공고]")
            for notice in notices:
                lines.append(
                    f"- {notice.get('announcement_title')} / {notice.get('notice_id')} / "
                    f"{notice.get('province')} {notice.get('city_county_district')}"
                )
        lines.append("\n[검색된 공식 근거]")
        evidence = retrieval_result.get("evidence", {})
        for section_type in retrieval_result.get("requested_sections", SECTION_QUERIES.keys()):
            lines.append(f"\n## {SECTION_LABELS.get(section_type, section_type)}")
            items = evidence.get(section_type, [])
            if not items:
                lines.append("- 검색 근거 없음: 공식 공고에서 추가 확인 필요")
                continue
            for item in items:
                source = item.get("source_name") or "출처 미입력"
                page = item.get("source_page") or "페이지 확인 필요"
                lines.append(f"- {item['text']}")
                lines.append(f"  근거: {source} / {page} / chunk={item.get('chunk_id')}")
        return "\n".join(lines)


def retrieve_policy_evidence(
    policy_id: str,
    user_profile: Any,
    *,
    data_path: Path | str = DEFAULT_DATA_DIR,
    backend: str = "hybrid",
    data_mode: str = DEFAULT_DATA_MODE,
    top_k: int = 1,
    min_score: float = 0.20,
) -> dict[str, Any]:
    retriever = PolicyRetriever(data_path=data_path, backend=backend, data_mode=data_mode)
    return retriever.retrieve_policy_evidence(policy_id, user_profile, top_k=top_k, min_score=min_score)


if __name__ == "__main__":
    demo_profile = {
        "sido": "경기도", "sigungu": "수원시", "birth_year": 2002,
        "owns_house": False, "residence_status": "월세", "household_size": 1,
        "income_known": False, "monthly_income": None,
    }
    retriever = PolicyRetriever(data_path=DEFAULT_DATA_DIR, backend="keyword", data_mode="auto")
    print(f"data_mode={retriever.data_mode}, chunks={retriever.chunk_count}")
    for policy_id in retriever.policy_ids:
        result = retriever.retrieve_policy_evidence(policy_id, demo_profile, top_k=1, min_score=0.0)
        print(policy_id, result["retrieval_status"], result.get("selected_notice_ids"))
