# -*- coding: utf-8 -*-
"""BGE-M3 + V2 Hybrid 점수 결합 로직의 외부 모델 없는 단위 테스트."""

from __future__ import annotations

import numpy as np

from rag_retriever import HybridBackend, _minmax_normalize


class FakeBackend:
    def __init__(self, scores):
        self.scores = np.asarray(scores, dtype=np.float32)

    def score(self, query, candidate_indices):
        del query
        return self.scores[list(candidate_indices)]


def main() -> None:
    assert np.allclose(_minmax_normalize(np.array([2.0, 4.0])), [0.0, 1.0])
    assert np.allclose(_minmax_normalize(np.array([3.0, 3.0])), [1.0, 1.0])

    hybrid = HybridBackend.__new__(HybridBackend)
    hybrid.dense_weight = 0.7
    hybrid.keyword_weight = 0.3
    hybrid.semantic = FakeBackend([0.2, 0.9, 0.4])
    hybrid.keyword = FakeBackend([8.0, 1.0, 4.0])
    hybrid.last_components = {}

    scores = hybrid.score("청년 월세 지원", [0, 1, 2])
    expected_dense = np.array([0.0, 1.0, 2.0 / 7.0], dtype=np.float32)
    expected_keyword = np.array([1.0, 0.0, 3.0 / 7.0], dtype=np.float32)
    expected = 0.7 * expected_dense + 0.3 * expected_keyword

    assert np.allclose(scores, expected, atol=1e-6)
    assert int(np.argmax(scores)) == 1
    assert set(hybrid.last_components) == {
        "dense_raw", "keyword_raw", "dense", "keyword", "hybrid"
    }

    print("[PASS] min-max 정규화")
    print("[PASS] Dense 0.70 + BM25 0.30 V2 Hybrid 결합")
    print("[PASS] 결합 점수 구성요소 기록")


if __name__ == "__main__":
    main()
