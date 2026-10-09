# -*- coding: utf-8 -*-
"""현재 Ollama 서버와 설정 모델을 확인합니다."""

from __future__ import annotations

from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import json
import os
import socket


def main() -> None:
    base_url = os.getenv(
        "LLM_BASE_URL",
        "http://localhost:11434",
    ).rstrip("/")
    model_name = os.getenv(
        "LLM_MODEL",
        "gemma3:4b",
    )

    if base_url.endswith("/api"):
        tags_url = f"{base_url}/tags"
    else:
        tags_url = f"{base_url}/api/tags"

    request = Request(tags_url, method="GET")

    try:
        with urlopen(request, timeout=10) as response:
            payload = json.loads(
                response.read().decode("utf-8")
            )
    except HTTPError as exc:
        raise SystemExit(
            f"[FAIL] Ollama HTTP 오류 {exc.code}: {exc.reason}"
        ) from exc
    except (URLError, socket.timeout, TimeoutError, OSError) as exc:
        raise SystemExit(
            "[FAIL] Ollama 서버에 연결할 수 없습니다.\n"
            "Ollama가 실행 중인지 확인하세요.\n"
            f"세부 오류: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(
            "[FAIL] Ollama 모델 목록 응답을 해석하지 못했습니다."
        ) from exc

    model_names = [
        str(model.get("name", ""))
        for model in payload.get("models", [])
    ]

    print("[PASS] Ollama 서버 연결")
    print(f"서버: {base_url}")
    print("설치 모델:")
    for name in model_names:
        print(f"- {name}")

    if model_name in model_names:
        print(f"\n[PASS] 설정 모델 사용 가능: {model_name}")
    else:
        print(f"\n[WARN] 설정 모델이 없습니다: {model_name}")
        print(f"설치 명령: ollama pull {model_name}")


if __name__ == "__main__":
    main()
