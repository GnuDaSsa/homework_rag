import json
import math
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests


APP_DIR = Path(__file__).resolve().parent
INDEX_PATH = APP_DIR / "index.html"

COHERE_EMBED_MODEL = os.getenv("COHERE_EMBED_MODEL", "embed-v4.0")
COHERE_CHAT_MODEL = os.getenv("COHERE_CHAT_MODEL", "command-r-08-2024")
COHERE_RERANK_MODEL = os.getenv("COHERE_RERANK_MODEL", "rerank-v3.5")
RAG_TABLE = os.getenv("RAG_TABLE", "documents_test")
DOCUMENT_KEY = os.getenv("DOCUMENT_KEY", "seongnam_youth_hackathon_2026")
DEFAULT_VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", "11"))
DEFAULT_RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "6"))


def load_dotenv() -> None:
    env_path = APP_DIR / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} 환경변수가 필요합니다.")
    return value


def now_ms() -> float:
    return time.perf_counter() * 1000


def post_json(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: int = 60) -> dict[str, Any]:
    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if not response.ok:
        raise RuntimeError(f"{response.status_code} {response.text[:700]}")
    return response.json()


def parse_vector(value: Any) -> list[float]:
    if isinstance(value, list):
        return [float(v) for v in value]
    if isinstance(value, str):
        return [float(v) for v in json.loads(value)]
    raise ValueError(f"지원하지 않는 embedding 형식: {type(value).__name__}")


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def embed_question(question: str) -> tuple[list[float], dict[str, Any]]:
    started = now_ms()
    data = post_json(
        "https://api.cohere.com/v2/embed",
        {"Authorization": f"Bearer {required_env('COHERE_API_KEY')}", "Content-Type": "application/json"},
        {
            "model": COHERE_EMBED_MODEL,
            "texts": [question],
            "input_type": "search_query",
            "embedding_types": ["float"],
            "output_dimension": 1536,
        },
        timeout=60,
    )
    return data["embeddings"]["float"][0], {
        "model": COHERE_EMBED_MODEL,
        "ms": round(now_ms() - started),
        "tokens": data.get("meta", {}).get("billed_units", {}),
    }


def fetch_document_rows() -> list[dict[str, Any]]:
    supabase_url = required_env("SUPABASE_URL").rstrip("/")
    supabase_key = required_env("SUPABASE_SERVICE_ROLE_KEY")
    response = requests.get(
        f"{supabase_url}/rest/v1/{RAG_TABLE}",
        headers={"apikey": supabase_key, "Authorization": f"Bearer {supabase_key}"},
        params={"select": "id,content,metadata,embedding", "limit": "5000"},
        timeout=60,
    )
    if not response.ok:
        raise RuntimeError(f"Supabase 조회 실패: {response.status_code} {response.text[:700]}")
    rows = response.json()
    return [row for row in rows if (row.get("metadata") or {}).get("document_key") == DOCUMENT_KEY]


def vector_search(question_embedding: list[float], top_k: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started = now_ms()
    rows = fetch_document_rows()
    scored = []
    for row in rows:
        scored.append(
            {
                "id": row["id"],
                "content": row.get("content") or "",
                "metadata": row.get("metadata") or {},
                "similarity": cosine_similarity(question_embedding, parse_vector(row["embedding"])),
            }
        )
    scored.sort(key=lambda item: item["similarity"], reverse=True)
    return scored[:top_k], {
        "table_chunks": len(rows),
        "candidate_chunks": min(top_k, len(scored)),
        "ms": round(now_ms() - started),
    }


def rerank(question: str, candidates: list[dict[str, Any]], top_n: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not candidates:
        return [], {"ms": 0, "reranked_chunks": 0}
    documents = [format_rerank_document(candidate) for candidate in candidates]
    started = now_ms()
    data = post_json(
        "https://api.cohere.com/v2/rerank",
        {"Authorization": f"Bearer {required_env('COHERE_API_KEY')}", "Content-Type": "application/json"},
        {
            "model": COHERE_RERANK_MODEL,
            "query": question,
            "documents": documents,
            "top_n": min(top_n, len(documents)),
        },
        timeout=60,
    )
    reranked = []
    for result in data.get("results", []):
        item = dict(candidates[result["index"]])
        item["rerank_score"] = result.get("relevance_score", 0)
        reranked.append(item)
    return reranked, {
        "ms": round(now_ms() - started),
        "reranked_chunks": len(reranked),
    }


def format_rerank_document(chunk: dict[str, Any]) -> str:
    metadata = chunk.get("metadata") or {}
    source = metadata.get("source_path") or metadata.get("section_title") or f"id={chunk.get('id')}"
    return f"{source}\n{chunk.get('content', '')}".strip()


def build_context(chunks: list[dict[str, Any]]) -> str:
    blocks = []
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata") or {}
        source = metadata.get("source_path") or metadata.get("section_title") or f"id={chunk.get('id')}"
        blocks.append(f"[{index}] {source}\n{chunk.get('content', '')}")
    return "\n\n".join(blocks)


def generate_answer(question: str, chunks: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    started = now_ms()
    context = build_context(chunks)
    data = post_json(
        "https://api.cohere.com/v2/chat",
        {"Authorization": f"Bearer {required_env('COHERE_API_KEY')}", "Content-Type": "application/json"},
        {
            "model": COHERE_CHAT_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "너는 성남시 정책 아이디어 청년 해커톤 공고 문서 기반 질의응답 도우미다. "
                        "제공된 컨텍스트만 근거로 답하고, 모르면 문서에서 확인되지 않는다고 말한다. "
                        "일정, 자격, 제출서류, 심사, 시상 관련 답변은 핵심 조건을 빠짐없이 정리한다."
                    ),
                },
                {
                    "role": "user",
                    "content": f"질문: {question}\n\n컨텍스트:\n{context}\n\n답변 끝에 사용한 근거 번호를 표시해줘.",
                },
            ],
            "temperature": 0.2,
            "max_tokens": 900,
        },
        timeout=120,
    )
    content = "".join(part.get("text", "") for part in data["message"].get("content", []) if part.get("type") == "text")
    usage = data.get("usage", {})
    return content, {
        "model": COHERE_CHAT_MODEL,
        "ms": round(now_ms() - started),
        "tokens": usage.get("tokens", usage.get("billed_units", {})),
    }


def answer_question(question: str, top_k: int, top_n: int) -> dict[str, Any]:
    total_started = now_ms()
    question_embedding, embedding_metrics = embed_question(question)
    candidates, search_metrics = vector_search(question_embedding, top_k)
    chunks, rerank_metrics = rerank(question, candidates, top_n)
    answer, generation_metrics = generate_answer(question, chunks)
    return {
        "question": question,
        "answer": answer,
        "chunks": [
            {
                "id": chunk["id"],
                "content": chunk["content"],
                "metadata": chunk["metadata"],
                "similarity": round(chunk["similarity"], 6),
                "rerank_score": round(chunk.get("rerank_score", 0), 6),
            }
            for chunk in chunks
        ],
        "metrics": {
            "table": RAG_TABLE,
            "document_key": DOCUMENT_KEY,
            "table_chunks": search_metrics["table_chunks"],
            "candidate_chunks": search_metrics["candidate_chunks"],
            "answer_chunks": len(chunks),
            "embedding_ms": embedding_metrics["ms"],
            "search_ms": search_metrics["ms"],
            "rerank_ms": rerank_metrics["ms"],
            "generation_ms": generation_metrics["ms"],
            "total_ms": round(now_ms() - total_started),
            "embedding_tokens": embedding_metrics["tokens"],
            "generation_tokens": generation_metrics["tokens"],
        },
        "models": {
            "embedding": embedding_metrics["model"],
            "rerank": COHERE_RERANK_MODEL,
            "generation": generation_metrics["model"],
        },
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    def send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            body = INDEX_PATH.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/api/health":
            self.send_json(200, {"ok": True, "table": RAG_TABLE, "document_key": DOCUMENT_KEY})
            return
        if path == "/api/count":
            try:
                self.send_json(200, {"chunks": len(fetch_document_rows()), "table": RAG_TABLE, "document_key": DOCUMENT_KEY})
            except Exception as exc:
                self.send_json(500, {"error": str(exc)})
            return
        self.send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/ask":
            self.send_json(404, {"error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            question = str(payload.get("question", "")).strip()
            if not question:
                self.send_json(400, {"error": "질문을 입력하세요."})
                return
            top_k = int(payload.get("top_k") or DEFAULT_VECTOR_TOP_K)
            top_n = int(payload.get("top_n") or DEFAULT_RERANK_TOP_N)
            self.send_json(200, answer_question(question, top_k, top_n))
        except Exception as exc:
            self.send_json(500, {"error": str(exc)})


def main() -> None:
    load_dotenv()
    port = int(os.getenv("PORT", "8767"))
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Hackathon RAG app: http://127.0.0.1:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
