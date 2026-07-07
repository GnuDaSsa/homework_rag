import argparse
import json
import os
import time
from getpass import getpass
from pathlib import Path
from typing import Any, Iterable

import requests
from supabase import create_client


DEFAULT_INPUT = "hackathon_chunks.json"
DEFAULT_TABLE = "documents_test"
DEFAULT_EMBED_MODEL = "embed-v4.0"
DEFAULT_OUTPUT_DIMENSION = 1536


def load_dotenv() -> None:
    app_dir = Path(__file__).resolve().parent
    for env_path in (app_dir.parent / ".env", app_dir / ".env"):
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def get_config_value(env_name: str, label: str, secret: bool = False) -> str:
    value = os.getenv(env_name, "").strip()
    if value:
        return value
    prompt = f"{label} ({env_name}): "
    value = getpass(prompt).strip() if secret else input(prompt).strip()
    if not value:
        raise ValueError(f"{env_name} 값이 필요합니다.")
    return value


def load_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        raise ValueError("입력 파일은 JSON 객체 배열이어야 합니다.")
    rows = []
    for index, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"{index}번째 항목이 객체가 아닙니다.")
        row_id = row.get("id", row.get("chunk_id", index))
        content = str(row.get("content") or row.get("text") or "").strip()
        if not content:
            raise ValueError(f"id={row_id} 항목의 content가 비어 있습니다.")
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        metadata.setdefault("source_row", index)
        metadata.setdefault("original_chunk_id", row_id)
        rows.append({"id": row_id, "content": content, "metadata": metadata})
    return rows


def batched(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def cohere_embed(
    api_key: str,
    texts: list[str],
    model: str,
    input_type: str,
    output_dimension: int,
    max_retries: int = 4,
) -> list[list[float]]:
    payload = {
        "model": model,
        "texts": texts,
        "input_type": input_type,
        "embedding_types": ["float"],
        "output_dimension": output_dimension,
    }
    for attempt in range(max_retries):
        response = requests.post(
            "https://api.cohere.com/v2/embed",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=60,
        )
        if response.ok:
            return response.json()["embeddings"]["float"]
        if attempt == max_retries - 1:
            raise RuntimeError(f"Cohere embed 실패: {response.status_code} {response.text[:700]}")
        time.sleep(2**attempt)
    raise RuntimeError("Cohere embed 실패")


def upsert_rows(supabase, table: str, rows: list[dict[str, Any]], max_retries: int = 4) -> None:
    for attempt in range(max_retries):
        try:
            supabase.table(table).upsert(rows, on_conflict="id").execute()
            return
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(2**attempt)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Cohere embed-v4.0으로 청크를 임베딩해 Supabase에 업로드합니다.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument("--model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--output-dimension", type=int, default=DEFAULT_OUTPUT_DIMENSION)
    parser.add_argument("--embed-batch-size", type=int, default=96)
    parser.add_argument("--db-batch-size", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    rows = load_rows(input_path)
    print(f"Loaded: {input_path}")
    print(f"Rows prepared: {len(rows)}")
    if rows:
        print(json.dumps({**rows[0], "content": rows[0]["content"][:180]}, ensure_ascii=True, indent=2))
    if args.dry_run:
        print("Dry run complete.")
        return

    cohere_api_key = get_config_value("COHERE_API_KEY", "Cohere API key", secret=True)
    supabase_url = get_config_value("SUPABASE_URL", "Supabase URL")
    supabase_key = get_config_value("SUPABASE_SERVICE_ROLE_KEY", "Supabase secret/service_role key", secret=True)
    supabase = create_client(supabase_url, supabase_key)

    uploaded = 0
    for embed_batch in batched(rows, args.embed_batch_size):
        embeddings = cohere_embed(
            cohere_api_key,
            [row["content"] for row in embed_batch],
            args.model,
            "search_document",
            args.output_dimension,
        )
        upload_rows = [
            {
                "id": row["id"],
                "content": row["content"],
                "metadata": row["metadata"],
                "embedding": embedding,
            }
            for row, embedding in zip(embed_batch, embeddings)
        ]
        for db_batch in batched(upload_rows, args.db_batch_size):
            upsert_rows(supabase, args.table, db_batch)
            uploaded += len(db_batch)
            print(f"Uploaded {uploaded}/{len(rows)}")

    print(f"Done. {uploaded} rows upserted into '{args.table}'.")


if __name__ == "__main__":
    main()
