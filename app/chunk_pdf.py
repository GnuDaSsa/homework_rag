import argparse
import json
import re
from pathlib import Path
from typing import Any

import pdfplumber


DOCUMENT_KEY = "seongnam_youth_hackathon_2026"
DOCUMENT_TITLE = "2026 성남시 정책 아이디어 청년 해커톤 대회 개최 공고"
ID_BASE = 202607070000


def clean_page_text(text: str) -> str:
    lines = []
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue
        if re.fullmatch(r"-\s*\d+\s*-", line):
            continue
        lines.append(line)
    return "\n".join(lines)


def extract_pages(pdf_path: Path) -> list[dict[str, Any]]:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            text = clean_page_text(page.extract_text(x_tolerance=1, y_tolerance=3) or "")
            if text:
                pages.append({"page": page_index, "text": text})
    return pages


def section_title_from_text(text: str) -> str:
    for line in text.splitlines():
        if re.match(r"^\d+\.\s+", line):
            return line
        if line.startswith("붙임") or line.startswith("[") or line.startswith("성남시 공고"):
            return line
    return "본문"


def split_long_text(text: str, max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(current) + len(paragraph) + 2 <= max_chars:
            current = f"{current}\n\n{paragraph}".strip()
            continue
        if current:
            chunks.append(current)
        if len(paragraph) <= max_chars:
            current = paragraph
            continue
        start = 0
        while start < len(paragraph):
            chunks.append(paragraph[start : start + max_chars])
            start += max_chars - overlap
        current = ""
    if current:
        chunks.append(current)

    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    overlapped = [chunks[0]]
    for previous, chunk in zip(chunks, chunks[1:]):
        tail = previous[-overlap:].strip()
        overlapped.append(f"{tail}\n\n{chunk}" if tail else chunk)
    return overlapped


def merge_short_sections(sections: list[str], min_chars: int = 260, max_chars: int = 1200) -> list[str]:
    merged: list[str] = []
    buffer = ""
    for section in sections:
        candidate = f"{buffer}\n\n{section}".strip() if buffer else section
        if buffer and (len(buffer) < min_chars or len(candidate) <= max_chars):
            buffer = candidate
            continue
        if buffer:
            merged.append(buffer)
        buffer = section
    if buffer:
        if merged and len(buffer) < min_chars and len(merged[-1]) + len(buffer) + 2 <= max_chars:
            merged[-1] = f"{merged[-1]}\n\n{buffer}"
        else:
            merged.append(buffer)
    return merged


def build_chunks(pages: list[dict[str, Any]], max_chars: int = 950, overlap: int = 120) -> list[dict[str, Any]]:
    chunks = []
    chunk_index = 1
    for page in pages:
        sections = re.split(r"(?=^\d+\.\s+)", page["text"], flags=re.MULTILINE)
        sections = merge_short_sections([section.strip() for section in sections if section.strip()])
        for section in sections:
            title = section_title_from_text(section)
            for part_index, content in enumerate(split_long_text(section, max_chars=max_chars, overlap=overlap), start=1):
                chunks.append(
                    {
                        "chunk_id": ID_BASE + chunk_index,
                        "content": content,
                        "metadata": {
                            "document_key": DOCUMENT_KEY,
                            "document_title": DOCUMENT_TITLE,
                            "source_type": "pdf",
                            "page_start": page["page"],
                            "page_end": page["page"],
                            "section_title": title,
                            "part": part_index,
                            "source_path": f"{DOCUMENT_TITLE} > p.{page['page']} > {title}",
                        },
                    }
                )
                chunk_index += 1
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="성남시 청년 해커톤 공고 PDF를 RAG용 JSON 청크로 변환합니다.")
    parser.add_argument("--input", required=True, help="PDF 파일 경로")
    parser.add_argument("--output", required=True, help="출력 JSON 경로")
    parser.add_argument("--max-chars", type=int, default=950)
    parser.add_argument("--overlap", type=int, default=120)
    args = parser.parse_args()

    pdf_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    pages = extract_pages(pdf_path)
    chunks = build_chunks(pages, max_chars=args.max_chars, overlap=args.overlap)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"pages": len(pages), "chunks": len(chunks), "output": str(output_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
