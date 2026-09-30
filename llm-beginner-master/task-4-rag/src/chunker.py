"""字符级切分：chunk_size / overlap 以字符计。"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    # overlap 防止锚点刚好落在边界被切断
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须 > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap 须满足 0 <= overlap < chunk_size")

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    n = len(text)
    if n == 0:
        return []

    chunks: List[str] = []
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        piece = text[start:end]
        if piece.strip():
            chunks.append(piece)
        if end >= n:
            break
        start = end - overlap
        # 防止 overlap 导致死循环
        if start <= end - chunk_size and end < n:
            start = end
    return chunks


def recursive_chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
    separators: Optional[List[str]] = None,
) -> List[str]:
    # 优先按段落/句号拆，超长段再固定窗口
    if separators is None:
        separators = ["\n\n", "\n", "。", "！", "？", "；", " "]

    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    sep = separators[0] if separators else ""
    rest_seps = separators[1:] if separators else []

    if sep:
        parts = text.split(sep)
    else:
        return chunk_text(text, chunk_size, overlap)

    chunks: List[str] = []
    buf = ""
    for i, part in enumerate(parts):
        # split 会丢掉 sep，这里补回去（末段除外）
        piece = part + (sep if i < len(parts) - 1 else "")
        if len(buf) + len(piece) <= chunk_size:
            buf += piece
        else:
            if buf.strip():
                if len(buf) > chunk_size:
                    chunks.extend(
                        recursive_chunk_text(buf, chunk_size, overlap, rest_seps)
                        if rest_seps
                        else chunk_text(buf, chunk_size, overlap)
                    )
                else:
                    chunks.append(buf)
            if len(piece) > chunk_size:
                chunks.extend(
                    recursive_chunk_text(piece, chunk_size, overlap, rest_seps)
                    if rest_seps
                    else chunk_text(piece, chunk_size, overlap)
                )
                buf = ""
            else:
                # 简单 overlap：接下一块时带上上一块尾部
                if chunks and overlap > 0:
                    prev_tail = chunks[-1][-overlap:]
                    buf = prev_tail + piece
                else:
                    buf = piece
    if buf.strip():
        if len(buf) > chunk_size:
            chunks.extend(
                recursive_chunk_text(buf, chunk_size, overlap, rest_seps)
                if rest_seps
                else chunk_text(buf, chunk_size, overlap)
            )
        else:
            chunks.append(buf)
    return [c for c in chunks if c.strip()]


def extract_pdf_text(pdf_path: str | Path) -> str:
    # 索引必须来自 kb.pdf，不能直接用 LaTeX
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages):
        t = page.extract_text() or ""
        # 合并过量空白，保留换行方便递归切分
        t = re.sub(r"[ \t]+", " ", t)
        t = re.sub(r"\n{3,}", "\n\n", t)
        pages.append(f"\n\n--- page {i + 1} ---\n\n{t}")
    return "\n".join(pages).strip()


def chunk_pdf(
    pdf_path: str | Path,
    chunk_size: int = 512,
    overlap: int = 128,
    strategy: str = "recursive",
) -> List[dict]:
    text = extract_pdf_text(pdf_path)
    if strategy == "fixed":
        parts = chunk_text(text, chunk_size, overlap)
    else:
        parts = recursive_chunk_text(text, chunk_size, overlap)

    out = []
    for i, p in enumerate(parts):
        m = re.search(r"--- page (\d+) ---", p)
        page = m.group(1) if m else "?"
        out.append(
            {
                "chunk_id": i,
                "text": p,
                "source": f"kb.pdf#page={page}#chunk={i}",
            }
        )
    return out
