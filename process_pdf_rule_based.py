"""
规则分段处理PDF - 不使用LLM
用法: python process_pdf_rule_based.py
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from dataprocess.pdf_parser import parse_pdf
from dataprocess.cleaner import clean_document_pages_with_markers
from dataprocess.llm_splitter import _extract_rule_tags
from dataprocess.metadata_extractor import extract_metadata, file_sha256
from dataprocess.schemas import ClauseChunk, ProcessedDocument, ProcessingStats


def split_text_rule_based(text: str, chunk_size: int = 400, chunk_overlap: int = 60) -> list[dict]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[dict] = []
    current = ""
    
    for para in paragraphs:
        if len(para) > chunk_size:
            if current:
                chunks.append({"text": current})
                current = ""
            start = 0
            while start < len(para):
                end = min(start + chunk_size, len(para))
                chunks.append({"text": para[start:end]})
                if end == len(para):
                    break
                start = max(end - chunk_overlap, start + 1)
            continue
        
        if not current:
            current = para
            continue
        
        candidate = f"{current}\n\n{para}"
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            chunks.append({"text": current})
            overlap_text = current[-chunk_overlap:] if chunk_overlap > 0 else ""
            current = (overlap_text + "\n\n" + para).strip()
            if len(current) > chunk_size:
                chunks.append({"text": current[:chunk_size]})
                current = current[chunk_size - chunk_overlap:]
    
    if current:
        chunks.append({"text": current})
    
    return [c for c in chunks if len(c["text"]) >= 40]


def process_pdf_rule_based(
    file_path: str,
    chunk_size: int = 400,
    province_code: str = "SX",
) -> ProcessedDocument:
    path = Path(file_path)
    
    print(f"[1/5] 解析PDF: {path.name}")
    pages = parse_pdf(str(path))
    print(f"      共 {len(pages)} 页")
    
    print(f"[2/5] 清理文本...")
    marked_text = clean_document_pages_with_markers(pages)
    print(f"      清理后文本长度: {len(marked_text)} 字符")
    
    print(f"[3/5] 规则分段 (chunk_size={chunk_size})...")
    chunks = split_text_rule_based(marked_text, chunk_size=chunk_size)
    print(f"      生成 {len(chunks)} 个chunks")
    
    print(f"[4/5] 提取元数据...")
    file_hash = file_sha256(path)
    metadata = extract_metadata(
        file_path=str(path),
        file_hash=file_hash,
        province_code_override=province_code,
        doc_text=marked_text[:500],
    )
    
    print(f"[5/5] 规则标签提取...")
    clauses: list[ClauseChunk] = []
    for i, chunk in enumerate(chunks):
        rule_tags = _extract_rule_tags(chunk["text"])
        
        clause = ClauseChunk(
            doc_name=metadata.doc_name,
            source_file=str(path),
            origin_doc_id=file_hash,
            province_code=province_code,
            doc_type=metadata.doc_type,
            doc_status=metadata.status,
            doc_issuer=metadata.issuer,
            title_path=f"chunk_{i+1}",
            clause_text=chunk["text"],
            clause_summary=chunk["text"][:50] if len(chunk["text"]) > 50 else chunk["text"],
            page_start=1,
            page_end=len(pages),
            token_count=max(1, len(chunk["text"]) // 2),
            rule_tags=rule_tags,
        )
        clauses.append(clause)
    
    stats = ProcessingStats(
        total_pages=len(pages),
        total_clauses=len(clauses),
    )
    
    document = ProcessedDocument(
        metadata=metadata,
        cleaned_text=marked_text,
        clauses=clauses,
        stats=stats,
        ocr_used=False,
        processing_flags=["rule_based_split", "rule_based_tag", f"chunk_size_{chunk_size}"],
    )
    
    return document


if __name__ == "__main__":
    pdf_path = r"data\docs\SX\山西电力市场规则体系（V16.0）.pdf"
    output_dir = Path("data/processed/SX")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    document = process_pdf_rule_based(
        file_path=pdf_path,
        chunk_size=400,
        province_code="SX",
    )
    
    output_file = output_dir / f"{document.metadata.file_hash}.json"
    output_file.write_text(
        json.dumps(document.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    
    print(f"\n处理完成!")
    print(f"  总页数: {document.stats.total_pages}")
    print(f"  总chunks: {document.stats.total_clauses}")
    print(f"  输出文件: {output_file}")