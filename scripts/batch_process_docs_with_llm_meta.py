"""
批量文档处理脚本 - 使用规则切分 + LLM元数据提取

功能：
1. 使用规则切分（快速可靠）
2. 使用 LLM 提取 issuer 和 issue_date
3. 输出到 data/processed 目录

用法：
python scripts/batch_process_docs_with_llm_meta.py [--province JB] [--limit 1]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json

from dataprocess.pdf_parser import parse_pdf
from dataprocess.cleaner import clean_document_pages_with_markers
from dataprocess.metadata_extractor import extract_metadata, file_sha256
from dataprocess.metadata_llm_extractor import extract_metadata_with_llm
from dataprocess.llm_splitter import _extract_rule_tags
from process_pdf_rule_based import split_text_rule_based
from dataprocess.schemas import ClauseChunk, ProcessedDocument, ProcessingStats, DocumentMetadata
from dataprocess.config import settings
from dataprocess.llm_client import build_llm_config


PROCESSED_DIR = Path("data/processed")


def process_single_doc(pdf_path: Path, province_code: str) -> tuple[bool, str]:
    """处理单个文档（规则切分 + LLM元数据）"""
    try:
        file_hash = file_sha256(pdf_path)
        
        print(f"[1/5] 解析 PDF...")
        pages = parse_pdf(str(pdf_path))
        total_pages = len(pages)
        print(f"      共 {total_pages} 页")
        
        print(f"[2/5] 清洗文本...")
        marked_text = clean_document_pages_with_markers(pages)
        print(f"      文本长度: {len(marked_text)} 字符")
        
        print(f"[3/5] 提取基础元数据...")
        metadata = extract_metadata(
            file_path=str(pdf_path),
            file_hash=file_hash,
            province_code_override=province_code,
            doc_text=marked_text[:500],
        )
        print(f"      文档名: {metadata.doc_name}")
        
        print(f"[4/5] LLM 提取 issuer/issue_date...")
        if settings.has_llm_config:
            llm_cfg = build_llm_config(settings)
            llm_meta = extract_metadata_with_llm(
                text=marked_text[:3000],
                cfg=llm_cfg,
                max_chars=3000,
            )
            if llm_meta["issuer"]:
                metadata.issuer = llm_meta["issuer"]
                print(f"      LLM issuer: {llm_meta['issuer']}")
            if llm_meta["issue_date"]:
                metadata.issue_date = llm_meta["issue_date"]
                print(f"      LLM issue_date: {llm_meta['issue_date']}")
        else:
            print(f"      跳过（无LLM配置）")
        
        print(f"[5/5] 规则切分...")
        raw_chunks = split_text_rule_based(marked_text, chunk_size=400, chunk_overlap=60)
        print(f"      得到 {len(raw_chunks)} 个chunk")
        
        clauses = []
        for i, chunk in enumerate(raw_chunks):
            rule_tags = _extract_rule_tags(chunk["text"])
            clause = ClauseChunk(
                doc_name=metadata.doc_name,
                source_file=str(pdf_path),
                origin_doc_id=file_hash,
                province_code=province_code,
                doc_type=metadata.doc_type,
                doc_status=metadata.status,
                doc_issuer=metadata.issuer,
                title_path=f"chunk_{i+1}",
                clause_text=chunk["text"],
                clause_summary=chunk["text"][:50] if len(chunk["text"]) > 50 else chunk["text"],
                page_start=1,
                page_end=total_pages,
                token_count=max(1, len(chunk["text"]) // 2),
                rule_tags=rule_tags,
            )
            clauses.append(clause)
        
        doc = ProcessedDocument(
            metadata=metadata,
            cleaned_text=marked_text,
            clauses=clauses,
            stats=ProcessingStats(total_pages=total_pages, total_clauses=len(clauses)),
            processing_flags=["rule_split", "llm_metadata", f"pages_{total_pages}"],
        )
        
        output_dir = PROCESSED_DIR / province_code
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{file_hash}.json"
        output_file.write_text(
            json.dumps(doc.model_dump(mode='json'), ensure_ascii=False, indent=2),
            encoding='utf-8'
        )
        
        return True, f"成功: {total_pages}页, {len(clauses)}条, issuer={metadata.issuer}, date={metadata.issue_date}"
    except Exception as e:
        import traceback
        return False, f"失败: {type(e).__name__}: {str(e)}\n{traceback.format_exc()}"


def get_province_docs(base_dir: Path = Path("data/docs")) -> dict[str, list[Path]]:
    """获取各省文档列表"""
    provinces = {}
    for province_dir in base_dir.iterdir():
        if province_dir.is_dir():
            pdf_files = list(province_dir.glob("*.pdf"))
            if pdf_files:
                provinces[province_dir.name] = sorted(pdf_files)
    return provinces


def main():
    parser = argparse.ArgumentParser(description='批量处理文档（规则切分 + LLM元数据）')
    parser.add_argument('--province', type=str, help='只处理指定省份')
    parser.add_argument('--limit', type=int, help='限制处理文档数量')
    args = parser.parse_args()
    
    print("=" * 60)
    print("批量文档处理 - 规则切分 + LLM元数据")
    print("=" * 60)
    
    all_provinces = get_province_docs()
    
    if args.province:
        provinces = {k: v for k, v in all_provinces.items() if k == args.province}
    else:
        provinces = all_provinces
    
    total_files = sum(len(files) for files in provinces.values())
    print(f"总文档数: {total_files}")
    
    pending_files = []
    for prov, files in provinces.items():
        for f in files:
            pending_files.append((prov, f))
    
    if args.limit:
        pending_files = pending_files[:args.limit]
        print(f"限制处理: {len(pending_files)}")
    
    stats = {"success": 0, "failed": 0}
    
    for i, (province, pdf_path) in enumerate(pending_files):
        print(f"\n[{i+1}/{len(pending_files)}] {province}/{pdf_path.name}")
        print(f"  大小: {pdf_path.stat().st_size / 1024:.1f} KB")
        
        success, msg = process_single_doc(pdf_path, province)
        
        if success:
            stats["success"] += 1
            print(f"  {msg}")
        else:
            stats["failed"] += 1
            print(f"  {msg}")
    
    print("\n" + "=" * 60)
    print("处理完成")
    print("=" * 60)
    print(f"成功: {stats['success']}")
    print(f"失败: {stats['failed']}")


if __name__ == '__main__':
    main()