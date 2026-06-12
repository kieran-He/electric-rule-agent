"""
处理单篇文章 - 使用 LLM 切分标签
输出到 data/llmprocessed 目录

用法：
python scripts/process_single_doc_llm.py <pdf_path> [--province SX]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json

from dataprocess.pdf_parser import parse_pdf
from dataprocess.cleaner import clean_document_pages_with_markers
from dataprocess.llm_splitter import split_into_clauses_with_llm
from dataprocess.metadata_extractor import extract_metadata, file_sha256
from dataprocess.metadata_llm_extractor import extract_metadata_with_llm
from dataprocess.schemas import ClauseChunk, ProcessedDocument, ProcessingStats
from dataprocess.config import settings
from dataprocess.llm_client import build_llm_config


PROCESSED_DIR = Path("data/llmprocessed")


def process_single_doc_with_llm(pdf_path: Path, province_code: str) -> tuple[bool, str]:
    """处理单个文档（使用 LLM 切分）"""
    try:
        file_hash = file_sha256(pdf_path)
        
        print(f"[1/4] 解析 PDF...")
        pages = parse_pdf(str(pdf_path))
        total_pages = len(pages)
        print(f"      共 {total_pages} 页")
        
        print(f"[2/4] 清洗文本...")
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
        print(f"      发布者: {metadata.issuer}")
        
        print(f"[4/5] LLM 提取 issuer/issue_date...")
        llm_cfg = build_llm_config(settings)
        llm_meta = extract_metadata_with_llm(
            text=marked_text,
            cfg=llm_cfg,
            max_chars=3000,
        )
        if llm_meta["issuer"]:
            metadata.issuer = llm_meta["issuer"]
            print(f"      LLM issuer: {llm_meta['issuer']}")
        if llm_meta["issue_date"]:
            metadata.issue_date = llm_meta["issue_date"]
            print(f"      LLM issue_date: {llm_meta['issue_date']}")
        
        print(f"[5/5] LLM 切分...")
        clauses = split_into_clauses_with_llm(
            text=marked_text,
            doc_name=metadata.doc_name,
            source_file=str(pdf_path),
            origin_doc_id=file_hash,
            cfg=llm_cfg,
            max_chars_per_call=6000,
        )
        print(f"      得到 {len(clauses)} 个条款")
        
        # Update clauses' doc_issuer
        if metadata.issuer:
            for clause in clauses:
                clause.doc_issuer = metadata.issuer
        
        doc = ProcessedDocument(
            metadata=metadata,
            cleaned_text=marked_text,
            clauses=clauses,
            stats=ProcessingStats(total_pages=total_pages, total_clauses=len(clauses)),
            processing_flags=["llm_split", f"pages_{total_pages}"],
        )
        
        output_dir = PROCESSED_DIR / province_code
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{file_hash}.json"
        output_file.write_text(
            json.dumps(doc.model_dump(mode='json'), ensure_ascii=False, indent=2),
            encoding='utf-8'
        )
        
        return True, f"成功: {total_pages}页, {len(clauses)}条, issuer={metadata.issuer}"
    except Exception as e:
        import traceback
        return False, f"失败: {type(e).__name__}: {str(e)}\n{traceback.format_exc()}"


def main():
    parser = argparse.ArgumentParser(description='使用 LLM 处理单个文档')
    parser.add_argument('pdf_path', type=str, help='PDF 文件路径')
    parser.add_argument('--province', type=str, default='SX', help='省份代码（默认 SX）')
    args = parser.parse_args()
    
    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"错误: 文件不存在 - {pdf_path}")
        sys.exit(1)
    
    if not settings.has_llm_config:
        print("错误: 缺少 LLM 配置，请设置 DOC_PROC_LLM_API_KEY 等环境变量")
        sys.exit(1)
    
    print("=" * 60)
    print("单文档处理 - LLM 切分")
    print("=" * 60)
    print(f"文件: {pdf_path}")
    print(f"省份: {args.province}")
    print(f"输出: {PROCESSED_DIR / args.province}")
    print()
    
    success, msg = process_single_doc_with_llm(pdf_path, args.province)
    
    print()
    print("=" * 60)
    if success:
        print("[成功]", msg)
    else:
        print("[失败]", msg)
    print("=" * 60)


if __name__ == '__main__':
    main()