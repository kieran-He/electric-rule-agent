"""
更新已处理文档的 issuer 和 issue_date 字段（使用 LLM 提取）

用法：
python scripts/update_metadata_with_llm.py [--province SX] [--limit 5]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json

from dataprocess.metadata_llm_extractor import extract_metadata_with_llm
from dataprocess.llm_client import build_llm_config
from dataprocess.config import settings
from dataprocess.schemas import ProcessedDocument


PROCESSED_DIR = Path("data/processed")
LLMPROCESSED_DIR = Path("data/llmprocessed")


def update_single_doc(json_path: Path) -> tuple[bool, str]:
    """Update single document's issuer and issue_date using LLM."""
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        doc = ProcessedDocument.model_validate(data)
        
        # Skip if both fields are already filled with high confidence
        if doc.metadata.issuer and doc.metadata.issue_date:
            return True, f"跳过: 已有issuer={doc.metadata.issuer}, issue_date={doc.metadata.issue_date}"
        
        print(f"\n处理: {json_path.name}")
        print(f"  当前 issuer: {doc.metadata.issuer}")
        print(f"  当前 issue_date: {doc.metadata.issue_date}")
        
        # Extract with LLM
        llm_cfg = build_llm_config(settings)
        result = extract_metadata_with_llm(
            text=doc.cleaned_text,
            cfg=llm_cfg,
            max_chars=3000,
        )
        
        print(f"  LLM issuer: {result['issuer']} (confidence: {result['issuer_confidence']})")
        print(f"  LLM issue_date: {result['issue_date']} (confidence: {result['date_confidence']})")
        
        # Update if LLM found something
        if result["issuer"]:
            doc.metadata.issuer = result["issuer"]
        if result["issue_date"]:
            doc.metadata.issue_date = result["issue_date"]
        
        # Update clauses' doc_issuer as well
        if result["issuer"]:
            for clause in doc.clauses:
                clause.doc_issuer = result["issuer"]
        
        # Save updated document
        json_path.write_text(
            json.dumps(doc.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        
        return True, f"更新: issuer={result['issuer']}, issue_date={result['issue_date']}"
    except Exception as e:
        import traceback
        return False, f"失败: {type(e).__name__}: {str(e)}\n{traceback.format_exc()}"


def main():
    parser = argparse.ArgumentParser(description="使用 LLM 更新文档的 issuer 和 issue_date")
    parser.add_argument("--province", type=str, help="只处理指定省份")
    parser.add_argument("--limit", type=int, help="限制处理文档数量")
    parser.add_argument("--input-dir", type=str, default="processed", help="输入目录: processed 或 llmprocessed")
    args = parser.parse_args()
    
    if not settings.has_llm_config:
        print("错误: 缺少 LLM 配置，请设置 DOC_PROC_LLM_API_KEY 等环境变量")
        sys.exit(1)
    
    input_dir = PROCESSED_DIR if args.input_dir == "processed" else LLMPROCESSED_DIR
    
    print("=" * 60)
    print("使用 LLM 更新文档元数据")
    print("=" * 60)
    print(f"输入目录: {input_dir}")
    
    # Find JSON files
    json_files = []
    if args.province:
        province_dir = input_dir / args.province
        if province_dir.exists():
            json_files = list(province_dir.glob("*.json"))
    else:
        for province_dir in input_dir.iterdir():
            if province_dir.is_dir():
                json_files.extend(province_dir.glob("*.json"))
    
    json_files = sorted(json_files)
    
    if args.limit:
        json_files = json_files[:args.limit]
    
    print(f"待处理文档: {len(json_files)}")
    
    if not json_files:
        print("没有找到 JSON 文件")
        return
    
    stats = {"success": 0, "failed": 0, "skipped": 0}
    
    for i, json_path in enumerate(json_files):
        print(f"\n[{i+1}/{len(json_files)}] {json_path}")
        success, msg = update_single_doc(json_path)
        
        if success:
            if "跳过" in msg:
                stats["skipped"] += 1
            else:
                stats["success"] += 1
            print(f"  {msg}")
        else:
            stats["failed"] += 1
            print(f"  {msg}")
    
    print("\n" + "=" * 60)
    print("处理完成")
    print("=" * 60)
    print(f"成功更新: {stats['success']}")
    print(f"跳过: {stats['skipped']}")
    print(f"失败: {stats['failed']}")


if __name__ == "__main__":
    main()