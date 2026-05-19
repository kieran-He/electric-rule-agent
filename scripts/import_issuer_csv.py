"""
Import issuer and issue_date from CSV file to processed JSON documents.

Usage:
    python scripts/import_issuer_csv.py [--csv data/issuer.csv] [--processed-dir data/processed]

The CSV should have columns: province_code, file_hash, doc_name, issuer, issue_date
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional


def parse_issue_date(date_str: str) -> Optional[str]:
    """Parse issue_date string to ISO format."""
    if not date_str:
        return None
    
    date_str = date_str.strip()
    
    if len(date_str) == 8 and date_str.isdigit():
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    
    if "-" in date_str:
        parts = date_str.split("-")
        if len(parts) == 3:
            year, month, day = parts
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
    
    return date_str


def load_csv_data(csv_path: Path) -> dict[str, dict]:
    """Load CSV data indexed by file_hash."""
    data = {}
    
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_hash = row.get("file_hash", "").strip()
            if file_hash:
                issuer = row.get("issuer", "").strip() or None
                issue_date = parse_issue_date(row.get("issue_date", ""))
                
                data[file_hash] = {
                    "province_code": row.get("province_code", "").strip(),
                    "doc_name": row.get("doc_name", "").strip(),
                    "issuer": issuer,
                    "issue_date": issue_date,
                }
    
    return data


def update_processed_json(json_path: Path, csv_data: dict[str, dict]) -> bool:
    """Update a processed JSON file with issuer/issue_date from CSV."""
    file_hash = json_path.stem
    
    if file_hash not in csv_data:
        return False
    
    csv_row = csv_data[file_hash]
    
    with json_path.open("r", encoding="utf-8") as f:
        doc_data = json.load(f)
    
    metadata = doc_data.get("metadata", {})
    
    if csv_row.get("issuer"):
        metadata["issuer"] = csv_row["issuer"]
    
    if csv_row.get("issue_date"):
        metadata["issue_date"] = csv_row["issue_date"]
    
    doc_data["metadata"] = metadata
    
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(doc_data, f, ensure_ascii=False, indent=2)
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Import issuer CSV to processed JSONs")
    parser.add_argument("--csv", default="data/issuer.csv", help="Path to issuer CSV file")
    parser.add_argument("--processed-dir", default="data/processed", help="Path to processed directory")
    args = parser.parse_args()
    
    csv_path = Path(args.csv)
    processed_dir = Path(args.processed_dir)
    
    if not csv_path.exists():
        print(f"CSV file not found: {csv_path}")
        return
    
    csv_data = load_csv_data(csv_path)
    print(f"Loaded {len(csv_data)} entries from CSV")
    
    updated_count = 0
    
    for province_dir in processed_dir.iterdir():
        if not province_dir.is_dir():
            continue
        
        for json_file in province_dir.glob("*.json"):
            if json_file.name.startswith("_"):
                continue
            
            if update_processed_json(json_file, csv_data):
                updated_count += 1
                print(f"Updated: {json_file.relative_to(processed_dir)}")
    
    print(f"\nTotal updated: {updated_count} files")


if __name__ == "__main__":
    main()