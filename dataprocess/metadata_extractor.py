from __future__ import annotations

import hashlib
import re
from datetime import date
from pathlib import Path

from dataprocess.province_mapping import detect_province_code
from dataprocess.schemas import DocumentMetadata


STATUS_HINTS = {
    "征求意见稿": "draft",
    "意见稿": "draft",
    "连续试运行": "trial",
    "连续运行": "trial",
    "试运行": "trial",
    "通知": "notice",
    "正式": "formal",
}

DOC_TYPE_HINTS = {
    "实施方案": "implementation_plan",
    "工作方案": "work_plan",
    "实施细则": "implementation_rules",
    "交易细则": "trading_rules",
    "中长期": "trading_rules",
    "结算": "settlement_rules",
    "计量": "metering_rules",
    "调频": "ancillary_service_rules",
    "零售": "retail_rules",
}

MARKET_TYPE_HINTS = {
    "中长期": "中长期",
    "现货": "现货",
    "零售": "零售",
    "结算": "结算",
    "计量": "计量",
    "调频": "辅助服务",
    "储能": "储能",
    "虚拟电厂": "虚拟电厂",
}

SUBJECT_HINTS = ["售电公司", "批发用户", "虚拟电厂", "独立储能", "电网代理购电用户"]


def file_sha256(path: str | Path) -> str:
    target = Path(path)
    digest = hashlib.sha256()
    with target.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_date_from_name(name: str) -> date | None:
    full_match = re.search(r"(20\d{2})[年\-_./ ](\d{1,2})[月\-_./ ](\d{1,2})日?", name)
    if full_match:
        year, month, day = map(int, full_match.groups())
        return date(year, month, day)
    return None


def extract_year(name: str) -> int | None:
    match = re.search(r"(20\d{2})", name)
    return int(match.group(1)) if match else None


def extract_version_name(name: str) -> str | None:
    match = re.search(r"(V\d+(?:\.\d+)?)|(\d{4}年\d{1,2}月修订版)|(\d{4}年修订版)", name)
    if not match:
        return None
    return next(group for group in match.groups() if group)


def extract_issuer(name: str) -> str | None:
    """
    Extract issuer from file name.
    
    Patterns:
    - Look for keywords like '发展和改革委员会', '能源局', '电力交易中心', etc.
    """
    issuer_patterns = [
        r"([^\s,，]+发展和改革委员会)",
        r"([^\s,，]+发展和改革委)",
        r"([^\s,，]+能源局)",
        r"([^\s,，]+工信厅)",
        r"([^\s,，]+工业和信息化厅)",
        r"([^\s,，]+电力交易中心)",
        r"国家能源局[^\s,，]+监管[^\s,，]+",
        r"国家能源局[东南西北]+监管局",
    ]
    
    for pattern in issuer_patterns:
        match = re.search(pattern, name)
        if match:
            return match.group(0)
    
    return None


def extract_metadata(file_path: str, file_hash: str, province_code_override: str | None = None) -> DocumentMetadata:
    path = Path(file_path)
    stem = path.stem

    status = "formal"
    for hint, value in STATUS_HINTS.items():
        if hint in stem:
            status = value
            break

    doc_type = "unknown"
    for hint, value in DOC_TYPE_HINTS.items():
        if hint in stem:
            doc_type = value
            break

    market_type = "综合"
    for hint, value in MARKET_TYPE_HINTS.items():
        if hint in stem:
            market_type = value
            break

    subject_scope = [subject for subject in SUBJECT_HINTS if subject in stem]

    detected_code = detect_province_code(file_path, stem)
    province_code = province_code_override or detected_code

    return DocumentMetadata(
        province_code=province_code,
        doc_name=stem,
        doc_type=doc_type,
        market_type=market_type,
        subject_scope=subject_scope,
        version_name=extract_version_name(stem),
        status=status,
        issuer=extract_issuer(stem),
        issue_date=parse_date_from_name(stem),
        effective_date=None,
        source_file=str(path),
        file_hash=file_hash,
        is_current=status != "draft",
        parent_doc_id=None,
        year=extract_year(stem),
    )