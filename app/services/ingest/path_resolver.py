from pathlib import Path
from typing import Optional


def resolve_docs_path(
    kb_scope: str,
    province_code: Optional[str],
    docs_path: Optional[str],
    docs_root: Optional[str],
    default_docs_root: str,
) -> str:
    if docs_path:
        return docs_path
    root = Path(docs_root or default_docs_root)
    if kb_scope == "global":
        return str(root / "global")
    if kb_scope == "province":
        if not province_code:
            raise ValueError("province_code is required when kb_scope=province")
        return str(root / province_code.upper())
    raise ValueError(f"unsupported kb_scope: {kb_scope}")


def build_missing_path_hint(path: str, kb_scope: str, province_code: Optional[str]) -> str:
    if kb_scope == "global":
        return (
            f"docs_path not found: {path}. "
            "Expected global docs under data/docs/global or pass explicit docs_path."
        )
    province = (province_code or "SN").upper()
    return (
        f"docs_path not found: {path}. "
        f"Expected province docs under data/docs/{province} or pass explicit docs_path."
    )