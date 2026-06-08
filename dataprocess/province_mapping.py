from __future__ import annotations

from pathlib import Path
from typing import List

PROVINCE_ALIASES: dict[str, str] = {
    "北京": "BJ",
    "天津": "TJ",
    "河北": "HE",
    "冀南": "JN",
    "冀北": "JB",
    "山西": "SX",
    "内蒙古": "NM",
    "蒙西": "MX",
    "蒙东": "MD",
    "辽宁": "LN",
    "吉林": "JL",
    "黑龙江": "HL",
    "上海": "SH",
    "江苏": "JS",
    "浙江": "ZJ",
    "安徽": "AH",
    "福建": "FJ",
    "江西": "JX",
    "山东": "SD",
    "河南": "HA",
    "湖北": "HB",
    "湖南": "HN",
    "广东": "GD",
    "广西": "GX",
    "海南": "HI",
    "重庆": "CQ",
    "四川": "SC",
    "贵州": "GZ",
    "云南": "YN",
    "西藏": "XZ",
    "陕西": "SN",
    "甘肃": "GS",
    "青海": "QH",
    "宁夏": "NX",
    "新疆": "XJ",
    "全国": "QG",
}

PROVINCE_CODE_ALIASES: dict[str, str] = {code: name for name, code in PROVINCE_ALIASES.items()}

PROVINCE_EXPANSION: dict[str, List[str]] = {
    "NM": ["MX", "MD"],
    "HE": ["JN", "JB"],
    "HI": ["HAN"],
    "HA": ["HEN"],
    "HN": ["HUN"],
}


def detect_province_code(file_path: str | Path, file_name: str | None = None) -> str:
    path = Path(file_path)
    stem = file_name if file_name else path.stem
    
    for alias, code in PROVINCE_ALIASES.items():
        if alias in stem:
            return code
    
    for part in path.parts:
        upper_part = part.upper()
        if upper_part in PROVINCE_CODE_ALIASES:
            return upper_part
    
    for part in path.parts:
        for alias, code in PROVINCE_ALIASES.items():
            if alias in part:
                return code
    
    return "UNKNOWN"


def get_province_name(code: str) -> str | None:
    return PROVINCE_CODE_ALIASES.get(code.upper())


def get_all_province_codes() -> list[str]:
    return list(PROVINCE_CODE_ALIASES.keys())