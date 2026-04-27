from dataclasses import dataclass
from typing import Dict, Optional


PROVINCE_ALIASES: Dict[str, str] = {
    "北京": "BJ",
    "天津": "TJ",
    "河北": "HE",
    "山西": "SX",
    "内蒙古": "NM",
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
}


@dataclass
class ProvinceDetection:
    province_code: Optional[str]
    confidence: float
    matched_text: Optional[str]


class ProvinceDetector:
    def detect(self, text: str) -> ProvinceDetection:
        candidates = []
        for alias, code in PROVINCE_ALIASES.items():
            if alias in text:
                candidates.append((alias, code))
        if not candidates:
            return ProvinceDetection(None, 0.0, None)
        if len(candidates) > 1:
            return ProvinceDetection(candidates[0][1], 0.6, candidates[0][0])
        alias, code = candidates[0]
        return ProvinceDetection(code, 0.95, alias)