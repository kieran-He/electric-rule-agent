from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class FeishuDocLinksManager:
    _instance: Optional[FeishuDocLinksManager] = None
    _links: dict[str, str] = {}

    def __new__(cls, config_path: str | None = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_path: str | None = None):
        if self._initialized:
            return
        self._initialized = True
        self._config_path = config_path
        self._load_links()

    def _load_links(self) -> None:
        if not self._config_path:
            logger.debug("No feishu doc links config path specified")
            return

        path = Path(self._config_path)
        if not path.exists():
            logger.info(f"Feishu doc links config not found: {path}")
            return

        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            self._links = data.get("doc_links", {})
            logger.info(f"Loaded {len(self._links)} feishu doc links from {path}")
        except Exception as e:
            logger.warning(f"Failed to load feishu doc links: {e}")
            self._links = {}

    def get_link(self, doc_name: str) -> Optional[str]:
        if doc_name in self._links:
            return self._links[doc_name]

        doc_name_normalized = doc_name.replace(" ", "").replace("　", "")
        for key, url in self._links.items():
            key_normalized = key.replace(" ", "").replace("　", "")
            if key_normalized == doc_name_normalized:
                return url
            if key_normalized in doc_name_normalized or doc_name_normalized in key_normalized:
                return url
            
            doc_keywords = self._extract_keywords(doc_name_normalized)
            key_keywords = self._extract_keywords(key_normalized)
            if doc_keywords and key_keywords:
                match_count = sum(1 for kw in doc_keywords if kw in key_keywords)
                min_required = max(2, int(len(doc_keywords) * 0.6))
                if match_count >= min_required:
                    return url

        return None
    
    def _extract_keywords(self, text: str) -> list[str]:
        keywords = []
        province_keywords = ["陕西", "甘肃", "山西", "山东", "安徽"]
        for p in province_keywords:
            if p in text:
                keywords.append(p)
        
        market_keywords = ["电力现货", "现货市场", "电力市场", "中长期", "结算", "交易", "计量", "调频", "辅助服务", "零售", "省间"]
        for m in market_keywords:
            if m in text:
                keywords.append(m)
        
        doc_type_keywords = ["实施细则", "细则", "规则", "办法", "方案", "通知", "规定", "工作方案"]
        for d in doc_type_keywords:
            if d in text:
                keywords.append(d)
        
        version_keywords = ["V2", "V3", "2025", "2026", "试行", "征求意见", "试运行", "修订"]
        for v in version_keywords:
            if v in text:
                keywords.append(v)
        
        return keywords

    def reload(self) -> None:
        self._load_links()

    @property
    def links(self) -> dict[str, str]:
        return self._links.copy()


_feishu_doc_links: Optional[FeishuDocLinksManager] = None
_last_config_path: Optional[str] = None


def get_feishu_doc_links(config_path: str | None = None) -> FeishuDocLinksManager:
    global _feishu_doc_links, _last_config_path
    
    if _feishu_doc_links is None:
        _feishu_doc_links = FeishuDocLinksManager(config_path)
        _last_config_path = config_path
    elif config_path and config_path != _last_config_path:
        _feishu_doc_links._config_path = config_path
        _feishu_doc_links.reload()
        _last_config_path = config_path
    
    return _feishu_doc_links