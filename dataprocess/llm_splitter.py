from __future__ import annotations

import json
import time
import re
from dataclasses import dataclass
from typing import Any

from dataprocess.cleaner import extract_page_range_from_text, remove_page_markers
from dataprocess.llm_client import call_llm_json, LLMConfig
from dataprocess.schemas import ClauseChunk, RuleTagExtraction


PAGE_MARKER_PATTERN = re.compile(r"⟦PAGE:(\d+)⟧")


SYSTEM_PROMPT = """你是电力市场规则文档结构解析器。
任务：把输入文本按"最小结构单元"切分。
输出必须是严格JSON，不要markdown。"""

USER_PROMPT_TEMPLATE = """请对以下带页码标记的文本做结构切分，返回JSON：
{
  "units": [
    {
      "raw_no": "原始编号，如第十八条/1.2.3/（一）/附件A，可为空字符串",
      "title": "标题，可为空字符串",
      "content": "该最小结构单元正文（移除⟦PAGE:N⟧标记），不能为空",
      "summary": "简洁摘要（不超过50字且少于content字数）",
      "page_start": 从⟦PAGE:N⟧提取的起始页码（数字）,
      "page_end": 从⟦PAGE:N⟧提取的结束页码（数字，跨页时填写）
    }
  ]
}

要求：
1) 只返回JSON，不能有markdown。
2) units至少1项。
3) content必须是原文片段，不要改写，移除所有⟦PAGE:N⟧标记。
4) summary不超过50字且必须少于content字数。
5) 页码从⟦PAGE:N⟧标记中提取，N为数字。
6) 编号层级混合时，优先保持原始编号。

文本如下：
__INPUT_TEXT__
"""


def _split_text_for_llm(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    
    parts = [part.strip() for part in text.split("\n\n") if part.strip()]
    
    if not parts:
        segments = []
        for i in range(0, len(text), max_chars):
            segments.append(text[i:i + max_chars])
        return segments
    
    segments: list[str] = []
    current: list[str] = []
    current_len = 0
    
    for part in parts:
        if len(part) > max_chars:
            if current:
                segments.append("\n\n".join(current))
                current = []
                current_len = 0
            for i in range(0, len(part), max_chars):
                segments.append(part[i:i + max_chars])
            continue
        
        part_len = len(part) + 2
        if current and current_len + part_len > max_chars:
            segments.append("\n\n".join(current))
            current = [part]
            current_len = part_len
        else:
            current.append(part)
            current_len += part_len
    
    if current:
        segments.append("\n\n".join(current))
    
    return segments


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    
    def try_parse_json(s: str) -> dict[str, Any] | None:
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return None
    
    result = try_parse_json(stripped)
    if result is not None:
        return result
    
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        json_str = stripped[start : end + 1]
        result = try_parse_json(json_str)
        if result is not None:
            return result
        
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        json_str = re.sub(r'[\x00-\x1f]', ' ', json_str)
        
        result = try_parse_json(json_str)
        if result is not None:
            return result
    
    raise json.JSONDecodeError("Failed to parse JSON from response", stripped, 0)


def _parse_structure_from_raw_no_title(raw_no: str, title: str | None) -> dict[str, str | None]:
    result = {
        "chapter_no": None,
        "chapter_title": None,
        "section_no": None,
        "section_title": None,
        "article_no": None,
    }
    
    if not raw_no:
        return result
    
    chapter_match = re.match(r"^第([一二三四五六七八九十]+)章", raw_no)
    if chapter_match:
        result["chapter_no"] = raw_no
        result["chapter_title"] = title
        return result
    
    if re.match(r"^([一二三四五六七八九十]+)$", raw_no):
        result["chapter_no"] = raw_no
        result["chapter_title"] = title
        return result
    
    if re.match(r"^([一二三四五六七八九十]+)、$", raw_no):
        result["chapter_no"] = raw_no.rstrip("、")
        result["chapter_title"] = title
        return result
    
    if re.match(r"^[（(][一二三四五六七八九十\d]+[）)]$", raw_no):
        result["section_no"] = raw_no
        result["section_title"] = title
        return result
    
    if re.match(r"^\d+\.\d+$", raw_no):
        result["section_no"] = raw_no
        result["section_title"] = title
        return result
    
    if re.match(r"^第.+?条", raw_no):
        result["article_no"] = raw_no
        return result
    
    return result


def _extract_rule_tags(text: str) -> RuleTagExtraction:
    rule = RuleTagExtraction()
    normalized_text = text.replace(" ", "")
    
    ENTITY_HINTS = ["售电公司", "批发用户", "虚拟电厂", "独立储能", "发电企业", "电力用户"]
    for entity in ENTITY_HINTS:
        if entity in text:
            rule.entity_type = entity
            break
    
    if "中长期" in normalized_text:
        rule.market_type = "中长期市场"
    elif "现货" in normalized_text:
        rule.market_type = "现货市场"
    elif "零售" in normalized_text:
        rule.market_type = "零售市场"
        rule.retail_related = True
    elif "辅助服务" in normalized_text:
        rule.market_type = "辅助服务市场"
    elif "容量" in normalized_text:
        rule.market_type = "容量市场"
    elif "储能" in normalized_text:
        rule.market_type = "储能市场"
    
    if "违约" in normalized_text or "考核" in normalized_text:
        rule.penalty_related = True
    if "绿电" in normalized_text:
        rule.green_power_related = True
    if "储能" in normalized_text:
        rule.storage_related = True
    if "虚拟电厂" in normalized_text:
        rule.vpp_related = True
    if "零售" in normalized_text:
        rule.retail_related = True
    
    return rule


def split_into_clauses_with_llm(
    *,
    text: str,
    doc_name: str,
    source_file: str,
    origin_doc_id: str | None,
    cfg: LLMConfig,
    max_chars_per_call: int = 4000,
    checkpoint_dir: str | None = None,
) -> list[ClauseChunk]:
    from pathlib import Path
    
    segments = _split_text_for_llm(text, max_chars=max_chars_per_call)
    all_units: list[dict[str, Any]] = []
    failures: list[str] = []
    
    print(f"[LLM] Splitting into {len(segments)} segments (max {max_chars_per_call} chars each)")
    
    checkpoint_path = None
    if checkpoint_dir and origin_doc_id:
        checkpoint_path = Path(checkpoint_dir) / f"{origin_doc_id}_checkpoint.json"
        if checkpoint_path.exists():
            try:
                checkpoint_data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                saved_units = checkpoint_data.get("units", [])
                saved_idx = checkpoint_data.get("last_segment_idx", -1)
                if saved_units:
                    all_units = saved_units
                    print(f"[LLM] Resumed from checkpoint: {len(all_units)} units, last segment {saved_idx + 1}")
            except Exception as e:
                print(f"[LLM] Failed to load checkpoint: {e}")
    
    start_idx = 0
    if all_units:
        for i, seg in enumerate(segments):
            seg_hash = str(hash(seg))[:8]
            if any(u.get("_seg_hash") == seg_hash for u in all_units):
                start_idx = i + 1
    
    for i, segment in enumerate(segments):
        if i < start_idx:
            continue
        try:
            if i > 0:
                time.sleep(2)
            print(f"[LLM] Processing segment {i + 1}/{len(segments)} ({len(segment)} chars)...")
            result = call_llm_json(
                cfg=cfg,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=USER_PROMPT_TEMPLATE.replace("__INPUT_TEXT__", segment),
            )
            units = result.get("units", [])
            seg_hash = str(hash(segment))[:8]
            if isinstance(units, list):
                for u in units:
                    u["_seg_hash"] = seg_hash
                all_units.extend(units)
                print(f"[LLM] Segment {i + 1} done, got {len(units)} units")
                
                if checkpoint_path:
                    try:
                        checkpoint_data = {
                            "units": all_units,
                            "last_segment_idx": i,
                            "total_segments": len(segments),
                            "source_file": source_file,
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        }
                        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                        checkpoint_path.write_text(json.dumps(checkpoint_data, ensure_ascii=False, indent=2), encoding="utf-8")
                    except Exception as e:
                        print(f"[LLM] Checkpoint save failed: {e}")
        except Exception as exc:
            failures.append(f"{type(exc).__name__}:{exc}")
            print(f"[LLM] Segment {i + 1} failed: {type(exc).__name__}")
            continue

    if checkpoint_path and checkpoint_path.exists():
        try:
            checkpoint_path.unlink()
            print(f"[LLM] Checkpoint cleaned up")
        except Exception:
            pass

    if not all_units:
        reason = failures[0] if failures else "no_units_returned"
        raise RuntimeError(f"LLM split returned no units: {reason}")

    clauses: list[ClauseChunk] = []
    for unit in all_units:
        raw_no = (unit.get("raw_no") or "").strip()
        title = (unit.get("title") or "").strip()
        content_raw = (unit.get("content") or "").strip()
        summary_raw = (unit.get("summary") or "").strip()

        if not content_raw:
            continue

        content = remove_page_markers(content_raw)

        page_start_raw = unit.get("page_start")
        page_end_raw = unit.get("page_end")

        if page_start_raw is not None:
            page_start = int(page_start_raw)
        else:
            page_start, _ = extract_page_range_from_text(content_raw)

        if page_end_raw is not None:
            page_end = int(page_end_raw)
        else:
            _, page_end = extract_page_range_from_text(content_raw)

        if summary_raw and len(summary_raw) < len(content) and len(summary_raw) <= 50:
            summary = summary_raw
        else:
            summary = content[:50]

        path_parts = []
        if raw_no:
            path_parts.append(raw_no)
        if title:
            path_parts.append(title)
        title_path = " > ".join(path_parts) if path_parts else "未归类条款"

        structure_info = _parse_structure_from_raw_no_title(raw_no, title)

        if not structure_info["article_no"]:
            if raw_no:
                match = re.search(r"(第.+?条|\d+(?:\.\d+){0,4}|[（(].+?[）)])", raw_no)
                if match:
                    structure_info["article_no"] = match.group(1)

        clauses.append(
            ClauseChunk(
                doc_name=doc_name,
                source_file=source_file,
                origin_doc_id=origin_doc_id,
                chapter_no=structure_info["chapter_no"],
                chapter_title=structure_info["chapter_title"],
                section_no=structure_info["section_no"],
                section_title=structure_info["section_title"],
                article_no=structure_info["article_no"],
                title_path=title_path,
                clause_text=content,
                clause_summary=summary,
                page_start=page_start,
                page_end=page_end,
                token_count=max(1, len(content) // 2),
                rule_tags=_extract_rule_tags(content),
            )
        )
    return clauses