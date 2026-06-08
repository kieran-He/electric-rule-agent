import json
import logging
from typing import List, Dict, Any

from dataprocess.province_mapping import PROVINCE_CODE_ALIASES as PROVINCE_CODE_NAME
from langchain_core.tools import tool

from app.agent.graph.tools.mock_data import generate_mock_policy_chunks

logger = logging.getLogger(__name__)


@tool
def retrieve_policy(query: str, provinces: List[str], show_chunks: bool = True) -> str:
    """
    Retrieve relevant policy documents for electricity market queries.
    
    Use this tool when the user asks about regulations, rules, policies,
    or requirements related to electricity trading, market access, pricing
    policies, or compliance matters.
    
    Args:
        query: The user's question about electricity policy
        provinces: List of province codes to search. 省份代码对照：
            陕西=SN, 山西=SX, 山东=SD, 河南=HA, 河北=HE, 海南=HI, 湖南=HN
            示例：["SN", "HA"] 表示陕西和河南
        show_chunks: Whether to include chunk references in output
        
    Returns:
        JSON string containing list of relevant policy chunks with content,
        source, title_path, and score fields, plus metadata about province coverage
    """
    logger.info(f"[PolicyTool] Retrieving policy for query: {query}, provinces: {provinces}, show_chunks: {show_chunks}")
    
    province_coverage = {}
    for code in provinces:
        province_coverage[code] = 0
    
    try:
        from app.agent.graph.electricity_agent_graph import _get_current_instance
        graph_instance = _get_current_instance()
        
        if graph_instance and graph_instance.orchestrator:
            orchestrator = graph_instance.orchestrator
            
            chunks, detected_codes, quality = orchestrator._retrieve(
                query=query,
                province_codes=provinces,
                top_k=8,
            )
            
            for chunk in chunks:
                source = chunk.metadata.get("source_name", "")
                for code in provinces:
                    province_name = PROVINCE_CODE_NAME.get(code, "")
                    if province_name in source:
                        province_coverage[code] += 1
            
            policy_chunks = [
                {
                    "content": chunk.text,
                    "source": chunk.metadata.get("source_name", ""),
                    "title_path": chunk.metadata.get("title_path", ""),
                    "score": getattr(chunk, 'score', None),
                    "issuer": chunk.metadata.get("issuer"),
                    "issue_date": chunk.metadata.get("issue_date"),
                    "effective_date": chunk.metadata.get("effective_date"),
                    "province_code": chunk.metadata.get("province_code", ""),
                }
                for chunk in chunks
            ]
            
            actual_province_codes = set(
                chunk.get("province_code", "").upper() 
                for chunk in policy_chunks 
                if chunk.get("province_code")
            )
            
            formatted_chunks = ""
            for i, chunk in enumerate(policy_chunks, 1):
                doc_name = chunk.get("source", "未知文档")
                title_path = chunk.get("title_path", "")
                content = chunk.get("content", "")
                formatted_chunks += f"\n\n[chunk-{i}] 《{doc_name}》\n"
                if title_path:
                    formatted_chunks += f"章节: {title_path}\n"
                formatted_chunks += f"内容: {content}\n"
            
            missing_provinces = [
                PROVINCE_CODE_NAME.get(code, code) 
                for code, count in province_coverage.items() 
                if count == 0
            ]
            
            logger.info(f"[PolicyTool] Retrieved {len(policy_chunks)} chunks from orchestrator, actual provinces: {actual_province_codes}")
            logger.info(f"[PolicyTool] Province coverage: {province_coverage}, missing: {missing_provinces}")
            
            if policy_chunks:
                result = {
                    "chunks": policy_chunks,
                    "detected_codes": detected_codes,
                    "actual_province_codes": list(actual_province_codes),
                    "formatted_chunks": formatted_chunks,
                    "province_coverage": province_coverage,
                    "missing_provinces": missing_provinces,
                    "quality": {
                        "is_low_quality": quality.is_low_quality,
                        "reason": quality.quality_reason,
                        "avg_score": quality.avg_score,
                        "chunk_count": quality.chunk_count,
                    }
                }
                return json.dumps(result, ensure_ascii=False)
        
    except Exception as e:
        logger.warning(f"[PolicyTool] Orchestrator retrieval failed: {e}, using mock data")
    
    mock_chunks = generate_mock_policy_chunks(query, provinces)
    logger.info(f"[PolicyTool] Using mock data: {len(mock_chunks)} chunks")
    
    return json.dumps({
        "chunks": mock_chunks,
        "province_coverage": province_coverage,
        "missing_provinces": list(PROVINCE_CODE_NAME.get(code, code) for code in provinces),
        "detected_codes": provinces,
    }, ensure_ascii=False)