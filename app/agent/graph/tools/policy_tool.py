import json
import logging
from typing import List, Dict, Any

from langchain_core.tools import tool

from app.agent.graph.tools.mock_data import generate_mock_policy_chunks

logger = logging.getLogger(__name__)


@tool
def retrieve_policy(query: str, provinces: List[str]) -> str:
    """
    Retrieve relevant policy documents for electricity market queries.
    
    Use this tool when the user asks about regulations, rules, policies,
    or requirements related to electricity trading, market access, pricing
    policies, or compliance matters.
    
    Args:
        query: The user's question about electricity policy
        provinces: List of province codes to search (e.g., ["SN", "SX"])
        
    Returns:
        JSON string containing list of relevant policy chunks with content,
        source, title_path, and score fields
    """
    logger.info(f"[PolicyTool] Retrieving policy for query: {query}, provinces: {provinces}")
    
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
            
            policy_chunks = [
                {
                    "content": chunk.text,
                    "source": chunk.metadata.get("source_name", ""),
                    "title_path": chunk.metadata.get("title_path", ""),
                    "score": getattr(chunk, 'score', None),
                }
                for chunk in chunks
            ]
            
            logger.info(f"[PolicyTool] Retrieved {len(policy_chunks)} chunks from orchestrator")
            
            if policy_chunks:
                return json.dumps(policy_chunks, ensure_ascii=False)
        
    except Exception as e:
        logger.warning(f"[PolicyTool] Orchestrator retrieval failed: {e}, using mock data")
    
    mock_chunks = generate_mock_policy_chunks(query, provinces)
    logger.info(f"[PolicyTool] Using mock data: {len(mock_chunks)} chunks")
    
    return json.dumps(mock_chunks, ensure_ascii=False)