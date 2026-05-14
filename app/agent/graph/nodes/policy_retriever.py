from __future__ import annotations

import logging
from typing import Dict, TYPE_CHECKING

from app.agent.graph.state import ElectricityAgentState

if TYPE_CHECKING:
    from app.langchain.orchestrator_hybrid import HybridQAOrchestrator

logger = logging.getLogger(__name__)


def policy_retriever_node(state: ElectricityAgentState) -> Dict:
    query = state["query"]
    provinces = state["provinces"]
    
    from app.agent.graph.electricity_agent_graph import ElectricityAgentGraph
    graph_instance = ElectricityAgentGraph._get_current_instance()
    
    if not graph_instance:
        logger.warning("[PolicyRetriever] No graph instance available")
        return {
            "policy_chunks": [],
            "tool_calls": state.get("tool_calls", []) + ["rag"],
        }
    
    orchestrator = graph_instance.orchestrator
    
    try:
        chunks, detected_codes, quality = orchestrator._retrieve(
            query=query,
            province_codes=provinces,
            top_k=8,
        )
        
        logger.info(f"[PolicyRetriever] Retrieved {len(chunks)} chunks")
        
        policy_chunks = [
            {
                "content": chunk.text,
                "source": chunk.metadata.get("source_name", ""),
                "title_path": chunk.metadata.get("title_path", ""),
                "score": getattr(chunk, 'score', None),
            }
            for chunk in chunks
        ]
        
        return {
            "policy_chunks": policy_chunks,
            "tool_calls": state.get("tool_calls", []) + ["rag"],
        }
    except Exception as e:
        logger.exception(f"[PolicyRetriever] Failed: {e}")
        return {
            "policy_chunks": [],
            "tool_calls": state.get("tool_calls", []) + ["rag"],
        }


