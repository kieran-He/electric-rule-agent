from typing import TypedDict, List, Dict, Any, Optional, Annotated
from langgraph.graph.message import add_messages


class ElectricityAgentState(TypedDict):
    messages: Annotated[List[Dict], add_messages]
    query: str
    intent: str
    intent_confidence: float
    intent_reason: str
    sub_intents: List[str]
    provinces: List[str]
    
    thoughts: List[Dict]
    iteration_count: int
    max_iterations: int
    
    tool_calls: List[Dict]
    tool_results: List[Dict]
    last_tool_calls: List[Dict]
    
    # 信息充足度标记
    sufficient_info: bool
    sufficiency_reason: str
    
    # 检索质量信息
    retrieval_quality: Optional[Dict]
    need_web_search: bool
    
    policy_chunks: List[Dict]
    electricity_data: Optional[Dict]
    analysis_result: Optional[Dict]
    chart_paths: List[str]
    
    answer: str
    citations: List[Dict]
    confidence: float
    done: bool
    
    metadata: Dict[str, Any]
    errors: List[Dict]


def create_initial_state(
    query: str,
    provinces: List[str] = None,
    session_id: str = None,
    history: List[Dict] = None,
    context: Dict[str, Any] = None,
    max_iterations: int = 5,
) -> ElectricityAgentState:
    import time
    return {
        "query": query,
        "provinces": provinces or ["SN"],
        "messages": history or [],
        "metadata": {
            "session_id": session_id,
            "context": context or {},
            "start_time": time.time(),
        },
        "intent": "",
        "intent_confidence": 0.0,
        "intent_reason": "",
        "sub_intents": [],
        "thoughts": [],
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "tool_calls": [],
        "tool_results": [],
        "last_tool_calls": [],
        "sufficient_info": False,
        "sufficiency_reason": "",
        "retrieval_quality": None,
        "need_web_search": False,
        "policy_chunks": [],
        "electricity_data": None,
        "analysis_result": None,
        "chart_paths": [],
        "answer": "",
        "citations": [],
        "confidence": 0.0,
        "done": False,
        "errors": [],
    }