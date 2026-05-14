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
    policy_chunks: List[Dict]
    electricity_data: Optional[Dict]
    analysis_result: Optional[Dict]
    answer: str
    metadata: Dict[str, Any]
    tool_calls: List[str]
    confidence: float