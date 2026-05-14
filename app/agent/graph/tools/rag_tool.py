from typing import Dict

from app.agent.graph.state import ElectricityAgentState


def rag_tool_node(state: ElectricityAgentState) -> Dict:
    return state