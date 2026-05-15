from app.agent.graph.tools.rag_tool import rag_tool_node
from app.agent.graph.tools.electricity_data_tool import electricity_data_tool_node
from app.agent.graph.tools.analysis_tool import analysis_tool_node

from app.agent.graph.tools.policy_tool import retrieve_policy
from app.agent.graph.tools.data_tool import fetch_electricity_data
from app.agent.graph.tools.analysis_tool import analyze_statistics
from app.agent.graph.tools.tool_registry import (
    get_all_tools,
    get_tools_for_agent,
    get_tools_by_names,
    get_tool_info,
    tool_exists,
    ALL_TOOLS,
)
from app.agent.graph.tools.mock_data import (
    generate_mock_electricity_data,
    generate_mock_policy_chunks,
)

REACT_TOOLS = [retrieve_policy, fetch_electricity_data, analyze_statistics]