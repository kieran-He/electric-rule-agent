import json
from typing import Dict, Any, List, Optional
from unittest.mock import MagicMock


class MockDataAdapter:
    def __init__(self, data: Dict[str, Any] = None, should_fail: bool = False):
        self.data = data or {
            "province": "SN",
            "metric": "load",
            "data": [1000.0, 1100.0, 1200.0],
            "data_points": 3,
        }
        self.should_fail = should_fail
        self._call_count = 0
    
    def fetch(self, province: str, metric: str, time_range: str) -> Dict[str, Any]:
        self._call_count += 1
        if self.should_fail:
            raise Exception("数据获取失败")
        result = self.data.copy()
        result["province"] = province
        result["metric"] = metric
        return result
    
    def get_call_count(self) -> int:
        return self._call_count


class MockOrchestrator:
    def __init__(self, chunks: List[Any] = None, should_fail: bool = False):
        self.chunks = chunks or []
        self.should_fail = should_fail
        self._call_count = 0
    
    def _retrieve(self, query: str, provinces: List[str]) -> tuple:
        self._call_count += 1
        if self.should_fail:
            raise Exception("检索失败")
        return (self.chunks, provinces, {"quality": "high"})
    
    def get_call_count(self) -> int:
        return self._call_count


def create_mock_chunk(text: str, source: str = "test_source", score: float = 0.85) -> MagicMock:
    chunk = MagicMock()
    chunk.text = text
    chunk.metadata = {"source_name": source, "title_path": "测试文档"}
    chunk.score = score
    return chunk


class MockGraphInstance:
    def __init__(
        self,
        llm_wrapper: Any = None,
        orchestrator: Any = None,
        data_adapter: Any = None,
        settings: Any = None,
    ):
        self.llm_wrapper = llm_wrapper
        self.orchestrator = orchestrator
        self.data_adapter = data_adapter
        self.settings = settings or MagicMock()
        self.settings.tools_enabled_list = None
        self.settings.agent_use_react = True
        self.settings.agent_max_iterations = 5
        self.settings.agent_tool_timeout = 30
        
        from app.agent.graph.handlers.iteration_control import IterationController
        self._iteration_controller = IterationController(
            max_iterations=5,
            timeout_seconds=30,
        )
    
    @staticmethod
    def create_default() -> "MockGraphInstance":
        from tests.agent.fixtures.mock_llm import MockLLMWrapper
        return MockGraphInstance(
            llm_wrapper=MockLLMWrapper(),
            orchestrator=MockOrchestrator(),
            data_adapter=MockDataAdapter(),
        )