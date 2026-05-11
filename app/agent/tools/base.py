from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ToolResult:
    success: bool
    output: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    citations: List[Any] = field(default_factory=list)
    tool_name: str = ""
    confidence: float = 0.0


class BaseTool(ABC):
    name: str
    description: str
    keywords: List[str] = field(default_factory=list)
    
    def __init__(self):
        self._context: Dict[str, Any] = {}
    
    def set_context(self, context: Dict[str, Any]) -> None:
        """Set execution context before running tool."""
        self._context = context
    
    def get_context(self) -> Dict[str, Any]:
        """Get current execution context."""
        return self._context
    
    @abstractmethod
    def execute(self, query: str, context: Dict[str, Any] = None) -> ToolResult:
        pass
    
    @abstractmethod
    def is_applicable(self, query: str) -> bool:
        pass