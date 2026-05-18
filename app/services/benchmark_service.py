from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class BenchmarkService:
    def __init__(self, benchmark_path: Optional[str] = None):
        if benchmark_path:
            self.benchmark_path = Path(benchmark_path)
        else:
            self.benchmark_path = Path(__file__).parent.parent.parent / "evaluation" / "benchmark.json"
        
        self._questions_cache: list[dict] | None = None
    
    def _load_questions(self) -> list[dict]:
        if self._questions_cache is not None:
            return self._questions_cache
        
        try:
            with open(self.benchmark_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            self._questions_cache = data.get("questions", [])
            logger.info(f"Loaded {len(self._questions_cache)} questions from benchmark.json")
            return self._questions_cache
        except Exception as e:
            logger.exception(f"Failed to load benchmark.json: {e}")
            return []
    
    def get_random_questions(self, count: int = 5, exclude_categories: Optional[list[str]] = None) -> list[dict]:
        if exclude_categories is None:
            exclude_categories = ["rejection"]
        
        questions = self._load_questions()
        
        filtered = [
            q for q in questions
            if q.get("category") not in exclude_categories
            and not q.get("should_reject", False)
        ]
        
        if len(filtered) <= count:
            return filtered
        
        return random.sample(filtered, count)
    
    def get_question_by_id(self, question_id: str) -> Optional[dict]:
        questions = self._load_questions()
        for q in questions:
            if q.get("question_id") == question_id:
                return q
        return None