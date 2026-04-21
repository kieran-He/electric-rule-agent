"""
Ragas Configuration Manager

Provides configuration management for Ragas LLM evaluation with GLM support
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RagasConfig:
    """Configuration for Ragas evaluation"""
    
    enabled: bool = False
    llm_endpoint: str = ""
    llm_api_key: str = ""
    llm_model: str = "glm-4"
    batch_size: int = 10
    timeout_seconds: int = 60
    use_mock: bool = False
    max_retries: int = 2
    
    # Performance tuning
    enable_progress_monitor: bool = True
    cache_results: bool = True
    cache_dir: str = "evaluation/.ragas_cache"
    
    # Metrics selection
    metrics: List[str] = field(default_factory=lambda: ["faithfulness", "answer_relevancy", "context_precision"])
    
    def validate(self) -> bool:
        """Validate configuration"""
        
        if not self.enabled:
            return True
        
        if self.use_mock:
            return True
        
        if not self.llm_endpoint:
            logger.error("Ragas enabled but llm_endpoint not configured")
            return False
        
        if not self.llm_api_key:
            logger.error("Ragas enabled but llm_api_key not configured")
            return False
        
        if self.batch_size < 1 or self.batch_size > 100:
            logger.warning(f"batch_size {self.batch_size} out of range [1, 100], using default 10")
            self.batch_size = 10
        
        return True
    
    @classmethod
    def from_env(cls) -> RagasConfig:
        """Load configuration from environment variables"""
        
        enabled = os.getenv("RAGAS_ENABLED", "false").lower() == "true"
        use_mock = os.getenv("RAGAS_USE_MOCK", "false").lower() == "true"
        
        # Use GLM config as default
        llm_endpoint = os.getenv("RAGAS_ENDPOINT", os.getenv("GLM_ENDPOINT", ""))
        llm_api_key = os.getenv("RAGAS_API_KEY", os.getenv("GLM_API_KEY", ""))
        llm_model = os.getenv("RAGAS_MODEL", os.getenv("GLM_MODEL", "glm-4"))
        
        batch_size = int(os.getenv("RAGAS_BATCH_SIZE", "10"))
        timeout_seconds = int(os.getenv("RAGAS_TIMEOUT", "60"))
        
        return cls(
            enabled=enabled,
            llm_endpoint=llm_endpoint,
            llm_api_key=llm_api_key,
            llm_model=llm_model,
            batch_size=batch_size,
            timeout_seconds=timeout_seconds,
            use_mock=use_mock,
        )
    
    @classmethod
    def from_file(cls, config_path: str) -> RagasConfig:
        """Load configuration from JSON file"""
        
        import json
        from pathlib import Path
        
        path = Path(config_path)
        if not path.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return cls()
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return cls(
            enabled=data.get("enabled", False),
            llm_endpoint=data.get("llm_endpoint", ""),
            llm_api_key=data.get("llm_api_key", ""),
            llm_model=data.get("llm_model", "glm-4"),
            batch_size=data.get("batch_size", 10),
            timeout_seconds=data.get("timeout_seconds", 60),
            use_mock=data.get("use_mock", False),
        )
    
    def to_env(self) -> None:
        """Export configuration to environment variables"""
        
        os.environ["RAGAS_ENABLED"] = str(self.enabled).lower()
        os.environ["RAGAS_USE_MOCK"] = str(self.use_mock).lower()
        os.environ["RAGAS_ENDPOINT"] = self.llm_endpoint
        os.environ["RAGAS_API_KEY"] = self.llm_api_key
        os.environ["RAGAS_MODEL"] = self.llm_model
        os.environ["RAGAS_BATCH_SIZE"] = str(self.batch_size)
        os.environ["RAGAS_TIMEOUT"] = str(self.timeout_seconds)
        
        # Also set OpenAI-compatible variables for Ragas
        if self.enabled and not self.use_mock:
            os.environ["OPENAI_API_KEY"] = self.llm_api_key
            os.environ["OPENAI_API_BASE"] = self.llm_endpoint


class RagasBatchProcessor:
    """
    Optimized batch processor for Ragas evaluation
    
    Features:
    - Batch processing with configurable size
    - Progress monitoring
    - Result caching
    - Error recovery
    """
    
    def __init__(self, config: RagasConfig):
        self.config = config
        self._cache: Dict[str, Any] = {}
        
        if config.cache_results:
            self._init_cache_dir()
    
    def _init_cache_dir(self) -> None:
        """Initialize cache directory"""
        from pathlib import Path
        
        cache_path = Path(self.config.cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
    
    def process_in_batches(
        self,
        questions: list[str],
        answers: list[str],
        contexts: list[list[str]],
        evaluator: Any,
    ) -> Dict[str, Any]:
        """
        Process evaluation in optimized batches
        
        Returns aggregated results with per-item scores and averages
        """
        
        total_items = len(questions)
        batch_size = self.config.batch_size
        
        if total_items <= batch_size:
            # Process all at once
            return evaluator.evaluate_batch(questions, answers, contexts)
        
        # Process in batches
        logger.info(f"Processing {total_items} items in batches of {batch_size}")
        
        all_results = {
            "faithfulness": {},
            "answer_relevancy": {},
            "context_precision": {},
        }
        
        total_batches = (total_items + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_items)
            
            batch_questions = questions[start_idx:end_idx]
            batch_answers = answers[start_idx:end_idx]
            batch_contexts = contexts[start_idx:end_idx]
            
            if self.config.enable_progress_monitor:
                logger.info(
                    f"Processing batch {batch_idx + 1}/{total_batches} "
                    f"(items {start_idx + 1}-{end_idx})"
                )
            
            try:
                batch_result = evaluator.evaluate_batch(
                    batch_questions,
                    batch_answers,
                    batch_contexts,
                )
                
                # Merge results
                for i in range(len(batch_questions)):
                    global_idx = start_idx + i
                    
                    all_results["faithfulness"][global_idx] = batch_result.get("faithfulness", {}).get(i, 0)
                    all_results["answer_relevancy"][global_idx] = batch_result.get("answer_relevancy", {}).get(i, 0)
                    all_results["context_precision"][global_idx] = batch_result.get("context_precision", {}).get(i, 0)
                    
            except Exception as e:
                logger.error(f"Batch {batch_idx + 1} failed: {e}")
                
                # Fill with zeros for failed batch
                for i in range(len(batch_questions)):
                    global_idx = start_idx + i
                    all_results["faithfulness"][global_idx] = 0
                    all_results["answer_relevancy"][global_idx] = 0
                    all_results["context_precision"][global_idx] = 0
        
        # Calculate averages
        faithfulness_values = list(all_results["faithfulness"].values())
        relevancy_values = list(all_results["answer_relevancy"].values())
        precision_values = list(all_results["context_precision"].values())
        
        all_results["avg_faithfulness"] = sum(faithfulness_values) / len(faithfulness_values) if faithfulness_values else 0
        all_results["avg_answer_relevancy"] = sum(relevancy_values) / len(relevancy_values) if relevancy_values else 0
        all_results["avg_context_precision"] = sum(precision_values) / len(precision_values) if precision_values else 0
        
        return all_results


def create_ragas_config_file(output_path: str = "evaluation/ragas_config.json") -> None:
    """
    Create default Ragas configuration file
    """
    
    import json
    from pathlib import Path
    
    default_config = {
        "enabled": False,
        "use_mock": True,
        "llm_endpoint": "",
        "llm_api_key": "",
        "llm_model": "glm-4",
        "batch_size": 10,
        "timeout_seconds": 60,
        "max_retries": 2,
        "enable_progress_monitor": True,
        "cache_results": True,
        "cache_dir": "evaluation/.ragas_cache",
        "metrics": ["faithfulness", "answer_relevancy", "context_precision"],
    }
    
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(default_config, f, indent=2)
    
    logger.info(f"Created default Ragas config: {output_path}")
    print(f"\nRagas Configuration File: {output_path}")
    print("\nTo enable Ragas evaluation:")
    print("1. Set enabled=true")
    print("2. Configure llm_endpoint and llm_api_key")
    print("3. Or use use_mock=true for testing")
    print("\nEnvironment variables (alternative):")
    print("  RAGAS_ENABLED=true")
    print("  RAGAS_ENDPOINT=https://your-glm-endpoint.com")
    print("  RAGAS_API_KEY=your-api-key")


if __name__ == "__main__":
    create_ragas_config_file()