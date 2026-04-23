from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
import logging

try:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision
    from ragas.llms import LangchainLLMWrapper
    from langchain_anthropic import ChatAnthropic
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    Dataset = None
    evaluate = None
    faithfulness = None
    answer_relevancy = None
    context_precision = None
    LangchainLLMWrapper = None
    ChatAnthropic = None

logger = logging.getLogger(__name__)


class RagasEvaluator:
    def __init__(
        self,
        llm_endpoint: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        llm_model: str = "MiniMax-M2.7",
    ):
        self.llm_endpoint = llm_endpoint or os.getenv("LLM_ENDPOINT", "")
        self.llm_api_key = llm_api_key or os.getenv("LLM_API_KEY", "")
        self.llm_model = llm_model
        self.ragas_llm = None
        self._setup_ragas()

    def _setup_ragas(self) -> None:
        if not RAGAS_AVAILABLE:
            return
        
        if self.llm_endpoint and self.llm_api_key:
            # Validate endpoint format
            if not self.llm_endpoint.startswith("http"):
                self.llm_endpoint = f"https://{self.llm_endpoint}"
            
            # Use LangchainLLMWrapper with ChatAnthropic for MiniMax
            try:
                langchain_llm = ChatAnthropic(
                    model=self.llm_model,
                    api_key=self.llm_api_key,
                    anthropic_api_url=self.llm_endpoint,
                    max_tokens=2048,
                    timeout=60,
                )
                self.ragas_llm = LangchainLLMWrapper(langchain_llm)
                logger.info(f"Ragas LLM setup complete: {self.llm_model} via {self.llm_endpoint}")
            except Exception as e:
                logger.warning(f"Failed to setup LangchainLLMWrapper: {e}. Falling back to OpenAI env vars.")
                os.environ["OPENAI_API_KEY"] = self.llm_api_key
                os.environ["OPENAI_API_BASE"] = self.llm_endpoint
                self.ragas_llm = None

    def evaluate_batch(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if not RAGAS_AVAILABLE:
            return {
                "faithfulness": {},
                "answer_relevancy": {},
                "context_precision": {},
                "error": "ragas not installed",
            }
        
        if not questions or not answers:
            return {
                "faithfulness": {},
                "answer_relevancy": {},
                "context_precision": {},
                "error": "empty input",
            }
        
        try:
            dataset = Dataset.from_dict({
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths or [""] * len(questions),
            })
            
            # Use custom LLM if available
            eval_kwargs = {}
            if self.ragas_llm is not None:
                eval_kwargs["llm"] = self.ragas_llm
            
            result = evaluate(
                dataset,
                metrics=[faithfulness, answer_relevancy, context_precision],
                **eval_kwargs,
            )
            
            faithfulness_scores = {}
            answer_relevancy_scores = {}
            context_precision_scores = {}
            
            if hasattr(result, "scores"):
                for i, score_dict in enumerate(result.scores):
                    faithfulness_scores[i] = score_dict.get("faithfulness", 0)
                    answer_relevancy_scores[i] = score_dict.get("answer_relevancy", 0)
                    context_precision_scores[i] = score_dict.get("context_precision", 0)
            else:
                avg_faithfulness = result.get("faithfulness", 0)
                avg_answer_relevancy = result.get("answer_relevancy", 0)
                avg_context_precision = result.get("context_precision", 0)
                
                for i in range(len(questions)):
                    faithfulness_scores[i] = avg_faithfulness
                    answer_relevancy_scores[i] = avg_answer_relevancy
                    context_precision_scores[i] = avg_context_precision
            
            return {
                "faithfulness": faithfulness_scores,
                "answer_relevancy": answer_relevancy_scores,
                "context_precision": context_precision_scores,
                "avg_faithfulness": sum(faithfulness_scores.values()) / len(faithfulness_scores) if faithfulness_scores else 0,
                "avg_answer_relevancy": sum(answer_relevancy_scores.values()) / len(answer_relevancy_scores) if answer_relevancy_scores else 0,
                "avg_context_precision": sum(context_precision_scores.values()) / len(context_precision_scores) if context_precision_scores else 0,
            }
        
        except Exception as e:
            logger.error(f"Ragas evaluation failed: {e}")
            return {
                "faithfulness": {},
                "answer_relevancy": {},
                "context_precision": {},
                "error": str(e),
            }

    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> Dict[str, float]:
        result = self.evaluate_batch(
            questions=[question],
            answers=[answer],
            contexts=[contexts],
            ground_truths=[ground_truth] if ground_truth else None,
        )
        
        return {
            "faithfulness": result.get("faithfulness", {}).get(0, 0),
            "answer_relevancy": result.get("answer_relevancy", {}).get(0, 0),
            "context_precision": result.get("context_precision", {}).get(0, 0),
        }

    def get_average_scores(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
    ) -> Dict[str, float]:
        result = self.evaluate_batch(questions, answers, contexts)
        
        return {
            "faithfulness": result.get("avg_faithfulness", 0),
            "answer_relevancy": result.get("avg_answer_relevancy", 0),
            "context_precision": result.get("avg_context_precision", 0),
        }

    def is_available(self) -> bool:
        return RAGAS_AVAILABLE and self.llm_api_key is not None


class MockRagasEvaluator:
    def __init__(self):
        pass

    def evaluate_batch(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        scores = {}
        for i, (question, answer, context) in enumerate(zip(questions, answers, contexts)):
            has_context = len(context) > 0 and any(c.strip() for c in context)
            has_answer = len(answer) > 0
            
            faithfulness = 0.85 if has_context and has_answer else 0.5
            answer_relevancy = 0.85 if has_answer else 0.5
            context_precision = 0.80 if has_context else 0.5
            
            scores[i] = {
                "faithfulness": faithfulness,
                "answer_relevancy": answer_relevancy,
                "context_precision": context_precision,
            }
        
        return {
            "faithfulness": {i: s["faithfulness"] for i, s in scores.items()},
            "answer_relevancy": {i: s["answer_relevancy"] for i, s in scores.items()},
            "context_precision": {i: s["context_precision"] for i, s in scores.items()},
            "avg_faithfulness": sum(s["faithfulness"] for s in scores.values()) / len(scores) if scores else 0,
            "avg_answer_relevancy": sum(s["answer_relevancy"] for s in scores.values()) / len(scores) if scores else 0,
            "avg_context_precision": sum(s["context_precision"] for s in scores.values()) / len(scores) if scores else 0,
        }

    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> Dict[str, float]:
        result = self.evaluate_batch(
            questions=[question],
            answers=[answer],
            contexts=[contexts],
            ground_truths=[ground_truth] if ground_truth else None,
        )
        
        return {
            "faithfulness": result.get("faithfulness", {}).get(0, 0),
            "answer_relevancy": result.get("answer_relevancy", {}).get(0, 0),
            "context_precision": result.get("context_precision", {}).get(0, 0),
        }

    def get_average_scores(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
    ) -> Dict[str, float]:
        result = self.evaluate_batch(questions, answers, contexts)
        
        return {
            "faithfulness": result.get("avg_faithfulness", 0),
            "answer_relevancy": result.get("avg_answer_relevancy", 0),
            "context_precision": result.get("avg_context_precision", 0),
        }

    def is_available(self) -> bool:
        return True


def get_ragas_evaluator(
    use_mock: bool = False,
    llm_endpoint: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    llm_model: str = "MiniMax-M2.7",
) -> Any:
    if use_mock or not RAGAS_AVAILABLE:
        return MockRagasEvaluator()
    
    return RagasEvaluator(
        llm_endpoint=llm_endpoint,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
    )