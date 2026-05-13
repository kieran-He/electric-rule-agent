"""
Answer Verifier Service

Provides lightweight and full RAGAS verification for answer quality assessment.
"""
from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, List, Optional

from app.schemas.verification import VerificationResult

if TYPE_CHECKING:
    from app.config import Settings
    from app.langchain.llm import MiniMaxLLMWrapper
    from evaluation.ragas_evaluator import RagasEvaluator

logger = logging.getLogger(__name__)


VERIFICATION_PROMPT = """请评估以下答案质量（严格基于上下文判断）：

问题：{query}
上下文摘要：{context_summary}
答案：{answer}

评估维度（1-10分，JSON格式输出）：
1. accuracy: 答案是否准确基于上下文，无幻觉（关键指标）
2. completeness: 是否充分回答问题核心
3. relevance: 是否切题，无无关内容

输出：{"accuracy": X, "completeness": X, "relevance": X}
仅输出JSON，无其他内容。"""


class AnswerVerifier:
    """
    Answer verification with lightweight LLM check and optional RAGAS evaluation.
    
    Flow:
    1. Quick LLM verification (single call)
    2. If confidence < threshold, trigger full RAGAS evaluation
    3. Return verification result with optional warning
    """

    def __init__(
        self,
        llm_wrapper: "MiniMaxLLMWrapper",
        ragas_evaluator: Optional["RagasEvaluator"],
        settings: "Settings",
    ):
        self.llm_wrapper = llm_wrapper
        self.ragas_evaluator = ragas_evaluator
        self.settings = settings

    def verify(
        self,
        query: str,
        answer: str,
        contexts: List[str],
    ) -> VerificationResult:
        """
        Verify answer quality with quick check and optional RAGAS evaluation.
        
        Args:
            query: User question
            answer: Generated answer
            contexts: Retrieved context chunks
            
        Returns:
            VerificationResult with scores and optional warning
        """
        if not contexts:
            return VerificationResult(
                confidence=0.3,
                needs_retry=True,
                warning="无上下文支持，答案可信度较低",
                verification_type="none",
            )

        try:
            quick_result = self._quick_verify(query, answer, contexts)
            
            if quick_result.confidence >= self.settings.verification_quick_threshold:
                return VerificationResult(
                    faithfulness=quick_result.faithfulness,
                    answer_relevancy=quick_result.answer_relevancy,
                    context_precision=quick_result.context_precision,
                    confidence=quick_result.confidence,
                    needs_retry=quick_result.confidence < self.settings.verification_retry_threshold,
                    warning=None,
                    verification_type="quick",
                )

            if self.ragas_evaluator and self.ragas_evaluator.is_available():
                full_scores = self._ragas_verify(query, answer, contexts)
                warning = self._build_warning(full_scores)
                
                return VerificationResult(
                    faithfulness=full_scores.get("faithfulness", 0),
                    answer_relevancy=full_scores.get("answer_relevancy", 0),
                    context_precision=full_scores.get("context_precision", 0),
                    confidence=full_scores.get("faithfulness", 0),
                    needs_retry=full_scores.get("faithfulness", 0) < self.settings.verification_retry_threshold,
                    warning=warning,
                    verification_type="ragas",
                )

            return VerificationResult(
                faithfulness=quick_result.faithfulness,
                answer_relevancy=quick_result.answer_relevancy,
                context_precision=quick_result.context_precision,
                confidence=quick_result.confidence,
                needs_retry=quick_result.confidence < self.settings.verification_retry_threshold,
                warning="答案可信度较低，建议核实" if quick_result.confidence < self.settings.verification_warning_threshold else None,
                verification_type="quick",
            )

        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return VerificationResult(
                confidence=0.5,
                needs_retry=False,
                warning=None,
                verification_type="error",
            )

    def _quick_verify(
        self,
        query: str,
        answer: str,
        contexts: List[str],
    ) -> VerificationResult:
        """
        Quick LLM-based verification with single call.
        
        Returns scores based on accuracy, completeness, and relevance.
        """
        context_summary = "\n".join(contexts[:3])[:1500]
        
        prompt = VERIFICATION_PROMPT.format(
            query=query,
            context_summary=context_summary,
            answer=answer[:2000],
        )

        try:
            response, _, _ = self.llm_wrapper.invoke(prompt, system="你是一个严格的答案质量评估专家。请客观评估，仅输出JSON格式结果。")
            
            scores = self._parse_verification_response(response)
            
            accuracy = scores.get("accuracy", 5) / 10.0
            completeness = scores.get("completeness", 5) / 10.0
            relevance = scores.get("relevance", 5) / 10.0
            
            confidence = accuracy * 0.5 + completeness * 0.25 + relevance * 0.25
            
            return VerificationResult(
                faithfulness=accuracy,
                answer_relevancy=relevance,
                context_precision=completeness,
                confidence=confidence,
                verification_type="quick",
            )

        except Exception as e:
            logger.warning(f"Quick verification failed: {e}")
            return VerificationResult(
                confidence=0.5,
                verification_type="quick_error",
            )

    def _parse_verification_response(self, response: str) -> dict:
        """Parse JSON scores from LLM response."""
        scores = {"accuracy": 5, "completeness": 5, "relevance": 5}
        
        try:
            json_patterns = [
                r'\{[^{}]*\}',
                r'\{(?:[^{}]|\{[^{}]*\})*\}',
            ]
            
            for pattern in json_patterns:
                matches = re.findall(pattern, response, re.DOTALL)
                for match in matches:
                    try:
                        parsed = json.loads(match)
                        if "scores" in parsed and isinstance(parsed["scores"], dict):
                            parsed = parsed["scores"]
                        if "evaluation" in parsed and isinstance(parsed["evaluation"], dict):
                            parsed = parsed["evaluation"]
                        
                        for key in ["accuracy", "completeness", "relevance"]:
                            if key in parsed:
                                val = parsed[key]
                                if isinstance(val, (int, float)):
                                    scores[key] = int(val)
                                elif isinstance(val, str):
                                    try:
                                        scores[key] = int(float(val))
                                    except ValueError:
                                        pass
                        return scores
                    except json.JSONDecodeError:
                        continue
        
        except Exception as e:
            logger.debug(f"JSON parse failed: {e}, falling back to regex")
        
        accuracy_match = re.search(r'"accuracy"\s*:\s*["\']?(\d+)["\']?', response)
        if accuracy_match:
            scores["accuracy"] = int(accuracy_match.group(1))
        
        completeness_match = re.search(r'"completeness"\s*:\s*["\']?(\d+)["\']?', response)
        if completeness_match:
            scores["completeness"] = int(completeness_match.group(1))
        
        relevance_match = re.search(r'"relevance"\s*:\s*["\']?(\d+)["\']?', response)
        if relevance_match:
            scores["relevance"] = int(relevance_match.group(1))
        
        return scores

    def _ragas_verify(
        self,
        query: str,
        answer: str,
        contexts: List[str],
    ) -> dict:
        """Full RAGAS evaluation for low-confidence answers."""
        try:
            return self.ragas_evaluator.evaluate_single(
                question=query,
                answer=answer,
                contexts=contexts,
            )
        except Exception as e:
            logger.error(f"RAGAS verification failed: {e}")
            return {"faithfulness": 0.5, "answer_relevancy": 0.5, "context_precision": 0.5}

    def _build_warning(self, scores: dict) -> Optional[str]:
        """Build warning message based on RAGAS scores."""
        faithfulness = scores.get("faithfulness", 1.0)
        answer_relevancy = scores.get("answer_relevancy", 1.0)
        
        warnings = []
        
        if faithfulness < self.settings.verification_warning_threshold:
            warnings.append("答案与参考内容一致性较低")
        
        if answer_relevancy < self.settings.verification_warning_threshold:
            warnings.append("答案与问题相关性较低")
        
        if warnings:
            return "，".join(warnings) + "，建议核实信息准确性"
        
        return None