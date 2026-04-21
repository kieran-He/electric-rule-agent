from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


@dataclass
class Document:
    doc_name: str
    doc_type: str
    province_code: str
    content: str
    clauses: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class GeneratedQuestion:
    question_id: str
    question: str
    category: str
    expected_docs: List[str]
    expected_articles: List[str]
    expected_answer_keywords: List[str]
    should_reject: bool
    expected_intent: str
    length_type: str
    scope_type: str


QUESTION_TEMPLATES = {
    "clause_qa": [
        "{主体}{年份}{规则类型}要求是什么？",
        "{市场类型}的{具体规则}如何执行？",
        "{主体}参与{市场类型}需要满足哪些条件？",
        "{规则类型}的具体规定有哪些？",
        "{时间维度}{市场类型}的{规则类型}是如何规定的？",
        "{主体}在{市场类型}中的{规则类型}要求",
        "{省份}{年份}{规则类型}的{具体条款}内容",
        "{规则类型}对{主体}有什么影响？",
    ],
    "flow_qa": [
        "{主体}参与{市场类型}的完整流程是什么？",
        "{主体}注册{市场类型}需要哪些步骤？",
        "{主体}准入{市场类型}的流程是怎样的？",
        "{省份}{主体}参与市场的详细流程",
        "从注册到交易的完整流程是什么？",
        "{主体}办理{市场类型}手续的流程",
    ],
    "compare_qa": [
        "{省份1}和{省份2}{规则类型}有什么区别？",
        "对比{主体1}和{主体2}的{规则类型}要求",
        "{市场类型1}和{市场类型2}的{规则类型}对比",
        "{省份}不同年份{规则类型}的变化",
        "{主体}在不同市场中的准入条件对比",
    ],
    "settlement_qa": [
        "{市场类型}的结算周期是多少？",
        "{主体}的结算方式是什么？",
        "{市场类型}的计量要求有哪些？",
        "日清月结的具体流程是什么？",
        "{主体}结算电费的计算方式",
    ],
    "rejection": [
        "请介绍一下中国股市的最新政策",
        "美国电力市场规则是什么？",
        "如何办理营业执照？",
        "新能源汽车补贴政策",
        "股票投资的税收政策",
    ],
}

SUBJECTS = ["发电企业", "售电公司", "电力用户", "虚拟电厂", "独立储能", "新能源企业", "火电企业", "风电企业"]
MARKETS = ["中长期", "现货", "零售", "辅助服务", "结算", "计量"]
YEARS = ["2024", "2025", "2026"]
PROVINCES = ["陕西", "山西", "山东", "广东", "江苏"]
RULE_TYPES = ["签约比例", "准入条件", "价格机制", "结算规则", "计量要求", "交易流程", "报价规则"]
TIME_DIMENSIONS = ["年度", "月度", "日前", "实时", "日内"]


class BenchmarkGenerator:
    def __init__(
        self,
        llm_endpoint: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        llm_model: str = "glm-4",
    ):
        self.llm_endpoint = llm_endpoint or os.getenv("GLM_ENDPOINT", "")
        self.llm_api_key = llm_api_key or os.getenv("GLM_API_KEY", "")
        self.llm_model = llm_model

    def _call_llm(self, prompt: str) -> str:
        if not self.llm_endpoint or not self.llm_api_key:
            return ""
        
        payload = {
            "model": self.llm_model,
            "messages": [
                {"role": "system", "content": "你是电力市场规则问答问题生成助手。根据给定的上下文生成多样化的问题。"},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.7,
        }
        
        headers = {
            "Authorization": f"Bearer {self.llm_api_key}",
            "Content-Type": "application/json",
        }
        
        try:
            with requests.Session() as session:
                session.trust_env = False
                response = session.post(
                    self.llm_endpoint,
                    json=payload,
                    headers=headers,
                    timeout=30,
                )
                response.raise_for_status()
                data = response.json()
                return data.get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception:
            return ""

    def generate_from_docs(
        self,
        docs_path: str,
        output_path: str,
        total_count: int = 100,
    ) -> None:
        docs = self._scan_documents(docs_path)
        questions = []
        
        clause_questions = self._generate_clause_questions(docs, count=40)
        questions.extend(clause_questions)
        
        flow_questions = self._generate_flow_questions(docs, count=20)
        questions.extend(flow_questions)
        
        compare_questions = self._generate_compare_questions(docs, count=15)
        questions.extend(compare_questions)
        
        settlement_questions = self._generate_settlement_questions(docs, count=15)
        questions.extend(settlement_questions)
        
        rejection_questions = self._generate_rejection_questions(count=10)
        questions.extend(rejection_questions)
        
        random.shuffle(questions)
        questions = questions[:total_count]
        
        for i, q in enumerate(questions):
            q.question_id = f"q{i+1:03d}"
        
        self._save_benchmark(questions, output_path)

    def _scan_documents(self, docs_path: str) -> List[Document]:
        docs = []
        path = Path(docs_path)
        if not path.exists():
            return docs
        
        for file_path in path.glob("**/*.pdf"):
            doc = Document(
                doc_name=file_path.stem,
                doc_type="pdf",
                province_code=self._extract_province(file_path.stem),
                content="",
            )
            docs.append(doc)
        
        for file_path in path.glob("**/*.docx"):
            doc = Document(
                doc_name=file_path.stem,
                doc_type="docx",
                province_code=self._extract_province(file_path.stem),
                content="",
            )
            docs.append(doc)
        
        return docs

    def _extract_province(self, filename: str) -> str:
        for province in PROVINCES:
            if province in filename:
                return province
        return "陕西"

    def _generate_clause_questions(
        self,
        docs: List[Document],
        count: int,
    ) -> List[GeneratedQuestion]:
        questions = []
        
        for _ in range(count):
            template = random.choice(QUESTION_TEMPLATES["clause_qa"])
            subject = random.choice(SUBJECTS)
            market = random.choice(MARKETS)
            year = random.choice(YEARS)
            province = random.choice(PROVINCES)
            rule_type = random.choice(RULE_TYPES)
            time_dim = random.choice(TIME_DIMENSIONS)
            
            question_text = template.format(
                主体=subject,
                年份=year,
                市场类型=market,
                规则类型=rule_type,
                省份=province,
                时间维度=time_dim,
                具体规则=rule_type,
                具体条款=random.choice(["准入条件", "签约比例", "价格上限"]),
            )
            
            length_type = self._classify_length(question_text)
            scope_type = random.choice(["微观条款", "宏观总结"])
            
            q = GeneratedQuestion(
                question_id="",
                question=question_text,
                category="clause_qa",
                expected_docs=[f"{province}{year}电力市场化交易实施方案"],
                expected_articles=["二、总体要求"],
                expected_answer_keywords=[subject, market, rule_type],
                should_reject=False,
                expected_intent="clause_qa",
                length_type=length_type,
                scope_type=scope_type,
            )
            questions.append(q)
        
        return questions

    def _generate_flow_questions(
        self,
        docs: List[Document],
        count: int,
    ) -> List[GeneratedQuestion]:
        questions = []
        
        flow_subjects = ["储能", "虚拟电厂", "售电公司", "新能源企业"]
        flow_markets = ["现货市场", "中长期市场", "辅助服务市场"]
        
        for _ in range(count):
            template = random.choice(QUESTION_TEMPLATES["flow_qa"])
            subject = random.choice(flow_subjects)
            market = random.choice(flow_markets)
            province = random.choice(PROVINCES)
            
            question_text = template.format(
                主体=subject,
                市场类型=market,
                省份=province,
            )
            
            q = GeneratedQuestion(
                question_id="",
                question=question_text,
                category="flow_qa",
                expected_docs=[f"{province}电力市场交易规则"],
                expected_articles=["三、准入流程"],
                expected_answer_keywords=["注册", "准入", "审核", "签约"],
                should_reject=False,
                expected_intent="flow_qa",
                length_type=self._classify_length(question_text),
                scope_type="流程说明",
            )
            questions.append(q)
        
        return questions

    def _generate_compare_questions(
        self,
        docs: List[Document],
        count: int,
    ) -> List[GeneratedQuestion]:
        questions = []
        
        for _ in range(count):
            template = random.choice(QUESTION_TEMPLATES["compare_qa"])
            province1 = random.choice(PROVINCES)
            province2 = random.choice([p for p in PROVINCES if p != province1])
            rule_type = random.choice(RULE_TYPES)
            subject1 = random.choice(SUBJECTS)
            subject2 = random.choice([s for s in SUBJECTS if s != subject1])
            market1 = random.choice(MARKETS)
            market2 = random.choice([m for m in MARKETS if m != market1])
            
            question_text = template.format(
                省份1=province1,
                省份2=province2,
                主体1=subject1,
                主体2=subject2,
                市场类型1=market1,
                市场类型2=market2,
                规则类型=rule_type,
                主体=random.choice(SUBJECTS),
                省份=random.choice(PROVINCES),
            )
            
            q = GeneratedQuestion(
                question_id="",
                question=question_text,
                category="compare_qa",
                expected_docs=[f"{province1}电力市场交易规则", f"{province2}电力市场交易规则"],
                expected_articles=["准入条件", "价格机制"],
                expected_answer_keywords=["区别", "对比", "差异"],
                should_reject=False,
                expected_intent="compare_qa",
                length_type=self._classify_length(question_text),
                scope_type="跨省对比",
            )
            questions.append(q)
        
        return questions

    def _generate_settlement_questions(
        self,
        docs: List[Document],
        count: int,
    ) -> List[GeneratedQuestion]:
        questions = []
        
        for _ in range(count):
            template = random.choice(QUESTION_TEMPLATES["settlement_qa"])
            market = random.choice(MARKETS)
            subject = random.choice(SUBJECTS)
            
            question_text = template.format(
                市场类型=market,
                主体=subject,
            )
            
            q = GeneratedQuestion(
                question_id="",
                question=question_text,
                category="settlement_qa",
                expected_docs=["电力市场结算规则"],
                expected_articles=["结算周期", "计量要求"],
                expected_answer_keywords=["日清", "月结", "结算", "计量"],
                should_reject=False,
                expected_intent="settlement_qa",
                length_type=self._classify_length(question_text),
                scope_type="结算说明",
            )
            questions.append(q)
        
        return questions

    def _generate_rejection_questions(
        self,
        count: int,
    ) -> List[GeneratedQuestion]:
        questions = []
        
        for i in range(count):
            template = random.choice(QUESTION_TEMPLATES["rejection"])
            
            q = GeneratedQuestion(
                question_id="",
                question=template,
                category="rejection",
                expected_docs=[],
                expected_articles=[],
                expected_answer_keywords=["未检索到", "无法回答"],
                should_reject=True,
                expected_intent="rejection",
                length_type=self._classify_length(template),
                scope_type="知识库外",
            )
            questions.append(q)
        
        return questions

    def _classify_length(self, question: str) -> str:
        length = len(question)
        if length <= 15:
            return "短问题"
        elif length <= 30:
            return "中等问题"
        else:
            return "长问题"

    def _save_benchmark(
        self,
        questions: List[GeneratedQuestion],
        output_path: str,
    ) -> None:
        data = {
            "version": "v1.0",
            "generated_at": str(Path(output_path).stat().st_mtime if Path(output_path).exists() else 0),
            "total_count": len(questions),
            "distribution": {
                "clause_qa": len([q for q in questions if q.category == "clause_qa"]),
                "flow_qa": len([q for q in questions if q.category == "flow_qa"]),
                "compare_qa": len([q for q in questions if q.category == "compare_qa"]),
                "settlement_qa": len([q for q in questions if q.category == "settlement_qa"]),
                "rejection": len([q for q in questions if q.category == "rejection"]),
            },
            "questions": [
                {
                    "question_id": q.question_id,
                    "question": q.question,
                    "category": q.category,
                    "expected_docs": q.expected_docs,
                    "expected_articles": q.expected_articles,
                    "expected_answer_keywords": q.expected_answer_keywords,
                    "should_reject": q.should_reject,
                    "expected_intent": q.expected_intent,
                    "length_type": q.length_type,
                    "scope_type": q.scope_type,
                }
                for q in questions
            ],
        }
        
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def generate_single_category(
        self,
        category: str,
        docs: List[Document],
        count: int,
    ) -> List[GeneratedQuestion]:
        generators = {
            "clause_qa": self._generate_clause_questions,
            "flow_qa": self._generate_flow_questions,
            "compare_qa": self._generate_compare_questions,
            "settlement_qa": self._generate_settlement_questions,
            "rejection": self._generate_rejection_questions,
        }
        
        generator = generators.get(category)
        if generator:
            return generator(docs, count)
        return []

    def enhance_with_llm(
        self,
        questions: List[GeneratedQuestion],
        docs: List[Document],
    ) -> List[GeneratedQuestion]:
        enhanced = []
        for q in questions:
            prompt = f"""
            原问题: {q.question}
            预期文档: {q.expected_docs}
            请根据电力市场规则上下文，优化这个问题使其更加具体和准确。
            同时提供3个关键词作为期望答案关键词。
            """
            
            llm_response = self._call_llm(prompt)
            if llm_response:
                q.question = llm_response.strip().split("\n")[0]
            
            enhanced.append(q)
        
        return enhanced