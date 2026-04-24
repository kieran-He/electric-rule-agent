"""Test Single Query Ragas Evaluation with Query Rewrite."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

import os
import warnings
warnings.filterwarnings('ignore')

os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("=" * 60)
print("Single Query Ragas Evaluation Test")
print("=" * 60)

from langchain_anthropic import ChatAnthropic
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import faithfulness, answer_relevancy
from ragas import evaluate
from datasets import Dataset

from app.config import settings
from app.repository import ChromaPolicyRepository
from app.langchain.bm25_indexer import BM25Indexer
from app.langchain.hybrid_retriever import HybridRetriever, BGEReranker
from app.langchain.query_rewriter import QueryRewriter
from app.langchain.llm import MiniMaxLLMWrapper

print("\n[Step 1] Initialize Ragas LLM...")
llm_api_key = os.getenv("LLM_API_KEY", "")
llm_endpoint = os.getenv("LLM_ENDPOINT", "")
llm_model = os.getenv("LLM_MODEL", "MiniMax-M2.7")

langchain_llm = ChatAnthropic(
    model=llm_model,
    api_key=llm_api_key,
    anthropic_api_url=llm_endpoint,
    max_tokens=2048,
    timeout=120,
)
ragas_llm = LangchainLLMWrapper(langchain_llm=langchain_llm)
print(f"Ragas LLM: {llm_endpoint} / {llm_model}")

print("\n[Step 2] Initialize Retrieval Chain...")
repo = ChromaPolicyRepository(
    persist_directory=settings.chroma_path,
    embedding_model_name=settings.embedding_model,
)

bm25 = BM25Indexer(k1=settings.bm25_k1, b=settings.bm25_b)
bm25.build_index()

llm_wrapper = MiniMaxLLMWrapper()
rewriter = QueryRewriter(
    llm_wrapper=llm_wrapper,
    enabled=settings.query_rewrite_enabled,
    min_length=settings.query_rewrite_min_length,
)

reranker = BGEReranker(
    model_name=settings.reranker_model,
    max_length=settings.reranker_max_length,
)

hybrid = HybridRetriever(
    vector_repo=repo,
    bm25_indexer=bm25,
    reranker=reranker,
    query_rewriter=rewriter,
    vector_top_k=settings.hybrid_vector_top_k,
    bm25_top_k=settings.hybrid_bm25_top_k,
    final_top_k=settings.hybrid_final_top_k,
    use_query_rewrite=settings.query_rewrite_enabled,
    query_rewrite_keep_original=settings.query_rewrite_keep_original,
)
print("Retrieval chain ready")

print("\n[Step 3] Test Query (口语化问题)...")
test_query = "交易规则是什么"
print(f"Query: '{test_query}'")

rewrite_result = rewriter.rewrite(test_query)
print(f"Rewritten: '{rewrite_result.rewritten_query}'")

print("\n[Step 4] Retrieve Contexts...")
chunks = hybrid.retrieve(test_query, ["SN"])
contexts = [c.text for c in chunks]
print(f"Retrieved: {len(contexts)} contexts")

print("\n[Step 5] Generate Answer...")
from app.langchain.orchestrator_hybrid import HybridQAOrchestrator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine = create_engine(settings.database_url)
Session = sessionmaker(bind=engine)
db = Session()

orchestrator = HybridQAOrchestrator(
    db=db,
    settings=settings,
    use_hybrid=True,
)

from app.schemas.query import QueryRequest
req = QueryRequest(
    query=test_query,
    session_id="test_session_001",
    province_codes=["SN"],
    top_k=8,
    need_citation=True,
)

answer_result = orchestrator.run(req)
answer = answer_result.answer
print(f"Answer: {answer[:200]}...")

print("\n[Step 6] Build Ragas Dataset...")
dataset = Dataset.from_dict({
    "question": [test_query],
    "answer": [answer],
    "contexts": [contexts],
})
print(f"Dataset: {len(dataset)} samples")

print("\n[Step 7] Run Ragas Evaluation...")
import time
start_time = time.time()

try:
    result = evaluate(
        dataset,
        metrics=[faithfulness],
        llm=ragas_llm,
    )
    
    elapsed = time.time() - start_time
    print(f"Evaluation completed in {elapsed:.1f}s")
    
    if hasattr(result, 'scores'):
        scores = result.scores[0]
        faithfulness_score = scores.get('faithfulness', 0)
        print(f"\nResults:")
        print(f"  faithfulness: {faithfulness_score:.3f}")
    
except Exception as e:
    print(f"[ERROR] Ragas evaluation failed: {e}")

print("\n" + "=" * 60)
print("Test Complete!")
print("=" * 60)