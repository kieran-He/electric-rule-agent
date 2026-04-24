"""Test query rewrite functionality."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.langchain.query_rewriter import QueryRewriter
from app.langchain.llm import MiniMaxLLMWrapper


def test_trigger_conditions():
    """Test should_rewrite trigger conditions."""
    print("\n=== Testing Trigger Conditions ===\n")
    
    rewriter = QueryRewriter(enabled=True, min_length=10)
    
    test_cases = [
        ("交易规则", "short query"),
        ("陕西政策", "missing domain keyword"),
        ("怎么结算", "colloquial expression"),
        ("电力市场交易规则", "should not trigger"),
        ("中长期交易怎么弄", "colloquial + domain keyword"),
    ]
    
    for query, description in test_cases:
        should, reason = rewriter.should_rewrite(query)
        status = "TRIGGER" if should else "SKIP"
        print(f"  [{status}] '{query}' ({description})")
        if should:
            print(f"         Reason: {reason}")


def test_rewrite_with_llm():
    """Test rewrite with actual LLM call."""
    print("\n=== Testing LLM Rewrite ===\n")
    
    try:
        llm = MiniMaxLLMWrapper()
    except Exception as e:
        print(f"  [SKIP] LLM not available: {e}")
        return
    
    rewriter = QueryRewriter(llm_wrapper=llm, enabled=True)
    
    test_queries = [
        "交易规则是什么",
        "结算怎么弄",
        "陕西政策",
    ]
    
    for query in test_queries:
        print(f"  Input: '{query}'")
        result = rewriter.rewrite(query)
        print(f"  Output: '{result.rewritten_query}'")
        print(f"  Triggered: {result.triggered}, Reason: {result.trigger_reason}")
        print()


def test_rewrite_disabled():
    """Test that disabled rewriter returns original query."""
    print("\n=== Testing Disabled Rewrite ===\n")
    
    rewriter = QueryRewriter(enabled=False)
    
    result = rewriter.rewrite("交易规则是什么")
    
    assert result.triggered is False
    assert result.rewritten_query == "交易规则是什么"
    print(f"  [PASS] Disabled rewriter returns original query")


if __name__ == "__main__":
    test_trigger_conditions()
    test_rewrite_disabled()
    test_rewrite_with_llm()
    print("\n=== All Tests Complete ===\n")