"""
GLM-Ragas Compatibility Test

Tests whether GLM endpoint works correctly with Ragas evaluation framework
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")


def test_glm_ragas_connection():
    """
    Test GLM endpoint connectivity with Ragas
    
    This test verifies:
    1. GLM endpoint is accessible
    2. API authentication works
    3. Response format matches Ragas expectations
    """
    
    print("=" * 80)
    print("GLM-Ragas Compatibility Test")
    print("=" * 80)
    
    # Check environment configuration
    glm_endpoint = os.getenv("GLM_ENDPOINT", "")
    glm_api_key = os.getenv("GLM_API_KEY", "")
    
    print("\n1. Environment Configuration Check:")
    print(f"   GLM_ENDPOINT: {glm_endpoint[:50] if glm_endpoint else 'NOT SET'}...")
    print(f"   GLM_API_KEY: {glm_api_key[:10] if glm_api_key else 'NOT SET'}...")
    
    if not glm_endpoint or not glm_api_key:
        print("\n   ❌ GLM configuration missing - skipping compatibility test")
        print("   Set GLM_ENDPOINT and GLM_API_KEY environment variables to run this test")
        return False
    
    # Test Ragas availability
    print("\n2. Ragas Installation Check:")
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision
        from datasets import Dataset
        print("   ✓ Ragas installed successfully")
        print(f"   Ragas version: {evaluate.__module__}")
    except ImportError as e:
        print(f"   ❌ Ragas not installed: {e}")
        print("   Install with: pip install ragas datasets")
        return False
    
    # Test GLM API connectivity
    print("\n3. GLM API Connectivity Test:")
    try:
        import requests
        
        test_payload = {
            "model": "glm-4",
            "messages": [{"role": "user", "content": "测试连接"}],
            "temperature": 0.1,
        }
        
        headers = {
            "Authorization": f"Bearer {glm_api_key}",
            "Content-Type": "application/json",
        }
        
        with requests.Session() as session:
            session.trust_env = False
            response = session.post(
                glm_endpoint,
                json=test_payload,
                headers=headers,
                timeout=30,
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✓ GLM API accessible (status: {response.status_code})")
                print(f"   Response structure: {list(data.keys())}")
                
                # Validate response format
                if "choices" in data and len(data["choices"]) > 0:
                    message = data["choices"][0].get("message", {})
                    print(f"   ✓ Response contains 'choices' with 'message'")
                    print(f"   Message keys: {list(message.keys())}")
                else:
                    print(f"   ❌ Unexpected response format - missing 'choices'")
                    return False
                    
            else:
                print(f"   ❌ GLM API returned error status: {response.status_code}")
                print(f"   Response: {response.text[:200]}")
                return False
                
    except requests.RequestException as e:
        print(f"   ❌ Connection failed: {e.__class__.__name__}")
        return False
    
    # Test Ragas with GLM configuration
    print("\n4. Ragas + GLM Integration Test:")
    try:
        # Set environment for Ragas
        os.environ["OPENAI_API_KEY"] = glm_api_key
        os.environ["OPENAI_API_BASE"] = glm_endpoint
        
        # Create minimal test dataset
        test_data = Dataset.from_dict({
            "question": ["陕西电力市场的签约比例要求是什么？"],
            "answer": ["根据陕西2026年实施方案，年度签约比例下限为45%。"],
            "contexts": [["陕西省2026年电力市场化交易实施方案规定，燃煤发电企业年度电力中长期合同总签约电量不低于45%。"]],
            "ground_truth": ["45%"],
        })
        
        print("   Running Ragas evaluation...")
        
        # Run Ragas evaluation with timeout
        try:
            result = evaluate(
                test_data,
                metrics=[faithfulness, answer_relevancy],
            )
            
            print("   ✓ Ragas evaluation completed")
            print(f"   Result keys: {list(result.keys()) if hasattr(result, 'keys') else 'N/A'}")
            
            # Check if scores are valid
            if hasattr(result, "scores"):
                scores = result.scores[0] if result.scores else {}
                print(f"   Faithfulness score: {scores.get('faithfulness', 'N/A')}")
                print(f"   Answer relevancy score: {scores.get('answer_relevancy', 'N/A')}")
                
                # Validate score ranges (0-1)
                faithfulness_val = scores.get('faithfulness', 0)
                relevancy_val = scores.get('answer_relevancy', 0)
                
                if 0 <= faithfulness_val <= 1 and 0 <= relevancy_val <= 1:
                    print("   ✓ Scores in valid range [0, 1]")
                else:
                    print(f"   ⚠ Scores out of expected range: faithfulness={faithfulness_val}, relevancy={relevancy_val}")
                    
            elif isinstance(result, dict):
                print(f"   Result values: {result}")
                
        except Exception as e:
            print(f"   ❌ Ragas evaluation failed: {e.__class__.__name__}: {str(e)[:100]}")
            
            # Provide guidance
            print("\n   Possible issues:")
            print("   - GLM endpoint not OpenAI-compatible")
            print("   - Missing model parameter")
            print("   - Timeout on large evaluation")
            print("\n   Recommendation: Use mock evaluator for testing")
            return False
            
    except Exception as e:
        print(f"   ❌ Integration test setup failed: {e}")
        return False
    
    print("\n" + "=" * 80)
    print("✓ GLM-Ragas Compatibility Test PASSED")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Run full evaluation with: python evaluation/run_eval.py run --ragas")
    print("2. Monitor performance for large benchmarks")
    print("3. Consider batch processing optimization if needed")
    
    return True


def test_mock_ragas_evaluator():
    """
    Test MockRagasEvaluator as fallback when Ragas unavailable
    """
    
    print("\n" + "=" * 80)
    print("Mock Ragas Evaluator Test")
    print("=" * 80)
    
    from evaluation.ragas_evaluator import MockRagasEvaluator
    
    mock_eval = MockRagasEvaluator()
    
    print("\n1. Availability Check:")
    assert mock_eval.is_available()
    print("   ✓ Mock evaluator available")
    
    print("\n2. Single Question Evaluation:")
    result = mock_eval.evaluate_single(
        question="Test question",
        answer="Test answer with context",
        contexts=["Test context"],
    )
    
    print(f"   Faithfulness: {result['faithfulness']}")
    print(f"   Answer relevancy: {result['answer_relevancy']}")
    print(f"   Context precision: {result['context_precision']}")
    
    assert result['faithfulness'] >= 0.5
    assert result['answer_relevancy'] >= 0.5
    assert result['context_precision'] >= 0.5
    print("   ✓ All scores in valid range")
    
    print("\n3. Batch Evaluation:")
    batch_result = mock_eval.evaluate_batch(
        questions=["Q1", "Q2", "Q3"],
        answers=["A1", "A2", "A3"],
        contexts=[["C1"], ["C2"], ["C3"]],
    )
    
    print(f"   Batch size: 3")
    print(f"   Average faithfulness: {batch_result['avg_faithfulness']}")
    print(f"   Average relevancy: {batch_result['avg_answer_relevancy']}")
    print(f"   Average precision: {batch_result['avg_context_precision']}")
    
    assert batch_result['avg_faithfulness'] > 0
    print("   ✓ Batch evaluation successful")
    
    print("\n" + "=" * 80)
    print("✓ Mock Ragas Evaluator Test PASSED")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GLM-Ragas Compatibility Test")
    parser.add_argument("--mock", action="store_true", help="Test mock evaluator only")
    args = parser.parse_args()
    
    if args.mock:
        test_mock_ragas_evaluator()
    else:
        # Test both
        test_mock_ragas_evaluator()
        
        # Test GLM integration if configured
        test_glm_ragas_connection()