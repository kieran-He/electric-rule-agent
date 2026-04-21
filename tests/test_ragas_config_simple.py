"""
简化测试：验证Ragas配置和Batch Processor
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.ragas_config import RagasConfig, RagasBatchProcessor
from evaluation.ragas_evaluator import MockRagasEvaluator


def test_config_creation():
    print("\n1. Test RagasConfig creation")
    
    # Default config
    config1 = RagasConfig()
    print(f"   Default: enabled={config1.enabled}, batch_size={config1.batch_size}")
    
    # From environment - mock mode
    import os
    os.environ["RAGAS_ENABLED"] = "true"
    os.environ["RAGAS_USE_MOCK"] = "true"  # Use mock to bypass endpoint validation
    os.environ["RAGAS_BATCH_SIZE"] = "15"
    config2 = RagasConfig.from_env()
    print(f"   From env: enabled={config2.enabled}, use_mock={config2.use_mock}, batch_size={config2.batch_size}")
    
    # Validate - should pass with mock
    assert config2.validate()
    print("   Validation passed")
    
    # Test with endpoint
    os.environ["RAGAS_ENDPOINT"] = "https://test.com"
    os.environ["RAGAS_API_KEY"] = "test-key"
    os.environ["RAGAS_USE_MOCK"] = "false"
    config3 = RagasConfig.from_env()
    assert config3.validate()
    print("   Endpoint validation passed")


def test_batch_processor():
    print("\n2. Test RagasBatchProcessor")
    
    config = RagasConfig(
        enabled=True,
        use_mock=True,
        batch_size=5,
        enable_progress_monitor=True,
    )
    
    processor = RagasBatchProcessor(config)
    mock_eval = MockRagasEvaluator()
    
    # Test batch processing
    questions = [f"Q{i}" for i in range(1, 12)]  # 11 questions
    answers = [f"A{i}" for i in range(1, 12)]
    contexts = [[f"C{i}"] for i in range(1, 12)]
    
    print(f"   Processing {len(questions)} questions in batches of {config.batch_size}")
    
    result = processor.process_in_batches(
        questions=questions,
        answers=answers,
        contexts=contexts,
        evaluator=mock_eval,
    )
    
    print(f"   Result keys: {list(result.keys())}")
    print(f"   Avg faithfulness: {result['avg_faithfulness']:.2f}")
    print(f"   Avg relevancy: {result['avg_answer_relevancy']:.2f}")
    
    assert result['avg_faithfulness'] > 0
    assert len(result['faithfulness']) == 11
    print("   Batch processing passed")


def test_config_export():
    print("\n3. Test config export to environment")
    
    config = RagasConfig(
        enabled=True,
        use_mock=True,
        batch_size=20,
    )
    
    config.to_env()
    
    import os
    assert os.getenv("RAGAS_ENABLED") == "true"
    assert os.getenv("RAGAS_BATCH_SIZE") == "20"
    print("   Environment export passed")


if __name__ == "__main__":
    print("=" * 60)
    print("Ragas Config & Batch Processor Test")
    print("=" * 60)
    
    test_config_creation()
    test_batch_processor()
    test_config_export()
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)