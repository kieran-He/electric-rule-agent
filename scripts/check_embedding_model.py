import sys
sys.stdout.reconfigure(encoding='utf-8')
from sentence_transformers import SentenceTransformer
import numpy as np

model_name = "BAAI/bge-small-zh-v1.5"
print(f"Loading model: {model_name}")

try:
    model = SentenceTransformer(model_name)
    test_text = "测试文本"
    embedding = model.encode(test_text)
    print(f"Embedding dimension: {len(embedding)}")
    print(f"Expected dimension: 384")
    
    if len(embedding) == 384:
        print("OK - dimension matches")
    else:
        print(f"MISMATCH - got {len(embedding)} instead of 384")
except Exception as e:
    print(f"Error: {e}")