$env:EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
python -m uvicorn app.main:app --host localhost --port 8000