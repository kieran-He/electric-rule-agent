import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
print(f"embedding_model: {settings.embedding_model}")