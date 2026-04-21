#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.session import SessionLocal
from app.db.models.document import Document
from app.db.models.clause import Clause
from app.db.models.rule_tag import RuleTag
from sqlalchemy import delete

with SessionLocal() as db:
    db.execute(delete(RuleTag))
    db.execute(delete(Clause))
    db.execute(delete(Document))
    db.commit()
    print("SQL DB cleared")