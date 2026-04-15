import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import health, service
from app.schemas import QueryRequest


def main() -> None:
    health_resp = health()
    print('[health]')
    print(json.dumps(health_resp.model_dump(), ensure_ascii=False, indent=2))

    query_resp = service.process(
        QueryRequest(query='2026年陕西电力市场中长期交易流程是什么？', session_id='smoke:local', top_k=5)
    )
    print('\n[query]')
    print(
        json.dumps(
            {
                'mode': query_resp.mode.value,
                'province_code': query_resp.province_code,
                'conclusion': query_resp.conclusion,
                'provincial_evidence_count': len(query_resp.provincial_evidence),
                'global_evidence_count': len(query_resp.global_evidence),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
