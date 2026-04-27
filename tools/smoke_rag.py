import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from app.main import app
from app.core.llm_client import LLMGenerationError
from app.core.repository import RepositoryError
from app.schemas import QueryMode, QueryRequest

QUERY_TEXT = '请按"准入-申报-出清-执行-结算"列出陕西中长期交易流程，并给出处条款'


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Smoke test for RAG query chain.',
        epilog=(
            'Examples:\n'
            '  python tools/smoke_rag.py --query "陕西中长期交易流程是什么" --province SN\n'
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('--query', default=QUERY_TEXT, help='query text')
    parser.add_argument('--province', default='SN', help='province code (default: SN)')
    parser.add_argument('--top-k', type=int, default=5, help='retrieval top_k (must be >= 1)')
    parser.add_argument('--session-id', default='smoke:local', help='session id')
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')

    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.query or not args.query.strip():
        parser.error('--query must not be empty')
    if args.top_k < 1:
        parser.error('--top-k must be >= 1')

    province_codes = [args.province.strip().upper()] if args.province and args.province.strip() else ["SN"]

    client = TestClient(app)

    try:
        health_resp = client.get("/health")
        print('[health]')
        print(json.dumps(health_resp.json(), ensure_ascii=False, indent=2))

        query_resp = client.post(
            "/query",
            json={
                "query": args.query.strip(),
                "session_id": args.session_id,
                "top_k": args.top_k,
                "province_codes": province_codes,
            },
        )
        print('\n[query]')
        print(json.dumps(query_resp.json(), ensure_ascii=False, indent=2))
        
        if query_resp.status_code != 200:
            return 1
        return 0
    except (LLMGenerationError, RepositoryError, ValueError) as exc:
        print(
            json.dumps(
                {
                    'error': exc.__class__.__name__,
                    'detail': str(exc),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1


if __name__ == '__main__':
    raise SystemExit(main())