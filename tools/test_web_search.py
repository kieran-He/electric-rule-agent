"""
测试联网搜索功能

测试 Tavily API 集成是否正常工作。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings
from app.core.web_search import WebSearchClient, create_web_search_client


def test_web_search_client():
    """直接测试 WebSearchClient"""
    print("=" * 60)
    print("测试 WebSearchClient")
    print("=" * 60)
    
    if not settings.tavily_api_key:
        print("⚠️ TAVILY_API_KEY 未配置")
        print("\n配置方法：")
        print("1. 访问 https://tavily.com 注册获取 API Key")
        print("2. 在 .env 文件中添加：")
        print("   TAVILY_API_KEY=your-api-key-here")
        return
    
    client = WebSearchClient(
        api_key=settings.tavily_api_key,
        max_results=5,
    )
    
    query = "陕西省现货市场交易时间"
    print(f"\n搜索查询: {query}")
    
    results = client.search(query)
    
    if results:
        print(f"\n找到 {len(results)} 个结果:")
        for i, r in enumerate(results, 1):
            print(f"\n{i}. {r.get('title', '无标题')}")
            print(f"   内容: {r.get('content', '')[:100]}...")
            print(f"   URL: {r.get('url', '')}")
        
        print("\n格式化上下文:")
        context = client.format_results_for_context(results)
        print(context[:500])
        print("✅ Tavily API 测试通过")
    else:
        print("⚠️ 搜索无结果")


def test_orchestrator_web_search():
    """测试 Orchestrator 的联网搜索回退"""
    print("\n" + "=" * 60)
    print("测试 Orchestrator 联网搜索回退")
    print("=" * 60)
    
    from app.langchain.orchestrator_hybrid import HybridQAOrchestrator
    from app.schemas.query import QueryRequest
    
    orchestrator = HybridQAOrchestrator()
    
    # 使用一个知识库中应该没有的问题触发 web search
    query = "2025年最新的AI大模型有哪些"
    print(f"\n测试查询: {query}")
    print("(此问题不在电力政策知识库，应触发联网搜索)")
    
    result = orchestrator.run(QueryRequest(
        query=query,
        session_id="test_web_search_001",
        province_codes=["SN"],
    ))
    
    print(f"\n答案: {result.answer}")
    
    if "网络搜索" in result.answer or "非知识库" in result.answer:
        print("✅ 联网搜索回退机制触发成功")
    else:
        print("⚠️ 可能未触发联网搜索回退")


def main():
    print("联网搜索功能测试\n")
    
    print("配置状态:")
    print(f"  TAVILY_API_KEY: {'已配置' if settings.tavily_api_key else '未配置'}")
    print(f"  WEB_SEARCH_ENABLED: {settings.web_search_enabled}")
    print(f"  WEB_SEARCH_MAX_RESULTS: {settings.web_search_max_results}")
    
    test_web_search_client()
    
    if settings.tavily_api_key:
        test_orchestrator_web_search()
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()