"""
Web Search Client using Tavily API

Provides real-time web search capability for RAG fallback.
"""
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class WebSearchClient:
    """
    Tavily API client for web search.
    
    Tavily is optimized for AI agents and RAG systems,
    returning clean content summaries instead of raw HTML.
    
    Usage:
        client = WebSearchClient(api_key="your-tavily-key")
        results = client.search("陕西省现货交易时间")
        # Returns: [{"title": "...", "content": "...", "url": "..."}]
    
    API Key: Get from https://tavily.com (free 1000 requests/month)
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        max_results: int = 5,
        search_depth: str = "basic",
    ):
        self.api_key = api_key
        self.max_results = max_results
        self.search_depth = search_depth
        self._client = None
        
    def _get_client(self):
        """Lazy load Tavily client to avoid import errors if not installed."""
        if self._client is None:
            try:
                from tavily import TavilyClient
                if not self.api_key:
                    raise ValueError("TAVILY_API_KEY not configured")
                self._client = TavilyClient(api_key=self.api_key)
            except ImportError:
                logger.error("tavily-python not installed. Run: pip install tavily-python")
                raise ImportError("tavily-python package required for web search")
        return self._client
    
    def search(
        self,
        query: str,
        search_depth: Optional[str] = None,
        include_domains: Optional[List[str]] = None,
    ) -> list[dict]:
        """
        Search web for query and return structured results.
        
        Args:
            query: Search query string
            search_depth: "basic" or "advanced" (default: self.search_depth)
            include_domains: List of domains to prioritize (e.g., ["gov.cn"])
            
        Returns:
            List of results with title, content, url fields
            Example: [
                {
                    "title": "陕西省现货交易规则",
                    "content": "交易时间为每日9:00-11:00...",
                    "url": "https://example.com/article"
                }
            ]
        """
        try:
            client = self._get_client()
            depth = search_depth or self.search_depth
            
            search_params = {
                "query": query,
                "max_results": self.max_results,
                "search_depth": depth,
            }
            if include_domains:
                search_params["include_domains"] = include_domains
            
            logger.info(f"Searching web for: {query[:50]} (depth={depth}, domains={include_domains})")
            
            response = client.search(**search_params)
            
            results = response.get("results", [])
            logger.info(f"Found {len(results)} web results")
            
            return results
            
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return []
    
    def format_results_for_context(self, results: list[dict]) -> str:
        """
        Format search results as context string for LLM.
        
        Args:
            results: List of search results
            
        Returns:
            Formatted string with title and content from each result
        """
        if not results:
            return "未找到相关网络信息"
        
        lines = []
        for i, result in enumerate(results, start=1):
            title = result.get("title", "无标题")
            content = result.get("content", "")
            url = result.get("url", "")
            
            # Truncate long content
            if len(content) > 300:
                content = content[:300] + "..."
            
            lines.append(f"{i}. {title}")
            lines.append(f"   {content}")
            if url:
                lines.append(f"   来源: {url}")
            lines.append("")
        
        return "\n".join(lines)
    
    def is_available(self) -> bool:
        """Check if web search is available (API key configured)."""
        return bool(self.api_key)


def create_web_search_client(settings) -> Optional[WebSearchClient]:
    """
    Create WebSearchClient from settings.
    
    Returns None if not configured, allowing graceful fallback.
    """
    if not settings.web_search_enabled or not settings.tavily_api_key:
        logger.info("Web search disabled or API key not configured")
        return None
    
    return WebSearchClient(
        api_key=settings.tavily_api_key,
        max_results=settings.web_search_max_results,
        search_depth=getattr(settings, 'web_search_depth', 'advanced'),
    )