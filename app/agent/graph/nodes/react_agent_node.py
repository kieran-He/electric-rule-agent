import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Tuple

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.agent.graph.state import ElectricityAgentState
from app.agent.graph.tools.tool_registry import get_tools_for_agent

logger = logging.getLogger(__name__)

REACT_SYSTEM_PROMPT = """你是电力市场分析助手。
可用工具：retrieve_policy, fetch_electricity_data, analyze_statistics, web_search。

省份代码对照（重要）:
- 陕西 = SN, 山西 = SX, 山东 = SD, 甘肃 = GS
- 河南 = HA, 湖北 = HB, 湖南 = HN, 安徽 = AH
- 江苏 = JS, 浙江 = ZJ, 北京 = BJ, 上海 = SH
- 广东 = GD, 四川 = SC, 重庆 = CQ, 云南 = YN
- 内蒙古 = NM(蒙西MX,蒙东MD), 河北 = HE(冀南JN,冀北JB)
- 其他省份请参考此格式使用两位大写字母代码

工具调用规则（重要）:
- retrieve_policy 和 web_search：只需要调用工具，无需提供query参数（系统会自动使用用户原始问题）
- provinces参数：如果问题明确提到省份，提供正确的省份代码（如["SN", "HA"]）；否则可省略，系统会自动推断
- fetch_electricity_data：需要提供 metric 参数（load/generation/price/new_energy）

回答规则：
1. 优先用工具获取事实，再回答。
2. 如果已获得足够信息，直接输出答案，不要继续调用工具。
3. 政策相关问题优先调用 retrieve_policy；涉及最新动态可调用 web_search。
4. 当引用政策片段时，请在正文中使用 [引用](#chunk-N) 格式。
5. 不要在答案末尾单独追加"参考文献"列表。
"""


def _infer_cited_indices_from_answer(answer: str, policy_chunks: List[Dict]) -> List[int]:
    """Infer cited chunks from explicit text matches when no chunk marker exists."""
    hits: List[Tuple[int, int]] = []

    for idx, chunk in enumerate(policy_chunks, 1):
        source = (chunk.get("source") or "").strip()
        title_path = (chunk.get("title_path") or "").strip()
        content = (chunk.get("content") or "").strip()

        # Priority 1: source/title mentioned in answer.
        for anchor in (source, title_path):
            if anchor and anchor in answer:
                hits.append((answer.find(anchor), idx))
                break
        else:
            # Priority 2: phrase match from chunk content.
            # Use medium-length phrases to reduce accidental hits.
            phrases = []
            for part in re.split(r"[。；;，,\n\r\t ]+", content):
                part = part.strip()
                if 8 <= len(part) <= 40:
                    phrases.append(part)
                if len(phrases) >= 6:
                    break
            for phrase in phrases:
                pos = answer.find(phrase)
                if pos >= 0:
                    hits.append((pos, idx))
                    break
                # Relaxed match: large common substring indicates same requirement statement.
                best = 0
                best_pos = -1
                for i in range(len(phrase)):
                    for j in range(i + 1, len(phrase) + 1):
                        frag = phrase[i:j]
                        if len(frag) <= best:
                            continue
                        p = answer.find(frag)
                        if p >= 0:
                            best = len(frag)
                            best_pos = p
                if best >= 12 and best_pos >= 0:
                    hits.append((best_pos, idx))
                    break

    hits.sort(key=lambda x: x[0])
    seen = set()
    ordered: List[int] = []
    for _, idx in hits:
        if idx not in seen:
            seen.add(idx)
            ordered.append(idx)
    return ordered


def _fill_and_renumber_citations(answer: str, policy_chunks: List[Dict]) -> Tuple[str, List[Dict]]:
    """Normalize explicit chunk markers and keep cited chunks in answer appearance order."""
    if not answer or not policy_chunks:
        return answer, policy_chunks

    token_pattern = re.compile(
        r'\[[^\]]*\]\(#chunk-(\d+)\)'  # [any](#chunk-n)
        r'|\[chunk-(\d+)\]'               # [chunk-n]
        r'|#chunk-(\d+)\b',               # #chunk-n
        flags=re.IGNORECASE,
    )
    marker_hint_pattern = re.compile(r'chunk-\w+', flags=re.IGNORECASE)

    raw_matches: List[int] = []
    seen = set()
    cited_indices: List[int] = []
    for match in token_pattern.finditer(answer):
        idx_text = match.group(1) or match.group(2) or match.group(3)
        if not idx_text:
            continue
        idx = int(idx_text)
        raw_matches.append(idx)
        if 1 <= idx <= len(policy_chunks) and idx not in seen:
            seen.add(idx)
            cited_indices.append(idx)

    if raw_matches and not cited_indices:
        logger.warning(
            "[ReActAgent] Citation markers found but no valid indices. raw=%s, chunk_count=%s, sample=%r",
            raw_matches,
            len(policy_chunks),
            answer[:200],
        )
    elif not raw_matches and marker_hint_pattern.search(answer):
        logger.warning(
            "[ReActAgent] Potential citation-like text found but no explicit marker matched. sample=%r",
            answer[:200],
        )

    if not cited_indices:
        inferred = _infer_cited_indices_from_answer(answer, policy_chunks)
        if inferred:
            ordered_chunks = [policy_chunks[i - 1] for i in inferred]
            logger.info(
                "[ReActAgent] Citation fallback by text-match: inferred_indices=%s, chunks_kept=%s",
                inferred,
                len(ordered_chunks),
            )
            return answer, ordered_chunks
        logger.info(
            "[ReActAgent] Citation post-processing: raw_markers=%s, valid=0, chunks_kept=0",
            len(raw_matches),
        )
        return answer, []

    renumber_map = {old_idx: new_idx for new_idx, old_idx in enumerate(cited_indices, 1)}
    processed_answer = answer

    # Phase 1: replace all recognized citation forms with temporary tokens.
    for old_idx in cited_indices:
        tmp = f"__CITE_TMP_{old_idx}__"
        processed_answer = re.sub(
            rf'\[[^\]]*\]\(#chunk-{old_idx}\)',
            tmp,
            processed_answer,
            flags=re.IGNORECASE,
        )
        processed_answer = re.sub(
            rf'\[chunk-{old_idx}\]',
            tmp,
            processed_answer,
            flags=re.IGNORECASE,
        )
        processed_answer = re.sub(
            rf'#chunk-{old_idx}\b',
            tmp,
            processed_answer,
            flags=re.IGNORECASE,
        )

    # Phase 2: fill doc_name and renumber sequentially.
    for old_idx in cited_indices:
        new_idx = renumber_map[old_idx]
        chunk = policy_chunks[old_idx - 1]
        doc_name = (chunk.get("source") or "").strip() or "unknown"
        processed_answer = processed_answer.replace(
            f"__CITE_TMP_{old_idx}__",
            f"[{doc_name}](#chunk-{new_idx})",
        )

    ordered_chunks = [policy_chunks[old_idx - 1] for old_idx in cited_indices]
    logger.info(
        "[ReActAgent] Citation post-processing: raw_markers=%s, valid_indices=%s, renumber_map=%s, chunks_kept=%s",
        len(raw_matches),
        cited_indices,
        renumber_map,
        len(ordered_chunks),
    )
    return processed_answer, ordered_chunks


def _build_messages(state: ElectricityAgentState) -> List:
    messages = [SystemMessage(content=REACT_SYSTEM_PROMPT)]

    history = state.get("messages", [])
    max_history_pairs = 4
    if len(history) > max_history_pairs * 2:
        history = history[-(max_history_pairs * 2):]

    for msg in history:
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
        elif isinstance(msg, HumanMessage):
            messages.append(msg)
        elif isinstance(msg, AIMessage):
            messages.append(msg)
        elif isinstance(msg, SystemMessage):
            continue
        else:
            messages.append(HumanMessage(content=str(msg)))

    if state.get("tool_results") and state.get("last_tool_calls"):
        tool_calls_for_msg = []
        for tc in state["last_tool_calls"]:
            tool_calls_for_msg.append(
                {
                    "name": tc.get("name", ""),
                    "args": tc.get("args", {}),
                    "id": tc.get("id", f"tool_{tc.get('name', 'unknown')}")
                }
            )

        messages.append(AIMessage(content="", tool_calls=tool_calls_for_msg))

        for result in state["tool_results"]:
            tool_name = result.get("tool_name", "unknown")
            tool_output = result.get("output", "")
            tool_call_id = result.get("tool_call_id", f"tool_{tool_name}")
            messages.append(ToolMessage(content=f"工具 {tool_name} 返回结果:\n{tool_output}", tool_call_id=tool_call_id))

        query = state["query"]
        sufficient_info = state.get("sufficient_info", False)
        sufficiency_reason = state.get("sufficiency_reason", "")
        need_web_search = state.get("need_web_search", False)
        retrieval_quality = state.get("retrieval_quality", {})
        has_web_search = any(r.get("tool_name") == "web_search" for r in state.get("tool_results", []))

        web_hint = ""
        if need_web_search and not has_web_search:
            reason = retrieval_quality.get("reason", "检索结果不足")
            web_hint = f"\n【检索质量提示】知识库结果不足: {reason}\n建议调用 web_search 补充信息。\n"

        if sufficient_info:
            reminder = (
                f"{web_hint}\n"
                f"【系统提示】信息已充足: {sufficiency_reason}\n"
                f"【重要】请基于上述工具结果直接生成完整答案，不要继续调用工具。用户原始问题是: {query}"
            )
        else:
            reminder = (
                f"{web_hint}\n"
                f"【重要提醒】用户原始问题是: {query}\n"
                "请基于工具结果回答该问题。若信息仍不足可以继续调用工具；若信息已足够请直接作答。"
            )
        messages.append(HumanMessage(content=reminder))

    if not state.get("messages") or state["iteration_count"] == 0:
        query = state["query"]
        provinces = state.get("provinces", ["SN"])
        province_str = ", ".join(provinces)
        user_prompt = f"用户问题: {query}\n关注省份: {province_str}"
        messages.append(HumanMessage(content=user_prompt))

    return messages


def react_agent_node(state: ElectricityAgentState) -> Dict[str, Any]:
    """ReAct agent node: decides next action based on current state."""
    logger.info(f"[ReActAgent] Iteration {state['iteration_count']}/{state['max_iterations']}")

    thoughts = state.get("thoughts", [])
    thoughts.append(
        {
            "iteration": state["iteration_count"],
            "phase": "thinking",
            "timestamp": datetime.now().isoformat(),
        }
    )

    try:
        from app.agent.graph.electricity_agent_graph import _get_current_instance

        graph_instance = _get_current_instance()
        if not graph_instance:
            logger.error("[ReActAgent] No graph instance available")
            return {
                "answer": "系统错误：无法访问分析图实例",
                "done": True,
                "errors": [{"error": "no_graph_instance"}],
            }

        llm_wrapper = graph_instance.llm_wrapper
        settings = graph_instance.settings

        enabled_tools = getattr(settings, "tools_enabled_list", None)
        if enabled_tools is None:
            tools_str = getattr(settings, "tools_enabled", None)
            if tools_str:
                enabled_tools = [t.strip() for t in tools_str.split(",") if t.strip()]
        tools = get_tools_for_agent(enabled_tools)

        messages = _build_messages(state)
        logger.info(f"[ReActAgent] Calling LLM with {len(messages)} messages, {len(tools)} tools")

        response = llm_wrapper.invoke_with_tools(messages, tools)
        logger.info(f"[ReActAgent] Response type: {type(response).__name__}")

        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_calls = []
            for tc in response.tool_calls:
                if isinstance(tc, dict):
                    tool_calls.append(
                        {
                            "name": tc.get("name", ""),
                            "args": tc.get("args", {}),
                            "id": tc.get("id", ""),
                        }
                    )
                else:
                    tool_calls.append(
                        {
                            "name": getattr(tc, "name", ""),
                            "args": getattr(tc, "args", {}),
                            "id": getattr(tc, "id", ""),
                        }
                    )

            logger.info(f"[ReActAgent] Tool calls: {[tc['name'] for tc in tool_calls]}")
            thoughts.append(
                {
                    "iteration": state["iteration_count"],
                    "phase": "tool_call",
                    "tools": [tc["name"] for tc in tool_calls],
                }
            )

            existing_chunks = state.get("policy_chunks", [])
            return {
                "tool_calls": tool_calls,
                "last_tool_calls": tool_calls,
                "iteration_count": state["iteration_count"] + 1,
                "thoughts": thoughts,
                "done": False,
                "policy_chunks": existing_chunks,
            }

        content = ""
        if hasattr(response, "content"):
            if isinstance(response.content, str):
                content = response.content
            elif isinstance(response.content, list):
                for block in response.content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        content += block.get("text", "")
                    elif hasattr(block, "text"):
                        content += block.text

        policy_chunks = state.get("policy_chunks", [])
        ordered_chunks = policy_chunks
        if policy_chunks:
            content, ordered_chunks = _fill_and_renumber_citations(content, policy_chunks)

        logger.info(f"[ReActAgent] Final answer: {len(content)} chars, {len(ordered_chunks)} chunks")
        thoughts.append(
            {
                "iteration": state["iteration_count"],
                "phase": "final_answer",
                "answer_length": len(content),
                "chunks_kept": len(ordered_chunks),
            }
        )

        return {
            "answer": content,
            "policy_chunks": ordered_chunks,
            "done": True,
            "thoughts": thoughts,
            "confidence": 0.85,
        }

    except Exception as e:
        logger.exception(f"[ReActAgent] Failed: {e}")
        thoughts.append(
            {
                "iteration": state["iteration_count"],
                "phase": "error",
                "error": str(e),
            }
        )
        return {
            "answer": f"处理请求时出现错误: {str(e)[:100]}",
            "done": True,
            "errors": [{"error": str(e), "iteration": state["iteration_count"]}],
            "thoughts": thoughts,
        }
