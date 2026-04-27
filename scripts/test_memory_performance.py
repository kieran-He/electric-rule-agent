#!/usr/bin/env python3
"""
Memory Performance Test - Compare Token Consumption

Tests token consumption for different memory strategies:
- Strategy 1: Fixed 6 turns (original)
- Strategy 2: 4 turns + summary (improved)
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))

from app.db.session import SessionLocal, init_db
from app.services.conversation_service import ConversationService

init_db()


def test_token_consumption():
    """Test token consumption for 4-turn + summary strategy"""
    
    service = ConversationService(SessionLocal, max_history_turns=4, enable_summary=True)
    session_id = "perf_test_session"
    
    # Clear previous data
    service.clear_history(session_id)
    
    # Simulate 8 rounds of conversation
    print("Simulating 8 rounds of conversation...")
    for i in range(8):
        service.append_turn(
            session_id=session_id,
            user_query=f"测试问题{i+1}：陕西交易规则的具体内容是什么",
            bot_reply=f"测试回答{i+1}：根据《陕西省电力市场交易实施细则》，交易规则包括中长期交易、现货交易、辅助服务等多个方面，具体流程包括注册、申报、竞价、结算等环节...",
            intent="clause_qa",
            province_code="SN"
        )
        print(f"  Turn {i+1} appended")
    
    # Get history and measure token consumption
    history = service.get_history(session_id)
    
    # Count tokens (approximate: 1 Chinese char = 1.5 tokens)
    total_chars = sum(len(h) for h in history)
    estimated_tokens = total_chars * 1.5
    
    print(f"\nMemory Strategy: 4 turns + summary (improved)")
    print(f"History items: {len(history)}")
    print(f"Total characters: {total_chars}")
    print(f"Estimated tokens: {estimated_tokens:.0f}")
    print(f"\nHistory preview:")
    for i, h in enumerate(history[:6]):
        print(f"  {i+1}. {h[:60]}...")
    
    # Cleanup
    service.clear_history(session_id)
    print("\nTest completed, data cleaned up")


def test_summary_generation():
    """Test summary generation at turn 5"""
    
    service = ConversationService(SessionLocal, max_history_turns=4, enable_summary=True)
    session_id = "summary_test_session"
    
    # Clear previous data
    service.clear_history(session_id)
    
    # Add 5 turns to trigger summary generation
    print("Testing summary generation at turn 5...")
    for i in range(5):
        service.append_turn(
            session_id=session_id,
            user_query=f"问题{i+1}",
            bot_reply=f"回答{i+1}" * 20,
            intent="test"
        )
    
    # Get history (should contain summary + 4 recent turns)
    history = service.get_history(session_id)
    
    print(f"\nHistory after 5 turns:")
    print(f"  Items: {len(history)}")
    print(f"  First item (summary): {history[0][:80]}...")
    print(f"  Recent turns: {len(history) - 1} items")
    
    # Cleanup
    service.clear_history(session_id)


def compare_strategies():
    """Compare token consumption: 6 turns vs 4 turns + summary"""
    
    print("\n" + "=" * 60)
    print("TOKEN CONSUMPTION COMPARISON")
    print("=" * 60)
    
    # Strategy 1: 6 turns (original)
    print("\nStrategy 1: Fixed 6 turns (original)")
    print("-" * 40)
    
    # Simulate 6 turns
    avg_query_length = 50  # chars
    avg_reply_length = 150  # chars
    
    # 6 turns = 12 items (Q+A)
    total_chars_6 = 6 * 2 * (avg_query_length + avg_reply_length)
    estimated_tokens_6 = total_chars_6 * 1.5
    
    print(f"Turns: 6")
    print(f"Items: 12 (Q+A)")
    print(f"Total chars: {total_chars_6}")
    print(f"Estimated tokens: {estimated_tokens_6:.0f}")
    
    # Strategy 2: 4 turns + summary
    print("\nStrategy 2: 4 turns + summary (improved)")
    print("-" * 40)
    
    # Summary: 200 chars
    summary_chars = 200
    summary_tokens = summary_chars * 1.5
    
    # Recent 4 turns
    recent_chars = 4 * 2 * (avg_query_length + avg_reply_length)
    recent_tokens = recent_chars * 1.5
    
    total_tokens_4 = summary_tokens + recent_tokens
    
    print(f"Summary chars: {summary_chars}")
    print(f"Summary tokens: {summary_tokens:.0f}")
    print(f"Recent turns: 4")
    print(f"Recent items: 8 (Q+A)")
    print(f"Recent chars: {recent_chars}")
    print(f"Recent tokens: {recent_tokens:.0f}")
    print(f"Total tokens: {total_tokens_4:.0f}")
    
    # Comparison
    reduction_pct = (estimated_tokens_6 - total_tokens_4) / estimated_tokens_6 * 100
    reduction_tokens = estimated_tokens_6 - total_tokens_4
    
    print("\n" + "=" * 60)
    print(f"Token reduction: {reduction_tokens:.0f} tokens (-{reduction_pct:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    print("Memory Performance Tests")
    print("=" * 60)
    
    # Test 1: Token consumption
    test_token_consumption()
    
    # Test 2: Summary generation
    test_summary_generation()
    
    # Test 3: Strategy comparison
    compare_strategies()