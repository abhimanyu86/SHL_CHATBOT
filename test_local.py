"""
test_local.py
-------------
Run this to verify your agent works correctly locally
before deploying to Render.

Usage:  python test_local.py

Tests all 4 behaviors:
  1. Clarify vague query
  2. Recommend on specific query
  3. Refine existing shortlist
  4. Compare two assessments
  5. Refuse off-topic
"""

import requests
import json

BASE = "http://localhost:8000"


def chat(messages: list[dict]) -> dict:
    resp = requests.post(f"{BASE}/chat", json={"messages": messages})
    resp.raise_for_status()
    return resp.json()


def print_result(label: str, result: dict):
    print(f"\n{'='*60}")
    print(f"TEST: {label}")
    print(f"{'='*60}")
    print(f"REPLY: {result['reply']}")
    print(f"RECOMMENDATIONS ({len(result['recommendations'])}):")
    for r in result["recommendations"]:
        print(f"  - {r['name']} [{r['test_type']}]")
        print(f"    {r['url']}")
    print(f"END_OF_CONVERSATION: {result['end_of_conversation']}")


def test_health():
    resp = requests.get(f"{BASE}/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
    print("✅ Health check passed")


def test_vague_query():
    """Agent should ask a clarifying question, NOT recommend."""
    messages = [{"role": "user", "content": "I need an assessment."}]
    result = chat(messages)
    print_result("VAGUE QUERY — expect clarifying question, no recommendations", result)
    assert len(result["recommendations"]) == 0, "❌ Should not recommend on vague query"
    assert result["end_of_conversation"] == False
    print("✅ Vague query test passed")


def test_specific_query():
    """Agent should recommend immediately on a specific query."""
    messages = [
        {
            "role": "user",
            "content": "We're hiring a mid-level Java developer, around 4 years experience, backend focus.",
        }
    ]
    result = chat(messages)
    print_result("SPECIFIC QUERY — expect recommendations", result)
    assert len(result["recommendations"]) >= 1, "❌ Should recommend on specific query"
    print("✅ Specific query test passed")


def test_multi_turn_and_refine():
    """Multi-turn conversation with refinement."""
    # Turn 1 — genuinely vague, no role
    messages = [{"role": "user", "content": "I need an assessment."}]
    r1 = chat(messages)
    print_result("TURN 1 — vague", r1)
    assert len(r1["recommendations"]) == 0
    # Turn 2 — add context
    messages += [
        {"role": "assistant", "content": r1["reply"]},
        {"role": "user", "content": "CXOs and directors, 15+ years experience. Selection purpose."},
    ]
    r2 = chat(messages)
    print_result("TURN 2 — specific context", r2)
    assert len(r2["recommendations"]) >= 1

    # Turn 3 — refine (add personality)
    messages += [
        {"role": "assistant", "content": r2["reply"]},
        {"role": "user", "content": "Actually, also add a cognitive reasoning test."},
    ]
    r3 = chat(messages)
    print_result("TURN 3 — refine: add cognitive", r3)
    assert len(r3["recommendations"]) >= 1
    print("✅ Multi-turn + refine test passed")


def test_compare():
    """Compare question — should answer from catalog, empty recommendations."""
    messages = [
        {"role": "user", "content": "What is the difference between OPQ32r and the Verify G+?"}
    ]
    result = chat(messages)
    print_result("COMPARE — expect explanation, no recommendations", result)
    assert len(result["recommendations"]) == 0, "❌ Compare should return empty recommendations"
    print("✅ Compare test passed")


def test_off_topic_refusal():
    """Off-topic — agent should refuse."""
    messages = [
        {
            "role": "user",
            "content": "What is the best hiring strategy for reducing bias in interviews?",
        }
    ]
    result = chat(messages)
    print_result("OFF-TOPIC — expect refusal", result)
    assert len(result["recommendations"]) == 0
    print("✅ Off-topic refusal test passed")


def test_legal_refusal():
    """Legal question — agent should refuse."""
    messages = [
        {"role": "user", "content": "Are we legally required to test all candidates under HIPAA?"}
    ]
    result = chat(messages)
    print_result("LEGAL QUESTION — expect refusal", result)
    assert len(result["recommendations"]) == 0
    print("✅ Legal refusal test passed")


def test_end_of_conversation():
    """User confirms — end_of_conversation should be True."""
    messages = [
        {"role": "user", "content": "We're hiring graduate financial analysts. Need numerical reasoning and finance knowledge tests."},
        {"role": "assistant", "content": "Here are my recommendations: ..."},
        {"role": "user", "content": "Perfect, that's exactly what we need. Confirmed."},
    ]
    result = chat(messages)
    print_result("END OF CONVERSATION — expect true", result)
    assert result["end_of_conversation"] == True, "❌ Should set end_of_conversation=True on confirmation"
    print("✅ End of conversation test passed")


if __name__ == "__main__":
    print("\n🔍 Running SHL Recommender tests...\n")
    test_health()
    test_vague_query()
    test_specific_query()
    test_multi_turn_and_refine()
    test_compare()
    test_off_topic_refusal()
    test_legal_refusal()
    test_end_of_conversation()
    print("\n✅ All tests completed.")