import requests

BASE = "http://localhost:8000"
history = []
turn_count = 0
MAX_TURNS = 7

print("\n🤖 SHL Assessment Recommender")
print("Type your message and press Enter.")
print("Type 'quit' to exit.\n")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() in ("quit", "exit", "q"):
        print("Goodbye!")
        break

    if not user_input:
        continue

    if turn_count >= MAX_TURNS:
        print("⚠️ Maximum conversation turns reached. Please restart.")
        break

    history.append({"role": "user", "content": user_input})

    try:
        resp = requests.post(
            f"{BASE}/chat",
            json={"messages": history},
            timeout=30,
        )
        resp.raise_for_status()
        result = resp.json()

        reply = result["reply"]
        recs = result["recommendations"]
        eoc = result["end_of_conversation"]

        print(f"\nAgent: {reply}")

        if recs:
            print(f"\n📋 Recommendations ({len(recs)}):")
            for i, r in enumerate(recs, 1):
                print(f"  {i}. {r['name']} [{r['test_type']}]")
                print(f"     {r['url']}")

        print()

        if eoc:
            print("✅ Conversation complete.")
            break

        history.append({"role": "assistant", "content": reply})
        turn_count += 1

    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP Error: {e}")
        # Remove last user message so conversation stays clean
        if history and history[-1]["role"] == "user":
            history.pop()
        break
    except Exception as e:
        print(f"❌ Error: {e}")
        break
