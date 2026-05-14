import requests
import json

BASE = "http://localhost:8000"
history = []

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

        if eoc:
            print("\n✅ Conversation complete.")
            break

        history.append({"role": "assistant", "content": reply})
        print()

    except requests.exceptions.HTTPError as e:
        print(f"❌ Error: {e}")
        break
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        break