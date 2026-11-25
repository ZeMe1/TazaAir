import time
from agent import ask_agent, build_safety_card


def print_card(answer_text: str):
    try:
        card = build_safety_card(answer_text)
        print("\n--- Structured Safety Card (LCEL + Pydantic) ---")
        print(f"City:               {card.city}")
        print(f"Smog Status:        {card.smog_status}")
        print(f"Cigarettes / 24h:   {card.cigarettes_per_day}")
        print(f"Window Advice:      {card.window_advice}")
        print(f"FFP2 Mask Needed:   {card.mask_needed}")
        print("------------------------------------------------")
    except Exception as error:
        print(f"Could not generate safety card: {error}")


def run_live_chat():
    print("\nLive Chat started! Type 'card' for a Pydantic summary card, or 'exit' to quit.\n")
    last_answer = ""

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye! Breathe clean air.")
            break
        if user_input.lower() == "card":
            if not last_answer:
                print("Ask about a city's air quality first so I can build a card.\n")
            else:
                print_card(last_answer)
                print()
            continue

        response = ask_agent(user_input, session_id="live_user")
        last_answer = response["answer"]
        tools_list = ", ".join(response["tools_used"]) if response["tools_used"] else "none (from memory)"
        print(f"\n[Tools called: {tools_list}]")
        print(f"TazaAir: {last_answer}\n")


def run_demo_conversations():
    demos = [
        {
            "title": "Conversation 1 — Multi-Turn Smog Check, Mountain Escape & Diary (Memory Test)",
            "session_id": "demo_session_1",
            "messages": [
                "How is the air quality and wind in Almaty right now?",
                "If I walk outside there for 3 hours, how many cigarettes is that? Also check Medeu to see if the air is cleaner.",
                "Save a short note to my air diary comparing Almaty and Medeu today.",
            ],
            "show_card_on_first": True,
        },
        {
            "title": "Conversation 2 — Astana Window Ventilation Check (Russian)",
            "session_id": "demo_session_2",
            "messages": [
                "Какой сейчас воздух и ветер в Астане? Можно ли открыть окно для проветривания?",
            ],
            "show_card_on_first": True,
        },
        {
            "title": "Conversation 3 — Reading Saved Diary & Handling Invalid City",
            "session_id": "demo_session_3",
            "messages": [
                "Show me all saved notes from my air diary, and also check air quality in WakandaForeverCity.",
            ],
            "show_card_on_first": False,
        },
    ]

    for demo in demos:
        print("\n" + "=" * 70)
        print(demo["title"])
        print("=" * 70)

        for index, msg in enumerate(demo["messages"]):
            print(f"\nUser: {msg}")
            response = ask_agent(msg, session_id=demo["session_id"])
            tools_list = ", ".join(response["tools_used"]) if response["tools_used"] else "none"
            print(f"[Tools called: {tools_list}]")
            print(f"TazaAir: {response['answer']}")

            if demo["show_card_on_first"] and index == 0:
                print_card(response["answer"])

            time.sleep(2)


def main():
    print("=" * 55)
    print("   TazaAir — AI Smog Shield & Air Quality Agent")
    print("=" * 55)
    print("1. Start live chat (interactive mode)")
    print("2. Run 3 example conversations (demo mode)")

    choice = input("Select mode (1 or 2): ").strip()
    if choice == "2":
        run_demo_conversations()
    else:
        run_live_chat()


if __name__ == "__main__":
    main()
