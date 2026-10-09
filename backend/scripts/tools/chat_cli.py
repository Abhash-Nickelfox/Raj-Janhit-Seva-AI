"""Phase 3 interactive test harness - a full multi-turn conversation against
the LangGraph conversation layer (app/graph/chat.py), unlike query_kb.py/
query_retrieval.py which test single-shot retrieval only.

Usage:
  python scripts/tools/chat_cli.py                # interactive, one running session
  python scripts/tools/chat_cli.py --debug        # also print profile/shortlist/intent each turn
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.stdout.reconfigure(encoding="utf-8")

from app.graph.chat import build_graph  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--thread", default="cli-session", help="conversation/session id")
    args = parser.parse_args()

    print("Loading knowledge base + building conversation graph...")
    graph = build_graph()
    config = {"configurable": {"thread_id": args.thread}}

    print("Jan Seva scheme chatbot (Phase 3). Type 'quit' to exit, 'reset' for a fresh session.\n")
    while True:
        try:
            msg = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not msg:
            continue
        if msg.lower() in ("quit", "exit"):
            break
        if msg.lower() == "reset":
            args.thread = args.thread + "-r"
            config = {"configurable": {"thread_id": args.thread}}
            print("(new session started)\n")
            continue

        result = graph.invoke({"raw_message": msg}, config=config)
        print(f"bot> {result.get('last_reply', '(no reply generated)')}\n")

        if args.debug:
            print(f"  [intent={result.get('intent')} turn_type={result.get('turn_type')} "
                  f"need='{result.get('need_text')}' category={result.get('need_category')} "
                  f"benefit={result.get('need_benefit_type')}]")
            print(f"  [profile={result.get('profile')}]")
            print(f"  [shown_scheme={result.get('shown_scheme_name')}]")
            if result.get("shortlist"):
                names = [s["scheme_name"] for s in result["shortlist"][:5]]
                print(f"  [shortlist top5={names}]")
            print()


if __name__ == "__main__":
    main()
