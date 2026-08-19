from agent import LocalAgent


def print_help():

    print(
        """
============================================================
Commands
============================================================

/help
    Show this help.

/stats
    Show memory statistics.

/memory <query>
    Search long-term memory.

/project <text>
    Add project memory.

/event <text>
    Add event memory.

/clear
    Delete all long-term memories.

/exit
    Exit program.

============================================================
"""
    )


def main():

    print(
        """
============================================================
 Local LLM Long-Running Agent
============================================================

 Model : LM Studio Local LLM
 Memory: SQLite + Summary + FAISS
 Tools : Disabled

============================================================
"""
    )

    agent = LocalAgent()

    print(
        "Agent is ready."
    )

    print(
        "Type /help for commands."
    )

    print()

    while True:

        try:

            user_input = input(
                "You > "
            ).strip()

        except (
            KeyboardInterrupt,
            EOFError
        ):

            print(
                "\nExiting..."
            )

            break

        if not user_input:

            continue


        # ====================================================
        # Exit
        # ====================================================

        if user_input == "/exit":

            break


        # ====================================================
        # Help
        # ====================================================

        if user_input == "/help":

            print_help()

            continue


        # ====================================================
        # Statistics
        # ====================================================

        if user_input == "/stats":

            stats = (
                agent.memory_statistics()
            )

            print()
            print(
                "Memory statistics:"
            )

            print(
                f"Total memories : "
                f"{stats['total']}"
            )

            print(
                f"FAISS vectors  : "
                f"{stats['faiss_vectors']}"
            )

            print(
                f"Types          : "
                f"{stats['types']}"
            )

            print()

            continue


        # ====================================================
        # Search memory
        # ====================================================

        if user_input.startswith(
            "/memory "
        ):

            query = user_input[
                len("/memory "):
            ].strip()

            results = (
                agent.memory.search(
                    query=query,
                    top_k=5,
                    score_threshold=0.0
                )
            )

            print()

            if not results:

                print(
                    "No memory found."
                )

            else:

                for i, memory in enumerate(
                    results,
                    1
                ):

                    print(
                        f"[{i}] "
                        f"score="
                        f"{memory['score']:.3f}"
                    )

                    print(
                        f"type="
                        f"{memory['type']}"
                    )

                    print(
                        memory['text']
                    )

                    print()

            continue


        # ====================================================
        # Project memory
        # ====================================================

        if user_input.startswith(
            "/project "
        ):

            text = user_input[
                len("/project "):
            ].strip()

            agent.add_project_memory(
                text
            )

            print(
                "[Project memory saved]"
            )

            continue


        # ====================================================
        # Event memory
        # ====================================================

        if user_input.startswith(
            "/event "
        ):

            text = user_input[
                len("/event "):
            ].strip()

            agent.add_event_memory(
                text
            )

            print(
                "[Event memory saved]"
            )

            continue


        # ====================================================
        # Clear memory
        # ====================================================

        if user_input == "/clear":

            confirm = input(
                "Delete all long-term "
                "memory? (yes/no): "
            )

            if confirm.lower() == "yes":

                agent.memory.clear_memory()

                print(
                    "[Long-term memory cleared]"
                )

            continue


        # ====================================================
        # Normal conversation
        # ====================================================

        try:

            response = agent.chat(
                user_input
            )

            print()
            print(
                "Agent >"
            )

            print(
                response
            )

            print()

        except Exception as e:

            print()
            print(
                "[ERROR]"
            )

            print(
                str(e)
            )

            print()


if __name__ == "__main__":

    main()