from llm import LocalLLM
from memory import MemoryManager

from config import (
    RECENT_TURNS,
    TOP_K,
    MEMORY_SCORE_THRESHOLD,
    SUMMARY_INTERVAL,
    SUMMARY_SOURCE_TURNS,
)


class LocalAgent:

    def __init__(self):

        self.llm = LocalLLM()

        self.memory = MemoryManager()

        self.turn_count = 0


    # ========================================================
    # Retrieve long-term memory
    # ========================================================

    def retrieve_memory(
        self,
        query
    ):

        memories = self.memory.search(
            query=query,
            top_k=TOP_K,
            score_threshold=MEMORY_SCORE_THRESHOLD
        )

        if not memories:

            return (
                "No relevant long-term "
                "memory was found."
            )

        sections = []

        for i, memory in enumerate(
            memories,
            1
        ):

            memory_type = memory.get(
                "type",
                "unknown"
            )

            score = memory.get(
                "score",
                0
            )

            sections.append(
                f"""
[Memory {i}]
Type: {memory_type}
Similarity: {score:.3f}

{memory['text']}
"""
            )

        return "\n".join(
            sections
        )


    # ========================================================
    # Build LLM context
    # ========================================================

    def build_context(
        self,
        user_input
    ):

        # ----------------------------------------------------
        # Long-term memory
        # ----------------------------------------------------

        long_term_memory = (
            self.retrieve_memory(
                user_input
            )
        )

        # ----------------------------------------------------
        # Recent conversation
        # ----------------------------------------------------

        recent_messages = (
            self.memory.get_recent_messages(
                RECENT_TURNS
            )
        )

        # ----------------------------------------------------
        # System prompt
        # ----------------------------------------------------

        system_prompt = f"""
You are a long-running Local AI Agent.

You operate using a Local LLM through LM Studio.

Your memory system has two layers:

1. Recent conversation memory
2. Long-term semantic memory using FAISS

==================================================
LONG-TERM MEMORY
==================================================

{long_term_memory}

==================================================
END LONG-TERM MEMORY
==================================================

Rules:

1. Use long-term memory only when relevant.
2. Do not blindly trust retrieved memories.
3. If retrieved memory conflicts with the current
   user message, prioritize the current message.
4. Maintain continuity with previous conversations.
5. Do not mention the internal memory system unless
   the user asks about it.
6. Answer in Korean unless the user requests another
   language.
"""

        messages = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]

        # ----------------------------------------------------
        # Recent conversation
        # ----------------------------------------------------

        for message in recent_messages:

            messages.append(
                message
            )

        # ----------------------------------------------------
        # Current user input
        # ----------------------------------------------------

        messages.append(
            {
                "role": "user",
                "content": user_input
            }
        )

        return messages


    # ========================================================
    # Chat
    # ========================================================

    def chat(
        self,
        user_input
    ):

        # ----------------------------------------------------
        # Save user input
        # ----------------------------------------------------

        self.memory.save_message(
            "user",
            user_input
        )

        # ----------------------------------------------------
        # Build context
        # ----------------------------------------------------

        messages = self.build_context(
            user_input
        )

        # ----------------------------------------------------
        # Local LLM
        # ----------------------------------------------------

        response = self.llm.chat(
            messages=messages,
            temperature=0.2,
            max_tokens=2048
        )

        # ----------------------------------------------------
        # Save response
        # ----------------------------------------------------

        self.memory.save_message(
            "assistant",
            response
        )

        self.turn_count += 1

        # ----------------------------------------------------
        # Memory consolidation
        # ----------------------------------------------------

        if (
            self.turn_count
            % SUMMARY_INTERVAL
            == 0
        ):

            self.consolidate_memory()

        return response


    # ========================================================
    # Memory consolidation
    # ========================================================

    def consolidate_memory(self):

        recent_messages = (
            self.memory.get_recent_messages(
                SUMMARY_SOURCE_TURNS
            )
        )

        if not recent_messages:

            return

        conversation_text = "\n".join(
            f"{message['role']}: "
            f"{message['content']}"
            for message in recent_messages
        )

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        summary_prompt = f"""
Analyze the following conversation and extract
information worth remembering for future sessions.

Conversation:

{conversation_text}

Create a concise long-term memory.

Focus on:

- user's goals
- ongoing projects
- technical environment
- important decisions
- important technical findings
- user preferences
- unresolved problems
- future plans

Do not include:

- greetings
- small talk
- temporary details
- redundant information

Return only the memory content.
"""

        summary = self.llm.chat(
            messages=[
                {
                    "role": "system",
                    "content":
                        "You are a long-term memory "
                        "consolidation system."
                },
                {
                    "role": "user",
                    "content":
                        summary_prompt
                }
            ],
            temperature=0.0,
            max_tokens=1000
        )

        if summary.strip():

            self.memory.add_memory(
                text=summary,
                memory_type="summary"
            )

            print(
                "\n[Long-term memory updated]"
            )


    # ========================================================
    # Explicit project memory
    # ========================================================

    def add_project_memory(
        self,
        text
    ):

        self.memory.add_memory(
            text=text,
            memory_type="project"
        )


    # ========================================================
    # Explicit event memory
    # ========================================================

    def add_event_memory(
        self,
        text
    ):

        self.memory.add_memory(
            text=text,
            memory_type="event"
        )


    # ========================================================
    # Statistics
    # ========================================================

    def memory_statistics(self):

        return self.memory.statistics()