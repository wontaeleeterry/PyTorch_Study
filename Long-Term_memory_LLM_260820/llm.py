from openai import OpenAI

from config import (
    LM_STUDIO_BASE_URL,
    LM_STUDIO_API_KEY,
    MODEL_NAME,
)


class LocalLLM:

    def __init__(self):

        self.client = OpenAI(
            base_url=LM_STUDIO_BASE_URL,
            api_key=LM_STUDIO_API_KEY,
        )

    # ========================================================
    # Chat
    # ========================================================

    def chat(
        self,
        messages,
        temperature=0.2,
        max_tokens=2048,
    ):

        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return response.choices[0].message.content.strip()