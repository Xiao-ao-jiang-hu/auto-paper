from openai import OpenAI
from .config import (
    API_KEY,
    API_BASE_URL,
    MODEL_VISION,
    MODEL_REASONING,
    MODEL_INSTRUCTION,
)
import os


class LLMClients:
    def __init__(self):
        if not API_KEY:
            raise ValueError("API_KEY environment variable is not set")

        self.client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

    def get_vision_completion(self, base64_images: list, prompt: str):
        # Qwen-VL-Plus specific message format
        content = [{"type": "text", "text": prompt}]
        for start_idx, img_b64 in enumerate(base64_images):
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                }
            )

        response = self.client.chat.completions.create(
            model=MODEL_VISION,
            messages=[{"role": "user", "content": content}],
            stream=False,
        )

        message = response.choices[0].message
        if message.content is None:
            print(
                f"Warning: Model returned None content. Finish reason: {response.choices[0].finish_reason}"
            )
            # Identify if there is a refusal
            if hasattr(message, "refusal"):
                print(f"Refusal: {message.refusal}")
            return ""

        return message.content

    def get_reasoning_completion(self, system_prompt: str, user_content: str):
        # Kimi-k2-thinking usually just takes standard user/system prompts
        # but triggers internal CoT.
        response = self.client.chat.completions.create(
            model=MODEL_REASONING,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.3,  # Lower temp for logic
        )
        return response.choices[0].message.content

    def get_instruction_completion(self, system_prompt: str, user_content: str):
        response = self.client.chat.completions.create(
            model=MODEL_INSTRUCTION,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.1,
        )
        return response.choices[0].message.content
