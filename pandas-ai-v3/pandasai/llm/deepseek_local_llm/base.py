from __future__ import annotations
from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from pandasai.core.prompts.base import BasePrompt

from pandasai.llm.base import LLM
import requests

class DeepSeekLocalLLM(LLM):
    def __init__(self, model_name: str = "deepseek-r1:14b", host: str = "http://localhost:11434"):
        self.model_name = model_name
        self.host = host

    def call(self, instruction: BasePrompt, _context=None) -> str:
        payload = {
            "model": self.model_name,
            "prompt": instruction.to_string()
        }
        response = requests.post(f"{self.host}/api/generate", json=payload)
        response.raise_for_status()
        return response.json().get("response", "")

    @property
    def type(self) -> str:
        return "deepseek_local_llm"
