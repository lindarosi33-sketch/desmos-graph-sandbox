# graph_brain.py
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)


class GraphBrain:
    def __init__(self, model_path):
        self.model_path = model_path
        self.api_base = "http://localhost:8003/v1"
        self.client = OpenAI(base_url=self.api_base, api_key="dummy", timeout=120)

    def load(self):
        """Connect to model server with retry (waits up to 60s for systemd startup)."""
        import time
        for attempt in range(12):
            try:
                models = self.client.models.list()
                logger.info("Connected to model server at %s (model: %s)", self.api_base, self.model_path)
                return True
            except Exception as e:
                if attempt < 11:
                    logger.warning("Model server not ready (attempt %d/12): %s", attempt + 1, e)
                    time.sleep(5)
                else:
                    logger.error("Failed to connect to model server after 12 attempts: %s", e)
                    return False

    def chat(self, messages, sampling_params=None, internet_allowed=False, tools=None):
        """
        messages: list of dicts [{"role": "user", "content": "..."}, ...]
        tools: optional list of function calling tool schemas — sent to the API
        returns:
            - str if tools is None (backwards compatible)
            - dict {"content": str|None, "tool_calls": list|None} if tools is provided
        """
        kwargs = {
            "model": self.model_path,
            "messages": messages,
            "temperature": 1.0,
            "top_p": 1.0,
            "presence_penalty": 2.0,
            "max_tokens": 32768,
        }
        if sampling_params:
            kwargs["temperature"] = sampling_params.temperature
            kwargs["max_tokens"] = sampling_params.max_tokens
        if tools:
            kwargs["tools"] = tools
            kwargs["extra_body"] = {"top_k": 40, "min_p": 0.0}

        response = self.client.chat.completions.create(**kwargs)
        msg = response.choices[0].message

        usage = {}
        if hasattr(response, 'usage') and response.usage:
            usage = {"prompt_tokens": response.usage.prompt_tokens, "completion_tokens": response.usage.completion_tokens}
            logger.info("Token usage: prompt=%s completion=%s", response.usage.prompt_tokens, response.usage.completion_tokens)

        raw = msg.content or ""
        if hasattr(msg, 'reasoning_content') and msg.reasoning_content:
            content = f"<think>\n{msg.reasoning_content}</think>\n\n{raw}"
        else:
            content = raw
        if tools:
            return {
                "content": content,
                "tool_calls": [tc.model_dump() for tc in msg.tool_calls] if msg.tool_calls else None,
                **usage,
            }
        return {"content": content, **usage}
