from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import sys
from typing import Optional

load_dotenv()

# Add api directory to path to import openai_helper
api_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'api')
if api_dir not in sys.path:
    sys.path.insert(0, api_dir)

from openai_helper import DEFAULT_OPENAI_MODEL, create_openai_request, uses_responses_api


class ChatOpenAI:
    def __init__(self, api_key: str = None, provider: str = "openai"):
        """
        Initialize ChatOpenAI with support for multiple providers.
        
        Args:
            api_key: API key for the provider
            provider: Provider name ("openai" or "together")
        """
        self.provider = provider.lower()
        self.api_key = api_key #or os.getenv("OPENAI_API_KEY")
        
        if self.api_key is None:
            raise ValueError(f"{self.provider.title()} API key must be provided either as parameter or environment variable")
        
        # Set base URL based on provider
        if self.provider == "together":
            self.base_url = "https://api.together.xyz/v1"
        else:
            self.base_url = None  # Use default OpenAI base URL

    def run(self, messages, model_name: str = DEFAULT_OPENAI_MODEL, text_only: bool = True, **kwargs):
        if not isinstance(messages, list):
            raise ValueError("messages must be a list")

        # Use the shared helper: Responses API for supported OpenAI models, Chat Completions
        # API for others. Web search is opt-in via kwargs.
        response = create_openai_request(
            api_key=self.api_key,
            provider=self.provider,
            model=model_name,
            messages=messages,
            stream=False,
            **kwargs
        )

        if text_only:
            # Handle both Responses API (has output_text) and Chat Completions API (has choices)
            if hasattr(response, 'output_text'):
                return response.output_text
            else:
                return response.choices[0].message.content

        return response

    async def arun(self, messages, model_name: str = DEFAULT_OPENAI_MODEL, text_only: bool = True, **kwargs):
        """Async version of run method."""
        if not isinstance(messages, list):
            raise ValueError("messages must be a list")

        # Create OpenAI client
        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        client = AsyncOpenAI(**client_kwargs)

        # Pop web_search from kwargs so it's never forwarded raw to the API
        web_search = kwargs.pop("web_search", None)

        if uses_responses_api(self.provider, model_name):
            if len(messages) == 1:
                input_text = messages[0]["content"]
            else:
                # Format multi-turn conversation
                formatted_parts = []
                for msg in messages:
                    role = msg["role"]
                    content = msg["content"]
                    if role == "system":
                        formatted_parts.append(f"System: {content}")
                    elif role == "user":
                        formatted_parts.append(f"User: {content}")
                    elif role == "assistant":
                        formatted_parts.append(f"Assistant: {content}")
                input_text = "\n\n".join(formatted_parts)

            request_params = {
                "model": model_name,
                "input": input_text,
                **kwargs
            }

            # Only add web_search tool when explicitly enabled.
            if web_search:
                request_params["tools"] = [{"type": "web_search"}]

            response = await client.responses.create(**request_params)

            if text_only:
                return response.output_text
            return response
        else:
            # For Together.ai and other models, use Chat Completions API
            # (web_search already popped above, so it won't be forwarded)
            request_params = {
                "model": model_name,
                "messages": messages,
                **kwargs
            }

            response = await client.chat.completions.create(**request_params)

            if text_only:
                return response.choices[0].message.content
            return response
