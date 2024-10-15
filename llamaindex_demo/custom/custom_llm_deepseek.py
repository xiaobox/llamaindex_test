import os
from openai import OpenAI
from typing import Any, Generator
from llama_index.core.llms import CustomLLM, CompletionResponse, CompletionResponseGen, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from pydantic import BaseModel, Field
from functools import cached_property
from dotenv import load_dotenv

# 从环境变量获取API密钥
load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise ValueError("DEEPSEEK_API_KEY environment variable is not set")

class DeepSeekChat(BaseModel):
    api_key: str = Field(default=DEEPSEEK_API_KEY)
    base_url: str = Field(default="https://api.deepseek.com")

    @cached_property
    def client(self) -> OpenAI:
        return OpenAI(api_key=self.api_key, base_url=self.base_url)

    def chat(
        self,
        system_message: str,
        user_message: str,
        model: str = "deepseek-chat",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> Any:
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=stream,
            )
            return response if stream else response.choices[0].message.content
        except Exception as e:
            raise

    def _stream_response(self, response) -> Generator[str, None, None]:
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

class DeepSeekLLM(CustomLLM):
    deep_seek_chat: DeepSeekChat = Field(default_factory=DeepSeekChat)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata()

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        response = self.deep_seek_chat.chat(
            system_message="你是一个聪明的AI助手", user_message=prompt, stream=False
        )
        return CompletionResponse(text=response)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        response = self.deep_seek_chat.chat(
            system_message="你是一个聪明的AI助手", user_message=prompt, stream=True
        )

        def response_generator():
            response_content = ""
            for chunk in self.deep_seek_chat._stream_response(response):
                if chunk:
                    response_content += chunk
                    yield CompletionResponse(text=response_content, delta=chunk)

        return response_generator()