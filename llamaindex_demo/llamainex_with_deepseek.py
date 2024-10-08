import os
import sys
import logging
from openai import OpenAI
from typing import Any, Generator
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from functools import cached_property

# 配置日志 创建一个与当前模块同名的 logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 从环境变量获取API密钥
load_dotenv()

API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not API_KEY:
    raise ValueError("DEEPSEEK_API_KEY environment variable is not set")


class DeepSeekChat(BaseModel):
    """DeepSeek聊天模型的封装类。"""

    api_key: str = Field(default=API_KEY)
    base_url: str = Field(default="https://api.deepseek.com")

    class Config:
        """Pydantic配置类。"""

        arbitrary_types_allowed = True  # 允许模型接受任意类型的字段
        # 这增加了灵活性，但可能降低类型安全性
        # 在本类中，这可能用于允许使用OpenAI客户端等复杂类型

    @cached_property
    def client(self) -> OpenAI:
        """创建并缓存OpenAI客户端实例。"""
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
        """
        使用DeepSeek API发送聊天请求。

        返回流式响应或完整响应内容。
        """
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
            logger.error(f"Error in DeepSeek API call: {e}")
            raise

    def _stream_response(self, response) -> Generator[str, None, None]:
        """处理流式响应，逐块生成内容。"""
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content


class DeepSeekLLM(CustomLLM):
    """DeepSeek语言模型的自定义实现。"""

    deep_seek_chat: DeepSeekChat = Field(default_factory=DeepSeekChat)

    @property
    def metadata(self) -> LLMMetadata:
        """返回LLM元数据。"""
        return LLMMetadata()

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """执行非流式完成请求。"""
        response = self.deep_seek_chat.chat(
            system_message="你是一个聪明的AI助手", user_message=prompt, stream=False
        )
        return CompletionResponse(text=response)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        """执行流式完成请求。"""
        response = self.deep_seek_chat.chat(
            system_message="你是一个聪明的AI助手", user_message=prompt, stream=True
        )

        def response_generator():
            """生成器函数，用于逐步生成响应内容。"""
            response_content = ""
            for chunk in self.deep_seek_chat._stream_response(response):
                if chunk:
                    response_content += chunk
                    yield CompletionResponse(text=response_content, delta=chunk)

        return response_generator()


# 设置环境变量，禁用tokenizers的并行处理，以避免潜在的死锁问题
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run_deepseek_query(query: str):
    # 从指定目录加载文档数据
    documents = SimpleDirectoryReader("data").load_data()

    # 设置LLM和嵌入模型
    Settings.llm = DeepSeekLLM()
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-zh-v1.5")

    # 创建索引和查询引擎
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine(streaming=True)

    # 执行查询
    print("DeepSeek 查询结果：")
    response = query_engine.query(query)

    # 处理并输出响应
    if hasattr(response, "response_gen"):
        # 流式输出
        for text in response.response_gen:
            print(text, end="", flush=True)
            sys.stdout.flush()  # 确保立即输出
    else:
        # 非流式输出
        print(response.response, end="", flush=True)

    print("\n查询完成")
