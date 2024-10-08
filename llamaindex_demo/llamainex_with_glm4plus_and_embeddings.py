import os
import sys
import logging
from zhipuai import ZhipuAI
from typing import Any, List
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback
from dotenv import load_dotenv
from functools import cached_property
from pydantic import Field

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 从环境变量获取API密钥
load_dotenv()

API_KEY = os.getenv("GLM_4_PLUS_API_KEY")
if not API_KEY:
    raise ValueError("GLM_4_PLUS_API_KEY environment variable is not set")


class GLM4LLM(CustomLLM):
    @cached_property
    def client(self):
        return ZhipuAI(api_key=API_KEY)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata()

    def chat_with_glm4(self, system_message, user_message):
        response = self.client.chat.completions.create(
            model="glm-4-plus",
            messages=[
                {
                    "role": "system",
                    "content": system_message,
                },
                {
                    "role": "user",
                    "content": user_message,
                },
            ],
            stream=True,
        )
        return response

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        response = self.chat_with_glm4("你是一个聪明的AI助手", prompt)
        full_response = "".join(
            chunk.choices[0].delta.content
            for chunk in response
            if chunk.choices[0].delta.content
        )
        return CompletionResponse(text=full_response)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        response = self.chat_with_glm4("你是一个聪明的AI助手", prompt)

        def response_generator():
            response_content = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    response_content += chunk.choices[0].delta.content
                    yield CompletionResponse(
                        text=response_content, delta=chunk.choices[0].delta.content
                    )

        return response_generator()


class ZhipuEmbeddings(BaseEmbedding):
    client: ZhipuAI = Field(default_factory=lambda: ZhipuAI(api_key=API_KEY))

    def __init__(
        self,
        model_name: str = "embedding-3",
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, **kwargs)
        self._model = model_name

    def invoke_embedding(self, query: str) -> List[float]:
        response = self.client.embeddings.create(
            model=self._model, input=[query], dimensions=256
        )

        # 检查响应是否成功
        if response.data and len(response.data) > 0:
            return response.data[0].embedding
        else:
            raise ValueError("Failed to get embedding from ZhipuAI API")

    def _get_query_embedding(self, query: str) -> List[float]:
        return self.invoke_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self.invoke_embedding(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [self._get_text_embedding(text) for text in texts]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._get_text_embeddings(texts)


# 设置环境变量，禁用tokenizers的并行处理
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run_glm4_query_with_embeddings(query: str):
    # 从指定目录加载文档数据
    documents = SimpleDirectoryReader("data").load_data()

    # 设置LLM和嵌入模型
    Settings.llm = GLM4LLM()
    Settings.embed_model = ZhipuEmbeddings()

    # 创建索引和查询引擎
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine(streaming=True)

    # 执行查询
    print("GLM-4 查询结果：")
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
