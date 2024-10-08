import os
import sys
import logging
from zhipuai import ZhipuAI
from typing import Any
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback
from dotenv import load_dotenv
from functools import cached_property
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

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


# 设置环境变量，禁用tokenizers的并行处理
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run_glm4_query(query: str):
    # 从指定目录加载文档数据
    documents = SimpleDirectoryReader("data").load_data()

    # 设置LLM和嵌入模型
    Settings.llm = GLM4LLM()
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-zh-v1.5")

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
