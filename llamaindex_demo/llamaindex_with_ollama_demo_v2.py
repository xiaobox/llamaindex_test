import time, sys
from functools import wraps
from typing import Callable, Any
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama


def time_it(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} 耗时: {end_time - start_time:.2f} 秒")
        return result

    return wrapper


class IndexBuilder:
    @time_it
    def load_documents(self) -> list[Document]:
        return SimpleDirectoryReader("data").load_data()

    @time_it
    def set_embed_model(self) -> None:
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-zh-v1.5")

    @time_it
    def set_llm_model(self) -> None:
        Settings.llm = Ollama(model="qwen2:7b", request_timeout=360.0)

    @time_it
    def create_index(self, documents: list[Document]) -> VectorStoreIndex:
        return VectorStoreIndex.from_documents(documents)

    @time_it
    def perform_query(self, index: VectorStoreIndex, query: str) -> str:
        query_engine = index.as_query_engine()
        return query_engine.query(query)


def run_ollama_v2_query(query: str):
    # 从指定目录加载文档数据
    documents = SimpleDirectoryReader("data").load_data()

    # 设置LLM和嵌入模型
    Settings.llm = Ollama(model="qwen2:7b", request_timeout=360.0)
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-zh-v1.5")

    # 创建索引和查询引擎
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine(streaming=True)

    # 执行查询
    print("Ollama V2 查询结果：")
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
