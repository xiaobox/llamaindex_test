import time
from functools import wraps
from typing import Callable, Any
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llamaindex_demo import logger


def time_it(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{func.__name__} 耗时: {end_time - start_time:.2f} 秒")
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
    builder = IndexBuilder()
    builder.set_embed_model()
    builder.set_llm_model()
    documents = builder.load_documents()
    index = builder.create_index(documents)
    response = builder.perform_query(index, query)
    print("Ollama V2 查询结果：")
    print(response)
