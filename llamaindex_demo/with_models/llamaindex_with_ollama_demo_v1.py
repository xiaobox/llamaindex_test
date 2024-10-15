# 导入所需的库和模块
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import sys
from llamaindex_demo import logger


def run_ollama_v1_query(query: str):
    # 从指定目录加载文档数据
    documents = SimpleDirectoryReader("data").load_data()

    # 设置嵌入模型，使用北京智源人工智能研究院的中文嵌入模型
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-zh-v1.5")

    # 设置语言模型，使用Ollama提供的Qwen2 7B模型，并设置请求超时时间
    Settings.llm = Ollama(model="qwen2:7b", request_timeout=360.0)

    # 使用加载的文档创建向量存储索引
    index = VectorStoreIndex.from_documents(documents)

    # 从索引创建查询引擎，设置为流式模式
    query_engine = index.as_query_engine(streaming=True)

    # 使用查询引擎执行特定查询
    response_stream = query_engine.query(query)

    # 流式输出响应
    logger.info("Ollama V1 查询结果：")
    for text in response_stream.response_gen:
        print(text, end="", flush=True)
        sys.stdout.flush()  # 确保立即输出

    logger.info("\n查询完成")