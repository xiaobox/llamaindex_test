import os
import sys
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llamaindex_demo.config import logger
from llamaindex_demo.custom_llm_deepseek import DeepSeekLLM

# 设置环境变量，禁用tokenizers的并行处理
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
    logger.info("DeepSeek 查询结果：")
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

    logger.info("\n查询完成")
