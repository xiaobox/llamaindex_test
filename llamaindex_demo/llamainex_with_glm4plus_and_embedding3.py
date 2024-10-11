import os
import sys
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llamaindex_demo import logger
from llamaindex_demo.custom_llm_glm import GLM4LLM
from llamaindex_demo.custom_embedding_zhipu import ZhipuEmbeddings

# 设置环境变量，禁用tokenizers的并行处理
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run_glm4_query_with_embeddings(query: str):
    # 从指定目录加载文档数据
    documents = SimpleDirectoryReader("data").load_data()

    # 设置LLM和嵌入模型
    Settings.llm = GLM4LLM()
    Settings.embed_model = ZhipuEmbeddings()

    # 创建索引和查询引擎 show_progress=True 显示 embedding 进度
    index = VectorStoreIndex.from_documents(documents, show_progress=True)

    query_engine = index.as_query_engine(streaming=True)

    # 执行查询
    logger.info("GLM-4 查询结果：")
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
