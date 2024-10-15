import os
import sys
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llamaindex_demo import logger
from llamaindex_demo.custom.custom_llm_glm_local import LocalGLM4


def run_glm4_query_with_local_glm_local_embedding(query: str):

    embed_model_path = "/home/nlp/model/Embedding/BAAI/bge-m3"
    pretrained_model_name_or_path = r"/home/nlp/model/LLM/THUDM/glm-4-9b-chat"

    # 设置LLM和嵌入模型
    Settings.llm = LocalGLM4(pretrained_model_name_or_path)
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=f"{embed_model_path}", device="cuda"
    )
    # Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-zh-v1.5")

    # 从指定目录加载文档数据
    documents = SimpleDirectoryReader(input_files=["./data/sample.txt"]).load_data()
    # 创建索引和查询引擎
    index = VectorStoreIndex.from_documents(documents)
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
