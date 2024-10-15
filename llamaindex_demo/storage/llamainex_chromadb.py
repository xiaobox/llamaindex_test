import os
import sys
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core import StorageContext
from llamaindex_demo import logger
from llamaindex_demo.custom.custom_llm_glm import GLM4LLM
from llamaindex_demo.custom.custom_embedding_zhipu import ZhipuEmbeddings

# 设置环境变量,禁用tokenizers的并行处理
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_or_create_index():
    # 初始化客户端,设置数据保存路径
    db = chromadb.PersistentClient(path="./chroma_db")
    # 创建或获取集合
    chroma_collection = db.get_or_create_collection("quickstart")
    # 将 chroma 指定为上下文的 vector_store
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 检查集合是否为空
    if chroma_collection.count() == 0:
        # 如果集合为空,加载文档并创建新的索引
        documents = SimpleDirectoryReader("./data").load_data()
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        logger.info("已创建新的索引")
    else:
        # 如果集合不为空,直接从vector_store加载索引
        index = VectorStoreIndex.from_vector_store(vector_store)
        logger.info("已加载现有索引")

    return index

def query_with_chromadb(query: str):
    # 设置LLM和嵌入模型
    Settings.llm = GLM4LLM()
    Settings.embed_model = ZhipuEmbeddings()

    # 加载或创建索引
    index = load_or_create_index()

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
