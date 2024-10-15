import os
import sys
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core import StorageContext, load_index_from_storage
from llamaindex_demo.custom.custom_llm_glm import GLM4LLM
from llamaindex_demo.custom.custom_embedding_zhipu import ZhipuEmbeddings

# 设置环境变量，禁用tokenizers的并行处理
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_or_create_index(is_create: bool = False):
    """
    获取或创建索引
    overwrite设置为 False 意味着如果同名的集合已存在，将不会覆盖它。
    dim 是向量维度，必须与 embedding 模型的维度一致。
    """
    vector_store = MilvusVectorStore(
        uri="http://localhost:19530",
        dim=256,
        overwrite=False,
        collection_name="llamaindex_collection",
    )

    if is_create:
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        documents = SimpleDirectoryReader("./data").load_data()
        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context
        )
        print("已成功创建并存储新的索引。")
    else:
        index = VectorStoreIndex.from_vector_store(vector_store)

    return index


def query_with_milvus(query: str, is_create: bool = False):
    # 设置LLM和嵌入模型
    Settings.llm = GLM4LLM()
    Settings.embed_model = ZhipuEmbeddings()

    # 加载或创建索引
    index = get_or_create_index(is_create)

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
