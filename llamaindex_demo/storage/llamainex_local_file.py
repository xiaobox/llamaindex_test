import os
import sys
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core import StorageContext, load_index_from_storage
from llamaindex_demo.custom.custom_llm_glm import GLM4LLM
from llamaindex_demo.custom.custom_embedding_zhipu import ZhipuEmbeddings

# 设置环境变量，禁用tokenizers的并行处理
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_or_create_index():

    # 检查是否存在有效的持久化索引
    if (
        os.path.exists("index")
        and os.path.isdir("index")
        and any(file.endswith(".json") for file in os.listdir("index"))
    ):
        print("正在加载现有索引...")
        storage_context = StorageContext.from_defaults(persist_dir="index")
        index = load_index_from_storage(storage_context)
    else:
        print("未找到有效的现有索引，正在创建新索引...")
        # 使用预定义的 DATA_DIR 常量
        documents = SimpleDirectoryReader("./data").load_data()
        # 创建新索引，显示 embedding 进度
        index = VectorStoreIndex.from_documents(documents, show_progress=True)
        # 持久化索引
        index.storage_context.persist(persist_dir="index")
        print("索引已创建并保存到本地。")

    return index


def query_with_local_file_index(query: str, is_create: bool = False):
    # 设置LLM和嵌入模型
    Settings.llm = GLM4LLM()
    Settings.embed_model = ZhipuEmbeddings()

    # 加载或创建索引
    index = load_or_create_index()

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
