import os
import sys
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.core.tools import QueryEngineTool
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llamaindex_demo import logger
from llamaindex_demo.custom.custom_llm_glm import GLM4LLM
from llamaindex_demo.custom.custom_embedding_zhipu import ZhipuEmbeddings
from llama_parse import LlamaParse

# 设置环境变量，禁用tokenizers的并行处理
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def toHtml(text: str) -> str:
    """遇到英文单词就用html标签包裹"""
    return re.sub(r"(\b[a-zA-Z]+\b)", r'<span style="color:red;">\1</span>', text)


def run_glm4_query_with_embeddings():
    # 从指定目录加载文档数据
    
    documents = SimpleDirectoryReader(
        input_files=["./data/sample.txt"]
    ).load_data()


    # 设置LLM和嵌入模型
    Settings.llm = GLM4LLM()
    Settings.embed_model = ZhipuEmbeddings()

    # 创建索引和查询引擎 show_progress=True 显示 embedding 进度
    index = VectorStoreIndex.from_documents(documents, show_progress=True)

    query_engine = index.as_query_engine(streaming=True)

    yc_tool = QueryEngineTool.from_defaults(
        query_engine,
        #name="YC 创始人的个人经理",
        #description="关于YC创始人Paul Graham的RAG引擎",
    )

    to_html_tool = FunctionTool.from_defaults(fn=toHtml)

    agent = ReActAgent.from_tools(
        [to_html_tool,yc_tool],
        verbose=True,
    )

    # 执行查询
    logger.info("agent 查询结果：")
    response = agent.chat("请描述一下作者的求学经历，并将英文用html高亮显示")

    print(response)
    
    response = agent.chat("在这些经历中哪些是积极快乐的")
    print(response)
    
    response = agent.chat("哪些是不快乐的？")
    print(response)
    

    logger.info("\n查询完成")


def main():
    run_glm4_query_with_embeddings()


if __name__ == "__main__":
    main()
