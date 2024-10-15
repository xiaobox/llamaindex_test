import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from llama_index.core import VectorStoreIndex, Settings
from llamaindex_demo import logger
from llamaindex_demo.custom.custom_llm_glm import GLM4LLM
from llamaindex_demo.custom.custom_embedding_zhipu import ZhipuEmbeddings
from llama_parse import LlamaParse

# 设置环境变量，禁用tokenizers的并行处理
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run_glm4_query_with_embeddings(query: str):

    documents2 = LlamaParse(result_type="markdown").load_data(
        "./data/test.pdf"
    )

     # 设置LLM和嵌入模型
    Settings.llm = GLM4LLM()
    Settings.embed_model = ZhipuEmbeddings()
    
    index2 = VectorStoreIndex.from_documents(documents2)
    query_engine2 = index2.as_query_engine()

    response2 = query_engine2.query(query)

    print(response2)

    logger.info("\n查询完成")


def main():
    run_glm4_query_with_embeddings("请描述一下作者的求学经历")


if __name__ == "__main__":
    main()
