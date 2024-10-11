import unittest
from llamaindex_demo.query.llamainex_query_retriever import query_with_custom_query
from llamaindex_demo.storage.llamainex_milvus import query_with_milvus
from llamaindex_demo.storage.llamainex_local_file import query_with_local_file_index
from llamaindex_demo.storage.llamainex_chromadb import query_with_chromadb
from llamaindex_demo.llamainex_with_glm4plus_and_embedding3 import (
    run_glm4_query_with_embeddings,
)
from llamaindex_demo.llamainex_with_glm4plus_only import run_glm4_query
from llamaindex_demo.llamainex_with_deepseek import run_deepseek_query
from llamaindex_demo.llamaindex_with_ollama_demo_v1 import run_ollama_v1_query
from llamaindex_demo.llamaindex_with_ollama_demo_v2 import run_ollama_v2_query
from llamaindex_demo.config import logger


class TestLlamaIndex(unittest.TestCase):
    def setUp(self):
        self.query = "作者会画画吗？"

    def test_custom_query(self):
        logger.info("\n使用自定义 query 查询：")
        query_with_custom_query(self.query, is_create=False)

    def test_milvus_storage(self):
        logger.info("\n使用 milvus 存储 index：")
        query_with_milvus(self.query, is_create=False)

    def test_local_file_storage(self):
        logger.info("\n使用本地文件存储 index：")
        query_with_local_file_index(self.query)

    def test_chromadb_storage(self):
        logger.info("\n使用 chromadb 存储 index：")
        query_with_chromadb(self.query)

    def test_glm4_with_embedding3(self):
        logger.info("\n使用 GLM-4 模型和 Embedding3 模型查询：")
        run_glm4_query_with_embeddings(self.query)

    def test_glm4_only(self):
        logger.info("\n只使用 GLM-4 模型查询：")
        run_glm4_query(self.query)

    def test_deepseek(self):
        logger.info("\n使用 DeepSeek 模型查询：")
        run_deepseek_query(self.query)

    def test_ollama_v1(self):
        logger.info("\n使用 Ollama 模型的第一版 demo 查询：")
        run_ollama_v1_query(self.query)

    def test_ollama_v2(self):
        logger.info("\n使用 Ollama 模型的第二版 demo 查询：")
        run_ollama_v2_query(self.query)


def run_selected_tests(test_names):
    suite = unittest.TestSuite()
    for test_name in test_names:
        suite.addTest(TestLlamaIndex(test_name))
    runner = unittest.TextTestRunner()
    runner.run(suite)


def main():

    # 组合执行多个测试用例,也可只执行一个
    selected_tests = [
        "test_milvus_storage",
        #'test_local_file_storage'
    ]
    run_selected_tests(selected_tests)


if __name__ == "__main__":
    main()
