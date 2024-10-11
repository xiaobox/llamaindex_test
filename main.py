from llamaindex_demo.storage.llamainex_local_file import query_with_local_file_index

from llamaindex_demo.llamainex_with_glm4plus_and_embedding3 import (
    run_glm4_query_with_embeddings,
)
from llamaindex_demo.storage.llamainex_chromadb import query_with_chromadb
from llamaindex_demo.storage.llamainex_milvus import query_with_milvus
from llamaindex_demo.llamainex_with_glm4plus_only import run_glm4_query
from llamaindex_demo.llamainex_with_deepseek import run_deepseek_query
from llamaindex_demo.llamaindex_with_ollama_demo_v1 import run_ollama_v1_query
from llamaindex_demo.llamaindex_with_ollama_demo_v2 import run_ollama_v2_query


def main():
    query = "作者学习过的编程语言有哪些？"

    # print("使用 GLM-4 模型查询：")
    # run_glm4_query(query)

    # print("\n使用 GLM-4 模型和嵌入查询：")
    # run_glm4_query_with_embeddings(query)

    # print("\n使用本地文件存储 index：")
    # query_with_local_file_index(query)

    # print("\n使用 chromadb 存储 index：")
    # query_with_chromadb(query)

    print("\n使用 milvus 存储 index：")
    query_with_milvus(query, is_create=False)

    # print("\n使用 DeepSeek 模型查询：")
    # run_deepseek_query(query)

    # print("\n使用 Ollama 模型的第一版 demo 查询：")
    # run_ollama_v1_query(query)

    # print("\n使用 Ollama 模型的第二版 demo 查询：")
    # run_ollama_v2_query(query)


if __name__ == "__main__":
    main()
