from zhipuai import ZhipuAI
import os

client = ZhipuAI(api_key=os.getenv("GLM_4_PLUS_API_KEY"))
response = client.embeddings.create(
    model="embedding-3",
    input=[
        "美食非常美味，服务员也很友好。",
        "这部电影既刺激又令人兴奋。",
        "阅读书籍是扩展知识的好方法。",
    ],
)
print(response)
