from llama_index.core.embeddings import BaseEmbedding
from zhipuai import ZhipuAI
from typing import List, Any
from pydantic import Field
from .config import ZHIPU_API_KEY

class ZhipuEmbeddings(BaseEmbedding):
    client: ZhipuAI = Field(default_factory=lambda: ZhipuAI(api_key=ZHIPU_API_KEY))

    def __init__(
        self,
        model_name: str = "embedding-3",
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, **kwargs)
        self._model = model_name

    def invoke_embedding(self, query: str) -> List[float]:
        response = self.client.embeddings.create(
            model=self._model, input=[query], dimensions=256
        )

        # 检查响应是否成功
        if response.data and len(response.data) > 0:
            return response.data[0].embedding
        else:
            raise ValueError("Failed to get embedding from ZhipuAI API")

    def _get_query_embedding(self, query: str) -> List[float]:
        return self.invoke_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self.invoke_embedding(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [self._get_text_embedding(text) for text in texts]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._get_text_embeddings(texts)