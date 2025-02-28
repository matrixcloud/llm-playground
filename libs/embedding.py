from dashscope import BatchTextEmbedding
import os
import requests

class EmbeddingClient:
    def create(url: str, text_type="document"):
        res = BatchTextEmbedding.call(
            model=BatchTextEmbedding.Models.text_embedding_async_v1,
            api_key = os.getenv("DASHSCOPE_API_KEY"),
            url=url,
            text_type=text_type
        )
        data = requests.get(res.output.url).json()

        return data.output.embedding