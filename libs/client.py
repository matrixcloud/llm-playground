import os
from typing import Iterable, List, Union
from openai import OpenAI
import math

class AiClient:
    def __init__(self):
        self.ai = OpenAI(
            # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
            api_key = os.getenv("DASHSCOPE_API_KEY"),  # 如何获取API Key：https://help.aliyun.com/zh/model-studio/developer-reference/get-api-key
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    def ask(self, messages: List[str]):
        res = self.ai.chat.completions.create(
            model="deepseek-r1",  # 此处以 deepseek-r1 为例，可按需更换模型名称。
            messages=messages
        )
        return res.choices[0].message.content

    def createEmbeddings(self, model: str, input: Union[str, List[str], Iterable[int], Iterable[Iterable[int]]]):
        return self.ai.embeddings.create(model=model, input=input)

    def batchCreateEmbeddings(self, model: str, batch_size: int, input: List[str]):
        batched_embeddings = []
        batch_number = 0

        for i in range(0, len(input), batch_size):
            print(f"create embeddings for batch: {batch_number}")
            print(len(input[i:i+batch_size]))
            embeddings = self.ai.embeddings.create(model=model, input=input[i:i + batch_size])
            batched_embeddings.append({'batch_number': batch_number, 'payload': embeddings})
            batch_number += 1

        return batched_embeddings
