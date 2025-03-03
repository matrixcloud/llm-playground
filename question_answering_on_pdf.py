from pypdf import PdfReader
import redis
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
import numpy as np
from dotenv import load_dotenv

from libs.client import AiClient
from libs.oss import OSSService

INDEX_NAME = "embeddings-index"
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_PWD = ""
PREFIX = "doc"

class DataService:
    def __init__(self, ai: AiClient):
        self.rd = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PWD
        )
        self.oss = OSSService()
        self.ai = ai

    def pdf_to_embeddings(self, pdf_path: str, chunk_length: int = 8000):
        reader = PdfReader(pdf_path)
        chunks = []

        for page in reader.pages:
            text_page = page.extract_text()
            chunks.extend([
                text_page[i:i+chunk_length].replace('\n', '')
                for i in range(0, len(text_page), chunk_length)
            ])

        print(f"embedding chunks length: {len(chunks)}")

        res = self.ai.batchCreateEmbeddings('text-embedding-v3', 10, chunks)
        # id: 块编号
        # text: 原始文本块
        # vector: 模型生成的嵌入
        embeddings = []
        for batched in res:
            for value in batched.embeddings:
                embeddings.append({
                    'id': f"{batched.batch_number}{value.index}",
                    'vector': value.embedding,
                    'text': chunks[batched.batch_number + value.index]
                })
        return embeddings

    def load_data_to_redis(self, embeddings):
        vector_dim = len(embeddings[0]['vector'])
        vector_number = len(embeddings)
        # Define RedisSearch fields
        text = TextField(name="text")
        text_embedding = VectorField("vector", "FLAT",
            {
                "TYPE": "FLOAT32",
                "DIM": vector_dim,
                "DISTANCE_METRIC": "COSINE",
                "INITIAL_CAP": vector_number
            })
        fields = [text, text_embedding]
        try:
            self.rd.ft(INDEX_NAME).info()
            print("Index already exists")
        except:
            self.rd.ft(INDEX_NAME).create_index(
                fields=fields,
                definition=IndexDefinition(prefix=[PREFIX], index_type=IndexType.HASH, language="chinese")
            )
            for embedding in embeddings:
                key = f"{PREFIX}:{str(embedding['id'])}"
                embedding["vector"] = np.array(embedding["vector"], dtype=np.float32).tobytes()
                self.rd.hset(key, mapping=embedding)
            print(f"Loaded {self.rd.info()['db0']['keys']} documents in Redis search index with name: {INDEX_NAME}")

    def drop_redis_data(self, index_name: str = INDEX_NAME):
        try:
            self.rd.ft(index_name).dropindex()
            print('Index dropped')
        except:
            # Index doees not exist
            print('Index does not exist')

    def search_redis(
            self,
            user_query: str,
            index_name: str = "embeddings-index",
            vector_field: str = "vector",
            return_fields: list = ["text", "vector_score"],
            hybrid_fields="*",
            k: int = 5,
            print_results: bool = False,
        ):
        # Creates embedding vector from user query
        embedded_query = self.ai.createEmbeddings(input=user_query, model='text-embedding-v3').data[0].embedding
        # Prepare the Query
        base_query = f'{hybrid_fields}=>[KNN {k} @{vector_field} $vector AS vector_score]'
        query = (
            Query(base_query)
            .return_fields(*return_fields)
            .sort_by("vector_score")
            .paging(0, k)
            .dialect(2)
        )
        params_dict = {"vector": np.array(embedded_query).astype(dtype=np.float32).tobytes()}
        # perform vector search
        results = self.rd.ft(index_name).search(query, params_dict)
        if print_results:
            for i, doc in enumerate(results.docs):
                score = 1 - float(doc.vector_score)
                print(f"{i}. {doc.text} (Score: {round(score, 3) })")
        return [doc['text'] for doc in results.docs]

class IntentService:
    def __init__(self, ai: AiClient):
        self.ai = ai

    def get_intent(self, question: str):
        print("Extracting keywords from the question")
        res = self.ai.ask([
            {"role": "user", "content": f"Extract the keywords from the following question: {question} and do not answer anything else, only the keywords"}
        ])
        return res

class ResponseService:
    def __init__(self, ai: AiClient):
        self.ai = ai

    def response(self, facts, question):
        res = self.ai.ask(
            [{"role": "user", "content": f"Based on the FACTS, give an answer to the QUESTION. QUESTION: {question}.FACTS: {facts}"}]
        )
        return res

def main():
    load_dotenv()
    ai = AiClient()
    data_svc = DataService(ai)
    intent_svc = IntentService(ai)
    response_svc = ResponseService(ai)

    data_svc.drop_redis_data()
    data = data_svc.pdf_to_embeddings('~/Downloads/塞尔达传说_旷野之息_中文版完全攻略本.pdf')
    data_svc.load_data_to_redis(data)
    question = '怎么打败风神兽?'
    intents = intent_svc.get_intent(question)
    print("intents: ", intents)
    facts = data_svc.search_redis(intents)
    print("facts: ", facts)
    answer = response_svc.response(facts, question)
    print(answer)

if __name__ == '__main__':
    main()
