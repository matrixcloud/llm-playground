from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.llms.tongyi import Tongyi
import os

loader = PyPDFLoader("/Users/atomliu/Downloads/塞尔达传说_旷野之息_中文版完全攻略本.pdf")
pages = loader.load_and_split()

embeddings = DashScopeEmbeddings(
    model="text-embedding-v2",
    # other params...
)

db = FAISS.from_documents(pages, embeddings)
q = "What is Link's traditional outfit color?"
print(db.similarity_search(q))

llm = Tongyi(model="deepseek-v3", api_key=os.getenv("DASHSCOPE_API_KEY"))
chain = RetrievalQA.from_llm(llm=llm, retriever=db.as_retriever())
q = "What is Link's traditional outfit color?"
print(chain.invoke(q))
