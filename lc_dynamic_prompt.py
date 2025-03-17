from langchain.prompts import PromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.output_parsers.string import StrOutputParser
import os

template = """Question: {question} Let's think step by step.
Answer: """
prompt = PromptTemplate(template=template, input_variables=["question"])

model = ChatTongyi(model="deepseek-v3", api_key=os.getenv("DASHSCOPE_API_KEY"))
chain = prompt | model | StrOutputParser()

question = """ What is the population of the capital of the country where the
Olympic Games were held in 2016? """

res = chain.invoke(question)
print(res)
