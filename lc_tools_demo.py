from langchain.agents import load_tools, create_react_agent, AgentExecutor
from langchain import hub
from langchain_community.chat_models.tongyi import ChatTongyi
import os

llm = ChatTongyi(model="deepseek-v3", api_key=os.getenv("DASHSCOPE_API_KEY"))

tools = load_tools(["wikipedia", "llm-math"], llm=llm)
agent = create_react_agent(
    tools=tools,
    llm=llm,
    prompt=hub.pull("hwchase17/react"),
)

question = """What is the square root of the population of the country that won the 2023 Rugby World Cup?
"""
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
res = agent_executor.invoke({"input": question})
print(res)
