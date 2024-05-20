import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMMathChain
from langchain.agents import Tool, AgentExecutor, create_react_agent

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

model = ChatOpenAI(model="gpt-4o", temperature=0, api_key=openai_api_key)

chat_template = ChatPromptTemplate.from_messages([
    ("system", "你是熱心助人的 AI，同時也是地理高手。"),
    ("human", "你好！"),
    ("ai", "你好~"),
    ("human", "{user_input}"),
])

llm_math = LLMMathChain.from_llm(model)
math_tool = Tool(
    name="Calculator",
    func=llm_math.run,
    description="你是計算數學問題的好用工具，告訴我計算出來的結果。",
)

llm_chain = chat_template | model
geo_tool = Tool(
    name="Geography master",
    func=llm_chain.invoke,
    description="你是地理問題的好用工具，要告訴我國家的首都。",
)

tools = [math_tool, geo_tool]

prompt = PromptTemplate.from_template("""I want you to be FritzAgent. An agent that use tools to get answers. You are reliable and trustworthy. You follow the rules:

Rule 1: Answer the following questions as best as you can with the Observations presented to you.
Rule 2: Never use information outside of the Observations presented to you.
Rule 3: Never jump to conclusions unless the information for the final answer is explicitly presented to you in Observation.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
Thought: you should always think about what to do next. Use the Observation to gather extra information, but never use information outside of the Observation.
Action: the action to take, should be one of [{tool_names}]
Action_input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer.
Final Answer: the final answer to the original input question

Begin!

Question: {input}
{agent_scratchpad}
""")

zero_shot_agent = create_react_agent(llm=model, tools=tools, prompt=prompt)

agent_executor = AgentExecutor(agent=zero_shot_agent, tools=tools, verbose=True)

inp = {"input": "請問一個三邊為 3 4 5 的三角形的面積是多少？"}
response = agent_executor.invoke(inp)
print("Q:", inp["input"])
print("A:", response["output"])

inp = {"input": "請告訴我日本的首都是哪裡？"}
response = agent_executor.invoke(inp)
print("Q:", inp["input"])
print("A:", response["output"])

inp = {"input": "請告訴我「狗」的英文翻譯？"}
response = agent_executor.invoke(inp)
print("Q:", inp["input"])
print("A:", response["output"])
