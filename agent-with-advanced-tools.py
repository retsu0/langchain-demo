import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import tool, BaseTool
from typing import Optional, Union

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, api_key=openai_api_key)

@tool
def circle_area(radius) -> float:
    """計算圓形的面積時，使用這個工具。"""
    return float(radius) ** 2 * 3.14

desc = (
    "Use this tool when ou need to calculate the area of a triangle"
    "To use the tool, you must provide all of the follwing parameters "
    "{'a_side', 'b_side', 'c_side'}."
)

class TriangleTool(BaseTool):
    name = "Triangle area calculator"
    description = desc

    def _run(
        self,
        sides: Optional[Union[int, float, str]] = None,
    ):
        import ast
        parsed_sides = ast.literal_eval(sides)

        a_side = parsed_sides.get("a_side")
        b_side = parsed_sides.get("b_side")
        c_side = parsed_sides.get("c_side")

        if not all([a_side, b_side, c_side]):
            print(a_side, b_side, c_side)
            raise ValueError("You must provide all of the following parameters: {'a_side', 'b_side', 'c_side'}")
        if a_side <=0 or b_side <= 0 or c_side <= 0:
            raise ValueError("All sides must be greater than 0")
        if a_side + b_side <= c_side or a_side + c_side <= b_side or b_side + c_side <= a_side:
            raise ValueError("The sum of any two sides must be greater than the third side")
        s = (a_side + b_side + c_side) / 2
        area = (s * (s - a_side) * (s - b_side) * (s - c_side)) ** 0.5
        return area

tools = [circle_area, TriangleTool()]

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

inp = {"input": "請問半徑為 10 的圓形的面積是多少？"}
response = agent_executor.invoke(inp)
print("Q:", inp["input"])
print("A:", response["output"])
