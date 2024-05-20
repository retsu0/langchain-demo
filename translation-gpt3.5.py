import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, api_key=openai_api_key)

class Translation(BaseModel):
    lang: str = Field(description="language of the translation")
    text: str = Field(description="translated text")

parser = JsonOutputParser(pydantic_object=Translation)

prompt = PromptTemplate(
    template="請把後面的句子翻譯成{language}。{content}\n{format_instructions}\n",
    input_variables=["language", "content"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | model | parser
output = chain.invoke({"language":"法文", "content":"我愛你。"})

print(output)

# for item in chain.stream({"language":"法文", "content":"我愛你。"}):
#     print(item)

