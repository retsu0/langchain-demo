from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

model = Ollama(model="llama3")

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
output = chain.invoke({"language":"French", "content":"我愛你。"})

print(output)
