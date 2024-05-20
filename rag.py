import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, api_key=openai_api_key)

loader = WebBaseLoader("https://www.books.com.tw/products/0010895449")

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)

embeddings_model = OpenAIEmbeddings(api_key=openai_api_key)

qdrant = Qdrant.from_documents(
    docs,
    embeddings_model,
    url="http://localhost:6333",
    force_recreate=True,
)

retriever = qdrant.as_retriever()

prompt = ChatPromptTemplate.from_template("""請依照 context 裡的資訊來回答問題:
<context>
{context}
</context>
Question: {input}""")

document_chain = create_stuff_documents_chain(model, prompt)

retrieval_chain = create_retrieval_chain(retriever, document_chain)

inp = {"input": "請問這本書的作者是？"}
response = retrieval_chain.invoke(inp)
print("Q:", inp["input"])
print("A:", response["answer"])

inp = {"input": "請問這本書的電子版價格？"}
response = retrieval_chain.invoke(inp)
print("\nQ:", inp["input"])
print("A:", response["answer"])

inp = {"input": "請簡介這本書？"}
response = retrieval_chain.invoke(inp)
print("\nQ:", inp["input"])
print("A:", response["answer"])
