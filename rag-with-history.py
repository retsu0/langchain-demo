import os
from dotenv import load_dotenv

import logging
logging.getLogger().setLevel(logging.ERROR) 

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

loader = WebBaseLoader("https://www.books.com.tw/products/0010895449")

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, api_key=openai_api_key)

embeddings_model = OpenAIEmbeddings(api_key=openai_api_key)

qdrant = Qdrant.from_documents(
    docs,
    embeddings_model,
    url="http://localhost:6333",
    collection_name="book",
    force_recreate=True,
)

retriever = qdrant.as_retriever()

prompt = ChatPromptTemplate.from_messages([
    ("system", "請依照 context 裡的資訊來回答問題:{context}。問題{input}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
    ])

document_chain = create_stuff_documents_chain(model, prompt)

retrieval_chain = create_retrieval_chain(retriever, document_chain)

chain_with_history = RunnableWithMessageHistory(
    retrieval_chain,
    lambda session_id: SQLChatMessageHistory(
        session_id="session_id", connection_string="sqlite:///langchain.db"
    ),
    input_messages_key="input",
    output_messages_key="answer",
    history_messages_key="history",
)

config = {"configurable": {"session_id": "session_id"}}

inp = {"input": "請問這本書的譯者是？"}
response = chain_with_history.invoke(inp, config=config)
print("Q:", inp["input"])
print("A:", response["answer"])

inp = {"input": "請問這本書的價格？"}
response = chain_with_history.invoke(inp, config=config)
print("\nQ:", inp["input"])
print("A:", response["answer"])

inp = {"input": "請問我問的第一個問題是？"}
response = chain_with_history.invoke(inp, config=config)
print("\nQ:", inp["input"])
print("A:", response["answer"])
