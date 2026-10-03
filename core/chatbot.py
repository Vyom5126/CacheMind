from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile")

prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant.

STRICT INSTRUCTIONS:
* Use ONLY the provided context to answer the user's question.
* If the provided context is not sufficient, say that the context is not sufficient.

Context:
{context}

Question:
{question}

Answer:
""")

chatbot = prompt | llm | StrOutputParser()
