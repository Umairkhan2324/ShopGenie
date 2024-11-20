
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
load_dotenv()

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    api_key=GROQ_API_KEY,
    temperature=0.5,
)
print(llm.invoke("hi how are you?"))

