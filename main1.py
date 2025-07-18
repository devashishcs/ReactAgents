from langchain_core.tools import tool
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
#tool defined
@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b."""
    return a * b
@tool
def add(a: int, b: int) -> int:
    """add a and b."""
    return a + b



def main1():
    llm = ChatGroq(api_key=GROQ_API_KEY, model="llama3-70b-8192", temperature=0)
    print("This is a simple tool demo.")
    

    llm_with_tools = llm.bind_tools([multiply, add])
    result = llm_with_tools.invoke("What is 2 plus 3?")
    print(result)
if __name__ == "__main__":
    main1()

