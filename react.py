##This is an reasoning engine for executing multi-step tasks using tools and agents.

from dotenv import load_dotenv

import os
from langchain_core.tools import tool
from langchain_groq import ChatGroq

load_dotenv()

@tool
def triple(x: int) -> int:
    """Triple the input value."""
    return x * 3

@tool
def square(x: int) -> int:
    """Square the input value."""
    return x * x
tools = [triple, square]

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(api_key=GROQ_API_KEY, model="llama3-70b-8192", temperature=0).bind_tools(tools)
