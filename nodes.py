from dotenv import load_dotenv
import os
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode

from react import llm, tools
load_dotenv()


SYSTEM_MESSAGE = """You are a helpful assistant that can use tools to answer the question."""

def run_agent_reasoning(state: MessagesState) -> MessagesState:
    """Run the agent reasoning step."""
    response = llm.invoke([
        {"role": "system", "content": SYSTEM_MESSAGE}
    ] + state["messages"])  # ✅ Use + to combine lists properly
    return {"messages": [response]}

tool_node = ToolNode(tools)
