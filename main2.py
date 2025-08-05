from dotenv import load_dotenv
import os
from langchain_core.messages import HumanMessage
# FIXED IMPORT: Use langgraph instead of langchain.graph
from langgraph.graph import MessagesState, StateGraph, END
from nodes import run_agent_reasoning, tool_node

load_dotenv()

# Constants
AGENT_REASON = "agent_reasoning"
ACT = "act"
LAST = -1 #Look at the most recent message

def should_continue(state: MessagesState) -> str:
    """Decide whether to continue with tools or end"""
    messages = state["messages"]
    if not messages or not hasattr(messages[LAST], 'tool_calls') or not messages[LAST].tool_calls:
        return END
    return ACT

# Create the flow
flow = StateGraph(MessagesState)

# Add nodes
flow.add_node(AGENT_REASON, run_agent_reasoning)
flow.add_node(ACT, tool_node)

# Set entry point
flow.set_entry_point(AGENT_REASON)

# Add edges
flow.add_conditional_edges(
    AGENT_REASON, 
    should_continue, 
    {END: END, ACT: ACT}
)
flow.add_edge(ACT, AGENT_REASON)

# Compile the app
app = flow.compile()

# FIXED METHOD NAME: draw_mermaid_png (not draw_mermain_png)
try:
    app.get_graph().draw_mermaid_png(output_file_path="flow.png")
    print("✅ Flow diagram saved as flow.png")
except Exception as e:
    print(f"⚠️ Could not save diagram: {e}")

if __name__ == "__main__":
    print("This is a simple tool demo.")
    
    # Example usage (uncomment to test):
    try:
        result = app.invoke({
            "messages": [HumanMessage(content="Whats the weather in tokyo?")]
        })
        #                 ↑
        #    This dictionary IS the initial state!
        print("Result:", result)
    except Exception as e:
        print(f"Error running app: {e}")