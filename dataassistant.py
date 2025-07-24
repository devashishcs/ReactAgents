"""
Scenario 1: AI Data Analyst Assistant
Goal: A user uploads a CSV file and asks questions like:
"Which city had the highest average revenue last quarter, and can you plot it?"

Tool Examples:
load_csv(path)

calculate_average(column)

filter_by(column, condition)

plot_bar_chart(data)

Why Agents Needed:
It must plan:

Load the CSV

Filter for "last quarter"

Calculate average revenue

Pick the city with the max

Plot the result

This is a multi-step chain. An LLM alone can’t execute this without an agent coordinating tool calls.
"""

from dotenv import load_dotenv
import os
from typing import Any
from langchain import hub
from langchain_groq import ChatGroq
from langchain.agents import create_react_agent, AgentExecutor, Tool
from tools import get_columns

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
def main():
    llm = ChatGroq(api_key=GROQ_API_KEY, model="llama3-70b-8192")
    prompt = hub.pull("hwchase17/react")

    tools = [get_columns]  # ✅ Tool list is correct

    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # Example query
    query = "what are all the coloumns in the dataset?"
    response = agent_executor.invoke({"input": query})
    print(response["output"])

if __name__ == "__main__":
    main()
