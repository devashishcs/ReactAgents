from dotenv import load_dotenv
import os
from typing import Any
from langchain import hub
from langchain_groq import ChatGroq
from langchain.agents import create_react_agent, AgentExecutor, Tool
from quickcommerce_tools import get_columns, group_by_sku_code, group_by_po_number, group_by_city_code, total_sales_by_sku, total_sales_by_po, plot_sales_by_sku, group_by_po_number

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
def main(query):
    llm = ChatGroq(api_key=GROQ_API_KEY, model="llama3-70b-8192")
    prompt = hub.pull("hwchase17/react")

    tools = [
        Tool(
        name="get_columns",
        func=get_columns,
        description="Returns name of coloumns across all sheets in the Excel file as JSON string.",
        return_direct=True  # ✅ This ensures the agent outputs only this result
    ),
    Tool(
        name="total_sales_by_sku",
        func=total_sales_by_sku,
        description="Returns total sales by SKU as JSON string.",
        return_direct=True  # ✅ This ensures the agent outputs only this result
    ),
    Tool(
        name="total_sales_by_po",
        func=total_sales_by_po,
        description="Returns total sales by PONumber as JSON string.",
        return_direct=True  # ✅ This ensures the agent outputs only this result
    )
]

    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    response = agent_executor.invoke({"input": query})
    return response["output"]

def chart(query):
    llm = ChatGroq(api_key=GROQ_API_KEY, model="llama3-70b-8192")
    prompt = hub.pull("hwchase17/react")

    tools = [
        Tool(
        name="get_columns",
        func=get_columns,
        description="Returns name of coloumns across all sheets in the Excel file as JSON string.",
        return_direct=True  # ✅ This ensures the agent outputs only this result
    ),
    Tool(
        name="total_sales_by_sku",
        func=total_sales_by_sku,
        description="Returns total sales by SKU as JSON string.",
        return_direct=True  # ✅ This ensures the agent outputs only this result
    ),
    Tool(
        name="total_sales_by_po",
        func=total_sales_by_po,
        description="Returns total sales by PONumber as JSON string.",
        return_direct=True  # ✅ This ensures the agent outputs only this result
    )
]

    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    response = agent_executor.invoke({"input": query})
    return response["output"]