from dotenv import load_dotenv
import os
from typing import Any
from langchain import hub
from langchain_groq import ChatGroq
from langchain.agents import create_react_agent, AgentExecutor, Tool
from quickcommerce_tools import get_columns, group_by_sku_code, group_by_po_number, group_by_city_code, total_sales_by_sku, plot_sales_by_sku, group_by_po_number

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
def main():
    print("GROQ_API_KEY:", GROQ_API_KEY)
    llm = ChatGroq(api_key=GROQ_API_KEY, model="llama3-70b-8192")
    prompt = hub.pull("hwchase17/react")

    tools = [total_sales_by_sku, plot_sales_by_sku]  # ✅ Tool list is correct

    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # Example query
    query = "Group the dataset by sku and return the result in JSON format" \
    "Plot the graph showing  SKUs by sales."
    
    response = agent_executor.invoke({"input": query})
    print(response["output"])

if __name__ == "__main__":
    main()