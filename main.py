from dotenv import load_dotenv
import os
from typing import Any
from langchain import hub
from langchain_groq import ChatGroq
from langchain.agents import create_react_agent, AgentExecutor, Tool
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_experimental.tools import PythonREPLTool
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
def main():

    instructions = """You are an agent designed to write and execute python code to answer questions.
    You have access to a python REPL, which you can use to execute python code.
    If you get an error, debug your code and try again.
    Only use the output of your code to answer the question. 
    You might know the answer without running any code, but you should still run the code to get the answer.
    If it does not seem like you can write code to answer the question,
    just return "I don't know" as the answer.
    """


    # Load the base prompt template for the ReAct agent from LangChain Hub
    base_prompt = hub.pull("langchain-ai/react-agent-template")


    # Customize the prompt template with specific instructions for the agent
    prompt = base_prompt.partial(instructions=instructions)


    # Add the Python REPL tool so the agent can execute Python code
    tools = [PythonREPLTool()]


    #two seperate agents for different tasks
    python_agent = create_react_agent(
        prompt=prompt,
        llm = ChatGroq(api_key=GROQ_API_KEY, model="llama3-70b-8192", temperature=0),
        tools=tools)
    

    csv_agent = create_react_agent(
        prompt=prompt,
        llm = ChatGroq(api_key=GROQ_API_KEY, model="llama3-70b-8192", temperature=0),
        tools=tools)

    #comes prebuit pythonrepltool, own prompt csv agent already returns an executor, simple queries but for
    # more complex queries, you can use the create_react_agent function
    
    # csv_agent = create_csv_agent(
    #     llm=ChatGroq(api_key=GROQ_API_KEY, model="llama3-8b-8192", temperature=0),
    #     path="laptopData.csv",
    #     verbose=True,
    #     allow_dangerous_code=True
    # )
    
    # Create an AgentExecutor to manage the agent and its tools
    python_agent_executor = AgentExecutor(agent=python_agent, tools=tools, verbose=True)
    csv_agent_executor = AgentExecutor(agent=csv_agent, tools=tools, verbose=True)
    
    # Invoke the agent with a specific input to generate and save QR codes
#     python_agent_executor.invoke(
#         input={
#             "input": """generate and save in current working directory 15 QR codes under qrcodes folder
#                                 that point to www.google.com , you have qrcode package installed already"""
#         }
#     )
#     csv_agent_executor.invoke(
#     input={
#         "input": """
#         Load the file 'laptopData.csv' in the current working directory using pandas.
#         Perform the following cleaning steps step by step:
        
#         1. Import pandas and load the CSV file
#         2. Display basic information about the dataset (shape, columns, head)
#         3. Check for missing values in each column
#         4. Drop any rows with missing values
#         5. Strip leading/trailing whitespace from column names
#         6. Convert all column names to lowercase
#         7. Remove duplicate rows
#         8. Display the cleaned dataset info
#         9. Save the cleaned DataFrame as 'cleaned_data.csv' in the same directory
#         """
#     }
# )
     ################################ Router Grand Agent ########################################################
    
    def python_agent_executor_wrapper(original_prompt: str) -> dict[str, Any]:
        return python_agent_executor.invoke({"input": original_prompt})
    def csv_agent_executor_wrapper(original_prompt: str) -> dict[str, Any]:
        return csv_agent_executor.invoke({"input": original_prompt})


    router_tools = [
        Tool(
            name="python_agent",
            func=python_agent_executor_wrapper,
            description="""useful when you need to transform natural language to python and execute the python code,
                          returning the results of the code execution
                          DOES NOT ACCEPT CODE AS INPUT"""
        ),
        Tool(
            name="csv_agent",
            func=csv_agent_executor_wrapper,
            description="""useful when you need to answer question over laptopData.csv file,
                        takes an input the entire question and returns the answer after running pandas calculations"""
        )
    ]
    router_prompt = base_prompt.partial(instructions="")
    router_agent = create_react_agent(
        prompt=router_prompt,
        llm=ChatGroq(api_key=GROQ_API_KEY, model="llama3-8b-8192", temperature=0),
        tools=router_tools
    )
    #executor
    router_agent_executor = AgentExecutor(agent=router_agent, tools=router_tools, verbose=True)
    #based on description it will route the request to the appropriate agent
    print(
        router_agent_executor.invoke(
            {
                "input": "Give me the best laptop for gaming.",
            }
        )
    )

    print(
        router_agent_executor.invoke(
            {
                "input": "Generate and save in current working directory 15 qrcodes that point to `www.udemy.com/course/langchain`",
            }
        )
    )

if __name__ == "__main__":
    main()