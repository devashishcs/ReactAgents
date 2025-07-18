Step	Action
1️⃣	You call router_agent_executor.invoke({"input": ...})
2️⃣	Router ReAct agent reads that input and selects a tool
3️⃣	The selected Tool(func=...) is called with the input string
4️⃣	That input string becomes original_prompt in your wrapper
5️⃣	Wrapper passes it down as {"input": original_prompt} to the sub-agent



when the task is clear and simple, you can use the bind_tools method to bind the tools to the LLM
and then invoke the LLM with a question that can be answered using the tools
agents are more powerful when the task is complex and requires multiple steps or reasoning
in this case, you can use the create_react_agent function to create an agent that
can use the tools to answer the question
