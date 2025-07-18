Step	Action
1️⃣	You call router_agent_executor.invoke({"input": ...})
2️⃣	Router ReAct agent reads that input and selects a tool
3️⃣	The selected Tool(func=...) is called with the input string
4️⃣	That input string becomes original_prompt in your wrapper
5️⃣	Wrapper passes it down as {"input": original_prompt} to the sub-agent