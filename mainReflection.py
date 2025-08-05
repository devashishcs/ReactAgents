from dotenv import load_dotenv
import os
from typing import List, Sequence
load_dotenv()
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
from langgraph.graph import MessageGraph, END
from chains import generate_chain, reflect_chain

REFLECT = "reflect"
GENERATE = "generate"

def generation_node(state: Sequence[BaseMessage]):
    return generate_chain.invoke({"messages": state})

def reflection_node(messages: Sequence[BaseMessage])-> list[BaseMessage]:
    res = reflect_chain.invoke({"messages": messages})
    return [HumanMessage(content=res.content)]

builder = MessageGraph()
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)
def should_continue(state: List[BaseMessage]):
    if len(state) > 6:
        return END
    return REFLECT
builder.add_conditional_edges(GENERATE, should_continue, {END: END, REFLECT: REFLECT})
builder.add_edge(REFLECT, GENERATE)
graph = builder.compile()
print(graph.get_graph().draw_mermaid())
graph.get_graph().print_ascii()
if __name__ == "__main__":
    print("This is a simple tool demo.")
    input_messages = HumanMessage(content="""Make this linkedin post better"
                                  Hard work doesn’t beat luck.

We’ve all been told “hard work always pays off” — but is that really true in every case?

There are people who:
✅ Work 12+ hours a day
✅ Learn new skills constantly
✅ Apply to 100+ jobs without results

And then there are others who:
🍀 Get referred at the right time
🍀 Land the perfect project by chance
🍀 Go viral with one post or product

Does this mean hard work is overrated? Or does it simply increase your chances of getting lucky?

I believe success today is a mix of:
🔹 Preparation
🔹 Persistence
🔹 And a pinch of good timing (aka luck)

What’s your take?
👉 Does hard work guarantee success, or does luck still hold the bigger card?
 
hashtag#Hardwork hashtag#CareerGrowth hashtag#Linkedin hashtag#Mindset hashtag#AI""")
    response = graph.invoke(input_messages)