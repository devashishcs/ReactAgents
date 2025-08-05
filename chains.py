#holds prompts and tools for the agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
reflect_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a  viral linkedin influencer grading a post. Generate critique and recommendations."
        "Always provide detailed recommendations including length, virality style etc. "),   
    
        MessagesPlaceholder(variable_name="messages"),
    ]
)

generate_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a linkedin techie influencer assistant tasked with eriting excellent linkedin post."
        "Generate the best possible post for the user's request"
        "If the user provide critique, respond with a new post that incorporates the critique."),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

llm = ChatGroq(api_key=GROQ_API_KEY, model="llama3-70b-8192", temperature=0)

generate_chain = generate_prompt | llm
reflect_chain = reflect_prompt | llm
