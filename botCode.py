import re
import json
from typing import TypedDict, List, Optional, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI  # or your preferred LLM
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
import asyncio

# State definition
class ChatbotState(TypedDict):
    messages: List[Dict[str, str]]
    user_age: Optional[int]
    insurance_type: Optional[str]
    user_query: str
    relevant_docs: List[Dict[str, Any]]
    missing_info: List[str]
    conversation_stage: str
    last_response: str

# Insurance Database (same as before)
INSURANCE_DATABASE = {
    "health": {
        "18-25": {"premium": "$150-200/month", "coverage": "Basic health coverage", "deductible": "$1,500"},
        "26-35": {"premium": "$180-250/month", "coverage": "Comprehensive health coverage", "deductible": "$1,200"},
        "36-50": {"premium": "$220-300/month", "coverage": "Enhanced coverage", "deductible": "$1,000"},
        "51-65": {"premium": "$280-400/month", "coverage": "Premium coverage", "deductible": "$800"}
    },
    "life": {
        "18-30": {"premium": "$20-40/month", "coverage": "$250,000-500,000", "type": "Term life"},
        "31-45": {"premium": "$35-70/month", "coverage": "$300,000-750,000", "type": "Term/Whole life"},
        "46-60": {"premium": "$80-150/month", "coverage": "$200,000-500,000", "type": "Whole/Universal life"}
    },
    "auto": {
        "18-25": {"premium": "$120-200/month", "coverage": "Liability + Comprehensive", "deductible": "$500-1000"},
        "26-40": {"premium": "$90-150/month", "coverage": "Full coverage", "deductible": "$250-750"},
        "41-65": {"premium": "$80-130/month", "coverage": "Mature driver rates", "deductible": "$250-500"}
    }
}

class InsuranceChatbotWithLLM:
    def __init__(self, llm_api_key: str = None):
        # Initialize LLM - replace with your preferred model
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo", 
            api_key=llm_api_key,
            temperature=0.7
        )
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow with LLM integration"""
        graph = StateGraph(ChatbotState)
        
        # Add nodes - now using LLMs
        graph.add_node("extract_info_with_llm", self._extract_info_with_llm)
        graph.add_node("ask_followup_with_llm", self._ask_followup_with_llm)
        graph.add_node("search_documents", self._search_documents)
        graph.add_node("generate_response_with_llm", self._generate_response_with_llm)
        
        # Define edges
        graph.add_conditional_edges(
            "extract_info_with_llm",
            self._should_collect_info,
            {
                "collect_info": "ask_followup_with_llm",
                "search_docs": "search_documents"
            }
        )
        
        graph.add_edge("ask_followup_with_llm", END)
        graph.add_edge("search_documents", "generate_response_with_llm")
        graph.add_edge("generate_response_with_llm", END)
        
        # Set entry point
        graph.set_entry_point("extract_info_with_llm")
        
        return graph.compile()
    
    async def _extract_info_with_llm(self, state: ChatbotState) -> ChatbotState:
        """Use LLM to extract age and insurance type from user message"""
        user_message = state["user_query"]
        
        # LLM Prompt for information extraction
        extraction_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an information extraction assistant. 
            Extract the user's age and insurance type from their message.
            
            Return ONLY a JSON object with this format:
            {"age": number_or_null, "insurance_type": "health/life/auto_or_null"}
            
            Examples:
            - "I'm 25 and need health insurance" -> {"age": 25, "insurance_type": "health"}
            - "Looking for car insurance" -> {"age": null, "insurance_type": "auto"}
            - "I need insurance" -> {"age": null, "insurance_type": null}"""),
            HumanMessage(content=f"Extract from: {user_message}")
        ])
        
        # Get LLM response
        llm_response = await self.llm.ainvoke(extraction_prompt.format_messages())
        
        try:
            # Parse JSON response
            extracted_data = json.loads(llm_response.content)
            
            if extracted_data.get("age"):
                state["user_age"] = extracted_data["age"]
            if extracted_data.get("insurance_type"):
                state["insurance_type"] = extracted_data["insurance_type"]
                
        except json.JSONDecodeError:
            # Fallback to regex if LLM fails
            age_match = re.search(r'\b(\d{1,2})\b', user_message)
            if age_match and 18 <= int(age_match.group(1)) <= 100:
                state["user_age"] = int(age_match.group(1))
        
        return state
    
    def _should_collect_info(self, state: ChatbotState) -> str:
        """Decide next step based on available information"""
        missing_info = []
        
        if not state.get("user_age"):
            missing_info.append("age")
        if not state.get("insurance_type"):
            missing_info.append("insurance_type")
        
        state["missing_info"] = missing_info
        
        if missing_info:
            return "collect_info"
        else:
            return "search_docs"
    
    async def _ask_followup_with_llm(self, state: ChatbotState) -> ChatbotState:
        """Use LLM to generate natural follow-up questions"""
        missing_info = state.get("missing_info", [])
        conversation_history = state.get("messages", [])
        
        # Create context for LLM
        history_text = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in conversation_history[-3:]  # Last 3 messages
        ])
        
        # LLM Prompt for generating follow-up questions
        followup_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a friendly insurance assistant. 
            Generate natural follow-up questions to collect missing information.
            
            Be conversational, helpful, and specific. Don't be robotic.
            
            Available insurance types: health, life, auto insurance"""),
            HumanMessage(content=f"""
            Conversation so far:
            {history_text}
            
            Missing information: {', '.join(missing_info)}
            
            Generate a helpful follow-up question to collect the missing information.
            """)
        ])
        
        # Get LLM response
        llm_response = await self.llm.ainvoke(followup_prompt.format_messages())
        response = llm_response.content
        
        state["last_response"] = response
        state["messages"].append({"role": "assistant", "content": response})
        return state
    
    def _search_documents(self, state: ChatbotState) -> ChatbotState:
        """Search insurance documents (same as before - no LLM needed here)"""
        age = state["user_age"]
        insurance_type = state["insurance_type"]
        
        age_bracket = self._get_age_bracket(age, insurance_type)
        
        relevant_docs = []
        if insurance_type in INSURANCE_DATABASE and age_bracket in INSURANCE_DATABASE[insurance_type]:
            doc = INSURANCE_DATABASE[insurance_type][age_bracket]
            relevant_docs.append({
                "insurance_type": insurance_type,
                "age_bracket": age_bracket,
                "data": doc
            })
        
        state["relevant_docs"] = relevant_docs
        return state
    
    async def _generate_response_with_llm(self, state: ChatbotState) -> ChatbotState:
        """Use LLM to generate final response with insurance information"""
        relevant_docs = state["relevant_docs"]
        user_age = state["user_age"]
        insurance_type = state["insurance_type"]
        
        if not relevant_docs:
            # LLM generates "no results" response
            no_results_prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="You are a helpful insurance assistant. Generate a polite response when no insurance options are found."),
                HumanMessage(content=f"No insurance options found for {insurance_type} insurance for age {user_age}. Suggest contacting support.")
            ])
            llm_response = await self.llm.ainvoke(no_results_prompt.format_messages())
            response = llm_response.content
        else:
            # LLM generates response with insurance data
            doc_data = relevant_docs[0]["data"]
            
            insurance_info = f"""
            Insurance Type: {insurance_type.title()}
            Age: {user_age}
            Premium: {doc_data.get('premium', 'Contact for quote')}
            Coverage: {doc_data.get('coverage', 'Standard coverage')}
            Deductible: {doc_data.get('deductible', 'N/A')}
            """
            
            response_prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="""You are a helpful insurance assistant. 
                Present insurance information in a friendly, well-formatted way using emojis and clear structure.
                Be enthusiastic but professional."""),
                HumanMessage(content=f"""
                Present this insurance information to the user:
                {insurance_info}
                
                Make it engaging and ask if they want more details.
                """)
            ])
            
            llm_response = await self.llm.ainvoke(response_prompt.format_messages())
            response = llm_response.content
        
        state["last_response"] = response
        state["messages"].append({"role": "assistant", "content": response})
        return state
    
    def _get_age_bracket(self, age: int, insurance_type: str) -> str:
        """Same age bracket logic as before"""
        if insurance_type == "health":
            if 18 <= age <= 25: return "18-25"
            elif 26 <= age <= 35: return "26-35"
            elif 36 <= age <= 50: return "36-50"
            elif 51 <= age <= 65: return "51-65"
        elif insurance_type == "life":
            if 18 <= age <= 30: return "18-30"
            elif 31 <= age <= 45: return "31-45"
            elif 46 <= age <= 60: return "46-60"
        elif insurance_type == "auto":
            if 18 <= age <= 25: return "18-25"
            elif 26 <= age <= 40: return "26-40"
            elif 41 <= age <= 65: return "41-65"
        return "general"
    
    async def chat(self, user_input: str, state: Optional[ChatbotState] = None) -> tuple[str, ChatbotState]:
        """Main chat interface"""
        if state is None:
            state = ChatbotState(
                messages=[],
                user_age=None,
                insurance_type=None,
                user_query="",
                relevant_docs=[],
                missing_info=[],
                conversation_stage="start",
                last_response=""
            )
        
        state["messages"].append({"role": "user", "content": user_input})
        state["user_query"] = user_input
        
        # Run the graph
        result = await self.graph.ainvoke(state)
        
        return result["last_response"], result

# Usage Example
async def main():
    # You'll need to provide your OpenAI API key
    chatbot = InsuranceChatbotWithLLM(llm_api_key="your-api-key-here")
    
    state = None
    test_messages = [
        "Hi, I need some insurance help",
        "I'm 28 years old",
        "I need health insurance",
        "Tell me more about the coverage"
    ]
    
    for message in test_messages:
        print(f"User: {message}")
        response, state = await chatbot.chat(message, state)
        print(f"Bot: {response}\n")
        print("-" * 50)

if __name__ == "__main__":
    asyncio.run(main())