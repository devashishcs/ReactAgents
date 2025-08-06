import re
import json
from typing import TypedDict, List, Optional, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
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

# Mock insurance database - replace with your actual documents
INSURANCE_DATABASE = {
    "health": {
        "18-25": {
            "premium": "$150-200/month",
            "coverage": "Basic health coverage, preventive care, emergency services",
            "deductible": "$1,500",
            "benefits": ["Doctor visits", "Prescription coverage", "Emergency care"]
        },
        "26-35": {
            "premium": "$180-250/month",
            "coverage": "Comprehensive health coverage with maternity benefits",
            "deductible": "$1,200",
            "benefits": ["Doctor visits", "Prescription coverage", "Maternity care", "Specialist visits"]
        },
        "36-50": {
            "premium": "$220-300/month",
            "coverage": "Enhanced coverage with preventive screenings",
            "deductible": "$1,000",
            "benefits": ["Doctor visits", "Prescription coverage", "Preventive screenings", "Chronic disease management"]
        },
        "51-65": {
            "premium": "$280-400/month",
            "coverage": "Premium coverage with comprehensive benefits",
            "deductible": "$800",
            "benefits": ["Doctor visits", "Prescription coverage", "Specialist care", "Advanced diagnostics"]
        }
    },
    "life": {
        "18-30": {
            "premium": "$20-40/month",
            "coverage": "$250,000-500,000",
            "type": "Term life insurance",
            "benefits": ["Death benefit", "Accidental death coverage"]
        },
        "31-45": {
            "premium": "$35-70/month",
            "coverage": "$300,000-750,000",
            "type": "Term or whole life insurance",
            "benefits": ["Death benefit", "Cash value (whole life)", "Loan options"]
        },
        "46-60": {
            "premium": "$80-150/month",
            "coverage": "$200,000-500,000",
            "type": "Whole life or universal life",
            "benefits": ["Death benefit", "Investment component", "Tax advantages"]
        }
    },
    "auto": {
        "18-25": {
            "premium": "$120-200/month",
            "coverage": "Liability + Comprehensive",
            "deductible": "$500-1000",
            "benefits": ["Liability coverage", "Collision", "Comprehensive", "Uninsured motorist"]
        },
        "26-40": {
            "premium": "$90-150/month",
            "coverage": "Full coverage with lower rates",
            "deductible": "$250-750",
            "benefits": ["Liability coverage", "Collision", "Comprehensive", "Rental car coverage"]
        },
        "41-65": {
            "premium": "$80-130/month",
            "coverage": "Mature driver discounts available",
            "deductible": "$250-500",
            "benefits": ["Liability coverage", "Collision", "Comprehensive", "Good driver discounts"]
        }
    }
}

class InsuranceChatbot:
    def __init__(self):
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        graph = StateGraph(ChatbotState)
        
        # Add nodes
        graph.add_node("classify_intent", self._classify_intent)
        graph.add_node("extract_info", self._extract_info)
        graph.add_node("collect_missing_info", self._collect_missing_info)
        graph.add_node("search_documents", self._search_documents)
        graph.add_node("generate_response", self._generate_response)
        graph.add_node("ask_followup", self._ask_followup)
        
        # Define edges
        graph.add_edge("classify_intent", "extract_info")
        
        # Conditional edges
        graph.add_conditional_edges(
            "extract_info",
            self._should_collect_info,
            {
                "collect_info": "collect_missing_info",
                "search_docs": "search_documents",
                "ask_followup": "ask_followup"
            }
        )
        
        graph.add_edge("collect_missing_info", "ask_followup")
        graph.add_edge("search_documents", "generate_response")
        graph.add_edge("ask_followup", END)
        graph.add_edge("generate_response", END)
        
        # Set entry point
        graph.set_entry_point("classify_intent")
        
        return graph.compile()
    
    def _classify_intent(self, state: ChatbotState) -> ChatbotState:
        """Classify user intent and initialize conversation"""
        user_message = state["messages"][-1]["content"] if state["messages"] else ""
        
        # Simple intent classification
        insurance_keywords = {
            "health": ["health", "medical", "doctor", "hospital", "healthcare"],
            "life": ["life", "death", "beneficiary", "term", "whole"],
            "auto": ["auto", "car", "vehicle", "driving", "collision"]
        }
        
        detected_type = None
        for ins_type, keywords in insurance_keywords.items():
            if any(keyword in user_message.lower() for keyword in keywords):
                detected_type = ins_type
                break
        
        state["conversation_stage"] = "extracting_info"
        state["user_query"] = user_message
        if detected_type:
            state["insurance_type"] = detected_type
            
        return state
    
    def _extract_info(self, state: ChatbotState) -> ChatbotState:
        """Extract age and insurance type from user message"""
        user_message = state["user_query"]
        
        # Extract age using regex
        age_match = re.search(r'\b(\d{1,2})\b', user_message)
        if age_match:
            age = int(age_match.group(1))
            if 18 <= age <= 100:  # Reasonable age range
                state["user_age"] = age
        
        # Extract insurance type if not already detected
        if not state.get("insurance_type"):
            insurance_types = ["health", "life", "auto"]
            for ins_type in insurance_types:
                if ins_type in user_message.lower():
                    state["insurance_type"] = ins_type
                    break
        
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
    
    def _collect_missing_info(self, state: ChatbotState) -> ChatbotState:
        """Identify what information is missing"""
        state["conversation_stage"] = "collecting_info"
        return state
    
    def _ask_followup(self, state: ChatbotState) -> ChatbotState:
        """Generate follow-up questions for missing information"""
        missing_info = state.get("missing_info", [])
        
        if "age" in missing_info and "insurance_type" in missing_info:
            response = "Hi! I'd be happy to help you find insurance options. Could you please tell me:\n1. Your age\n2. What type of insurance you're looking for (health, life, or auto)?"
        elif "age" in missing_info:
            response = "To provide you with the most accurate insurance information, could you please share your age?"
        elif "insurance_type" in missing_info:
            response = "What type of insurance are you interested in? I can help with:\n- Health insurance\n- Life insurance\n- Auto insurance"
        else:
            response = "I have all the information I need. Let me search for insurance options for you."
        
        state["last_response"] = response
        state["messages"].append({"role": "assistant", "content": response})
        return state
    
    def _search_documents(self, state: ChatbotState) -> ChatbotState:
        """Search insurance documents based on age and type"""
        age = state["user_age"]
        insurance_type = state["insurance_type"]
        
        # Determine age bracket
        age_bracket = self._get_age_bracket(age, insurance_type)
        
        # Search in database
        relevant_docs = []
        if insurance_type in INSURANCE_DATABASE and age_bracket in INSURANCE_DATABASE[insurance_type]:
            doc = INSURANCE_DATABASE[insurance_type][age_bracket]
            relevant_docs.append({
                "insurance_type": insurance_type,
                "age_bracket": age_bracket,
                "data": doc
            })
        
        state["relevant_docs"] = relevant_docs
        state["conversation_stage"] = "searching_complete"
        return state
    
    def _get_age_bracket(self, age: int, insurance_type: str) -> str:
        """Determine age bracket based on insurance type"""
        if insurance_type == "health":
            if 18 <= age <= 25:
                return "18-25"
            elif 26 <= age <= 35:
                return "26-35"
            elif 36 <= age <= 50:
                return "36-50"
            elif 51 <= age <= 65:
                return "51-65"
        elif insurance_type == "life":
            if 18 <= age <= 30:
                return "18-30"
            elif 31 <= age <= 45:
                return "31-45"
            elif 46 <= age <= 60:
                return "46-60"
        elif insurance_type == "auto":
            if 18 <= age <= 25:
                return "18-25"
            elif 26 <= age <= 40:
                return "26-40"
            elif 41 <= age <= 65:
                return "41-65"
        
        return "general"  # fallback
    
    def _generate_response(self, state: ChatbotState) -> ChatbotState:
        """Generate final response with insurance information"""
        relevant_docs = state["relevant_docs"]
        
        if not relevant_docs:
            response = f"I couldn't find specific insurance options for a {state['user_age']}-year-old looking for {state['insurance_type']} insurance. Please contact our support team for personalized assistance."
        else:
            doc = relevant_docs[0]
            data = doc["data"]
            insurance_type = doc["insurance_type"].title()
            
            response = f"Great! Here are {insurance_type} insurance options for age {state['user_age']}:\n\n"
            response += f"💰 **Premium**: {data.get('premium', 'Contact for quote')}\n"
            response += f"🏥 **Coverage**: {data.get('coverage', 'Standard coverage')}\n"
            
            if 'deductible' in data:
                response += f"💳 **Deductible**: {data['deductible']}\n"
            
            if 'benefits' in data:
                response += f"\n✅ **Key Benefits**:\n"
                for benefit in data['benefits']:
                    response += f"   • {benefit}\n"
            
            response += f"\nWould you like more details about any specific aspect of this {insurance_type.lower()} insurance plan?"
        
        state["last_response"] = response
        state["messages"].append({"role": "assistant", "content": response})
        state["conversation_stage"] = "complete"
        return state
    
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
        
        # Add user message to state
        state["messages"].append({"role": "user", "content": user_input})
        state["user_query"] = user_input
        
        # Run the graph
        result = await self.graph.ainvoke(state)
        
        return result["last_response"], result

# Usage example and testing
async def main():
    """Example usage of the insurance chatbot"""
    chatbot = InsuranceChatbot()
    state = None
    
    print("Insurance Chatbot Started! Type 'quit' to exit.\n")
    
    # Simulate conversation
    test_messages = [
        "Hi, I need health insurance",
        "I'm 28 years old",
        "Can you tell me more about the deductible?",
        "What about life insurance for someone my age?"
    ]
    
    for message in test_messages:
        print(f"User: {message}")
        response, state = await chatbot.chat(message, state)
        print(f"Bot: {response}\n")
        print("-" * 50)

# Interactive chat function
async def interactive_chat():
    """Interactive chat session"""
    chatbot = InsuranceChatbot()
    state = None
    
    print("🏥 Insurance Chatbot Started! Type 'quit' to exit.\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Bot: Thank you for using our insurance chatbot. Have a great day!")
            break
        
        if not user_input:
            continue
        
        try:
            response, state = await chatbot.chat(user_input, state)
            print(f"Bot: {response}\n")
        except Exception as e:
            print(f"Bot: Sorry, I encountered an error: {str(e)}")
            print("Bot: Please try rephrasing your question.\n")

if __name__ == "__main__":
    # Run the example
    print("Running example conversation...")
    asyncio.run(main())
    
    print("\n" + "="*60)
    print("Starting interactive chat...")
    # Uncomment the line below for interactive mode
    # asyncio.run(interactive_chat())