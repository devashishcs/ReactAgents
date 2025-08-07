from flask import Flask, request, jsonify, session
from flask_cors import CORS
import re
import json
import uuid
import asyncio
from datetime import datetime, timedelta
from typing import TypedDict, List, Optional, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
import os
from functools import wraps

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'session-key')
CORS(app)  # Enable CORS for frontend integration
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# State definition (same as before) this is the memory box
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

# Global storage for conversation states (in production, use Redis or database)
conversation_states = {}

class InsuranceChatbotAPI:
    def __init__(self, llm_api_key: str = None):
        self.llm_api_key = llm_api_key or GROQ_API_KEY
        if not self.llm_api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass it directly.")
        
        self.llm =  ChatGroq(api_key=GROQ_API_KEY, model="llama3-70b-8192", temperature=0)
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow with LLM integration"""
        graph = StateGraph(ChatbotState)
        
        # Add nodes
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
        
        try:
            llm_response = await self.llm.ainvoke(extraction_prompt.format_messages())
            extracted_data = json.loads(llm_response.content)
            
            if extracted_data.get("age"):
                state["user_age"] = extracted_data["age"]
            if extracted_data.get("insurance_type"):
                state["insurance_type"] = extracted_data["insurance_type"]
                
        except (json.JSONDecodeError, Exception):
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
        
        history_text = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in conversation_history[-3:]
        ])
        
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
        
        try:
            llm_response = await self.llm.ainvoke(followup_prompt.format_messages())
            response = llm_response.content
        except Exception as e:
            # Fallback response if LLM fails
            if "age" in missing_info and "insurance_type" in missing_info:
                response = "Hi! I'd be happy to help you find insurance options. Could you tell me your age and what type of insurance you're looking for? (health, life, or auto)"
            elif "age" in missing_info:
                response = "Could you please tell me your age so I can find the best insurance options for you?"
            elif "insurance_type" in missing_info:
                response = "What type of insurance are you interested in? I can help with health, life, or auto insurance."
            else:
                response = "I need a bit more information to help you better."
        
        state["last_response"] = response
        state["messages"].append({"role": "assistant", "content": response})
        return state
    
    def _search_documents(self, state: ChatbotState) -> ChatbotState:
        """Search insurance documents"""
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
            try:
                no_results_prompt = ChatPromptTemplate.from_messages([
                    SystemMessage(content="You are a helpful insurance assistant. Generate a polite response when no insurance options are found."),
                    HumanMessage(content=f"No insurance options found for {insurance_type} insurance for age {user_age}. Suggest contacting support.")
                ])
                llm_response = await self.llm.ainvoke(no_results_prompt.format_messages())
                response = llm_response.content
            except Exception:
                response = f"I apologize, but I couldn't find specific {insurance_type} insurance options for your age group. Please contact our support team for personalized assistance."
        else:
            doc_data = relevant_docs[0]["data"]
            
            insurance_info = f"""
            Insurance Type: {insurance_type.title()}
            Age: {user_age}
            Premium: {doc_data.get('premium', 'Contact for quote')}
            Coverage: {doc_data.get('coverage', 'Standard coverage')}
            Deductible: {doc_data.get('deductible', 'N/A')}
            """
            
            try:
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
            except Exception:
                response = f"""🎯 Great! I found {insurance_type} insurance options for you:

📊 **Premium**: {doc_data.get('premium', 'Contact for quote')}
🛡️ **Coverage**: {doc_data.get('coverage', 'Standard coverage')}
💰 **Deductible**: {doc_data.get('deductible', 'N/A')}

Would you like more details about this plan or have any questions? 😊"""
        
        state["last_response"] = response
        state["messages"].append({"role": "assistant", "content": response})
        return state
    
    def _get_age_bracket(self, age: int, insurance_type: str) -> str:
        """Get age bracket for insurance type"""
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

# Initialize the chatbot instance
try:
    chatbot = InsuranceChatbotAPI()
except ValueError as e:
    print(f"Error initializing chatbot: {e}")
    chatbot = None

# Helper function to run async functions in Flask
def async_route(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(f(*args, **kwargs))
        finally:
            loop.close()
    return wrapper

# API Routes

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "Insurance Chatbot API",
        "timestamp": datetime.now().isoformat(),
        "chatbot_ready": chatbot is not None
    })

@app.route('/api/chat/start', methods=['POST'])
def start_conversation():
    """Start a new conversation"""
    conversation_id = str(uuid.uuid4())
    
    # Initialize conversation state
    conversation_states[conversation_id] = {
        "state": ChatbotState(
            messages=[],
            user_age=None,
            insurance_type=None,
            user_query="",
            relevant_docs=[],
            missing_info=[],
            conversation_stage="start",
            last_response=""
        ),
        "created_at": datetime.now(),
        "last_activity": datetime.now()
    }
    
    return jsonify({
        "conversation_id": conversation_id,
        "message": "Hi! I'm your insurance assistant. How can I help you find the right insurance today?",
        "status": "success"
    })

@app.route('/api/chat/<conversation_id>', methods=['POST'])
@async_route
async def chat_message(conversation_id):
    """Send a message in an existing conversation"""
    if not chatbot:
        return jsonify({"error": "Chatbot not initialized. Please check OpenAI API key."}), 500
    
    # Get request data
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({"error": "Message is required"}), 400
    
    user_message = data['message']
    
    # Check if conversation exists
    if conversation_id not in conversation_states:
        return jsonify({"error": "Conversation not found"}), 404
    
    # Get conversation state
    conversation = conversation_states[conversation_id]
    state = conversation["state"]
    
    try:
        # Process message
        response, updated_state = await chatbot.chat(user_message, state)
        
        # Update conversation state
        conversation_states[conversation_id] = {
            "state": updated_state,
            "created_at": conversation["created_at"],
            "last_activity": datetime.now()
        }
        
        return jsonify({
            "response": response,
            "conversation_id": conversation_id,
            "user_info": {
                "age": updated_state.get("user_age"),
                "insurance_type": updated_state.get("insurance_type")
            },
            "status": "success"
        })
        
    except Exception as e:
        return jsonify({
            "error": f"Error processing message: {str(e)}",
            "status": "error"
        }), 500

@app.route('/api/chat/<conversation_id>/history', methods=['GET'])
def get_conversation_history(conversation_id):
    """Get conversation history"""
    if conversation_id not in conversation_states:
        return jsonify({"error": "Conversation not found"}), 404
    
    conversation = conversation_states[conversation_id]
    state = conversation["state"]
    
    return jsonify({
        "conversation_id": conversation_id,
        "messages": state.get("messages", []),
        "user_info": {
            "age": state.get("user_age"),
            "insurance_type": state.get("insurance_type")
        },
        "created_at": conversation["created_at"].isoformat(),
        "last_activity": conversation["last_activity"].isoformat(),
        "status": "success"
    })

@app.route('/api/chat/<conversation_id>', methods=['DELETE'])
def delete_conversation(conversation_id):
    """Delete a conversation"""
    if conversation_id not in conversation_states:
        return jsonify({"error": "Conversation not found"}), 404
    
    del conversation_states[conversation_id]
    
    return jsonify({
        "message": "Conversation deleted successfully",
        "status": "success"
    })

@app.route('/api/insurance/types', methods=['GET'])
def get_insurance_types():
    """Get available insurance types and age brackets"""
    return jsonify({
        "insurance_types": list(INSURANCE_DATABASE.keys()),
        "age_brackets": {
            insurance_type: list(brackets.keys())
            for insurance_type, brackets in INSURANCE_DATABASE.items()
        },
        "status": "success"
    })

@app.route('/api/insurance/quote', methods=['POST'])
def get_insurance_quote():
    """Get insurance quote directly"""
    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body is required"}), 400
    
    age = data.get('age')
    insurance_type = data.get('insurance_type')
    
    if not age or not insurance_type:
        return jsonify({"error": "Age and insurance_type are required"}), 400
    
    if not isinstance(age, int) or age < 18 or age > 100:
        return jsonify({"error": "Age must be between 18 and 100"}), 400
    
    if insurance_type not in INSURANCE_DATABASE:
        return jsonify({"error": f"Invalid insurance type. Available: {list(INSURANCE_DATABASE.keys())}"}), 400
    
    # Get age bracket
    if insurance_type == "health":
        if 18 <= age <= 25: age_bracket = "18-25"
        elif 26 <= age <= 35: age_bracket = "26-35"
        elif 36 <= age <= 50: age_bracket = "36-50"
        elif 51 <= age <= 65: age_bracket = "51-65"
        else: age_bracket = None
    elif insurance_type == "life":
        if 18 <= age <= 30: age_bracket = "18-30"
        elif 31 <= age <= 45: age_bracket = "31-45"
        elif 46 <= age <= 60: age_bracket = "46-60"
        else: age_bracket = None
    elif insurance_type == "auto":
        if 18 <= age <= 25: age_bracket = "18-25"
        elif 26 <= age <= 40: age_bracket = "26-40"
        elif 41 <= age <= 65: age_bracket = "41-65"
        else: age_bracket = None
    else:
        age_bracket = None
    
    if not age_bracket or age_bracket not in INSURANCE_DATABASE[insurance_type]:
        return jsonify({"error": "No insurance options available for this age and type"}), 404
    
    quote = INSURANCE_DATABASE[insurance_type][age_bracket]
    
    return jsonify({
        "age": age,
        "insurance_type": insurance_type,
        "age_bracket": age_bracket,
        "quote": quote,
        "status": "success"
    })

# Cleanup old conversations (run periodically in production)
def cleanup_old_conversations():
    """Remove conversations older than 24 hours"""
    cutoff_time = datetime.now() - timedelta(hours=24)
    expired_conversations = [
        conv_id for conv_id, conv_data in conversation_states.items()
        if conv_data["last_activity"] < cutoff_time
    ]
    
    for conv_id in expired_conversations:
        del conversation_states[conv_id]
    
    return len(expired_conversations)

@app.route('/api/admin/cleanup', methods=['POST'])
def admin_cleanup():
    """Admin endpoint to cleanup old conversations"""
    cleaned_count = cleanup_old_conversations()
    return jsonify({
        "cleaned_conversations": cleaned_count,
        "active_conversations": len(conversation_states),
        "status": "success"
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Set environment variables if not already set
    if not os.getenv('OPENAI_API_KEY'):
        print("Warning: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY='your-api-key-here'")
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        debug=os.getenv('FLASK_ENV') == 'development'
    )