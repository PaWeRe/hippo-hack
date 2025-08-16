"""
Medical Agent Graph for LangGraph Studio
"""

import os
from typing import TypedDict, List, Dict, Any, Annotated, Optional
from dataclasses import dataclass
from langchain_mistralai import ChatMistralAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, InjectedState
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_tavily import TavilySearch
from langchain.retrievers import MultiQueryRetriever, EnsembleRetriever
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class MedicalAgentState(TypedDict):
    """State for the medical agent workflow"""

    messages: Annotated[List, add_messages]
    xray_results: Optional[Dict[str, float]]
    patient_info: Optional[Dict[str, Any]]


def setup_medical_knowledge():
    """Setup comprehensive medical knowledge base with diagnosis guidelines and return both retriever and vector store"""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    medical_texts = [
        # Cardiomegaly - Expert Guidelines
        """DIAGNOSIS: Cardiomegaly
        DEFINITION: Heart enlargement with cardiothoracic ratio >0.5 on chest X-ray
        HIGH CONFIDENCE (>0.8): URGENT - Schedule echocardiogram within 1-2 weeks, refer to cardiologist, monitor blood pressure daily, restrict sodium intake <2g/day, evaluate for heart failure
        MODERATE CONFIDENCE (0.5-0.8): ROUTINE - Follow-up chest X-ray in 3-6 months, consider echocardiogram if symptoms present, monitor for shortness of breath and chest pain
        PATIENT FACTORS: Elderly patients require more urgent evaluation, hypertensive patients need immediate BP control
        FOLLOW-UP: Cardiology consultation within 2 weeks for high confidence findings""",
        # Pneumonia - Expert Guidelines
        """DIAGNOSIS: Pneumonia
        DEFINITION: Lung infection with consolidation visible on chest X-ray
        HIGH CONFIDENCE (>0.7): URGENT - Start antibiotic treatment immediately, monitor oxygen saturation, schedule follow-up in 48-72 hours
        MODERATE CONFIDENCE (0.4-0.7): URGENT - Clinical correlation needed, assess symptoms, consider sputum culture, monitor temperature and respiratory status
        PATIENT FACTORS: Age >65 requires hospitalization consideration, immunocompromised patients need immediate evaluation
        COMPLICATIONS: Monitor for sepsis, respiratory failure, pleural effusion""",
        # Pleural Effusion - Expert Guidelines
        """DIAGNOSIS: Pleural_Effusion  
        DEFINITION: Fluid accumulation in pleural space, visible as costophrenic angle blunting
        HIGH CONFIDENCE (>0.6): URGENT - Refer to pulmonologist, consider thoracentesis for large effusion, investigate underlying cause, monitor respiratory symptoms
        MODERATE CONFIDENCE (0.3-0.6): ROUTINE - Follow-up chest imaging in 2-4 weeks, monitor for shortness of breath, clinical correlation with symptoms
        UNDERLYING CAUSES: Heart failure, infection, malignancy, trauma - workup required
        PROCEDURES: Thoracentesis indicated for large effusions or diagnostic uncertainty""",
        # Pneumothorax - Expert Guidelines
        """DIAGNOSIS: Pneumothorax
        DEFINITION: Air in pleural space causing lung collapse, visible as lung edge without markings
        HIGH CONFIDENCE (>0.5): EMERGENCY - Immediate medical attention required, go to emergency room, may require chest tube insertion, monitor for tension pneumothorax
        MODERATE CONFIDENCE (0.2-0.5): URGENT - Urgent chest CT for confirmation, monitor respiratory status closely, avoid air travel until cleared
        RISK FACTORS: Young tall males, COPD patients, trauma history
        TREATMENT: Small pneumothorax may resolve spontaneously, large requires intervention""",
        # Emphysema - Expert Guidelines
        """DIAGNOSIS: Emphysema
        DEFINITION: Lung tissue destruction with hyperinflation, flattened diaphragms on X-ray
        HIGH CONFIDENCE (>0.6): ROUTINE - Pulmonology referral for spirometry, smoking cessation counseling, consider bronchodilator therapy, pulmonary rehabilitation program
        MODERATE CONFIDENCE (0.3-0.6): ROUTINE - Follow-up with primary care, monitor for progressive dyspnea, lifestyle modifications discussion
        SMOKING CESSATION: Essential for preventing progression, refer to cessation programs
        MANAGEMENT: Bronchodilators, oxygen therapy if indicated, vaccination against respiratory infections""",
        # Normal - Expert Guidelines
        """DIAGNOSIS: Normal
        DEFINITION: No acute findings on chest X-ray, normal heart size and lung fields
        HIGH CONFIDENCE (>0.8): ROUTINE - No acute findings, routine follow-up as needed, continue regular health maintenance, address any persistent symptoms clinically
        REASSURANCE: Normal imaging does not exclude all conditions, clinical correlation important
        FOLLOW-UP: Routine health maintenance, annual imaging if risk factors present""",
        # Atelectasis - Expert Guidelines
        """DIAGNOSIS: Atelectasis
        DEFINITION: Lung collapse or incomplete expansion, appears as increased opacity with volume loss
        HIGH CONFIDENCE (>0.6): ROUTINE - Chest physiotherapy, incentive spirometry, treat underlying cause, pulmonology consultation if persistent
        MODERATE CONFIDENCE (0.3-0.6): ROUTINE - Monitor for improvement, breathing exercises, follow-up imaging in 2-4 weeks
        CAUSES: Post-operative, prolonged bed rest, airway obstruction, mucus plugging
        TREATMENT: Airway clearance, mobilization, treat underlying obstruction""",
        # Mass/Nodule - Expert Guidelines
        """DIAGNOSIS: Mass/Nodule
        DEFINITION: Discrete opacity >3cm (mass) or <3cm (nodule) on chest X-ray
        HIGH CONFIDENCE (>0.7): URGENT - Immediate oncology referral, chest CT with contrast, tissue sampling consideration, staging workup
        MODERATE CONFIDENCE (0.4-0.7): URGENT - Chest CT within 1 week, compare with prior imaging, consider PET scan
        MALIGNANCY RISK: Size, spiculation, growth rate, patient age and smoking history
        WORKUP: CT chest, bronchoscopy or biopsy as indicated, multidisciplinary evaluation""",
        # General X-ray interpretation guidelines
        """SYSTEMATIC CHEST X-RAY INTERPRETATION: 
        1) Verify patient details and image quality
        2) Assess lung fields for consolidation, masses, pneumothorax
        3) Evaluate heart size and shape (cardiothoracic ratio)
        4) Check mediastinal contours and tracheal position  
        5) Examine diaphragms and costophrenic angles
        6) Look at bones and soft tissues
        7) Compare with previous imaging when available
        QUALITY FACTORS: Adequate inspiration, proper positioning, adequate penetration""",
    ]

    documents = [Document(page_content=text) for text in medical_texts]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    split_docs = text_splitter.split_documents(documents)
    vector_store = FAISS.from_documents(split_docs, embeddings)

    return vector_store, vector_store.as_retriever(search_kwargs={"k": 2})


# Tool definitions
@tool
def extract_top_findings(
    state: Annotated[MedicalAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    k: int = 3,
) -> Command:
    """Extract the top k most likely findings from X-ray results. Use this when a patient asks to interpret their X-ray findings."""

    # Check if there are X-ray results in the state
    xray_results = state.get("xray_results")
    if not xray_results:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        "No X-ray results available to analyze. Please ensure X-ray results are loaded first.",
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )

    # Extract the top k diagnoses (highest probability scores)
    sorted_findings = sorted(xray_results.items(), key=lambda x: x[1], reverse=True)
    top_k_diagnoses = sorted_findings[:k]

    # Format the findings
    formatted_results = []

    for i, (diagnosis, confidence) in enumerate(top_k_diagnoses, 1):
        finding_text = f"{diagnosis}: {confidence:.3f} confidence"
        formatted_results.append(f"{i}. {finding_text}")

    result_message = f"Top {len(top_k_diagnoses)} X-ray findings:\n" + "\n".join(
        formatted_results
    )

    return Command(
        update={
            "messages": [
                ToolMessage(
                    result_message,
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


@tool
def retrieve_medical_knowledge(
    query: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    search_type: str = "smart",  # "local", "web", or "smart" (LLM chooses)
    k: int = 3,
) -> Command:
    """Hybrid retrieval tool that intelligently chooses between local FAISS knowledge and web search.

    Args:
        query: Medical query to search for
        search_type: "local" (FAISS only), "web" (Tavily only), or "smart" (LLM chooses best source)
        k: Number of results to return
    """

    global medical_retriever

    try:
        if search_type == "smart":
            # Let the LLM decide based on query characteristics
            search_type = _determine_optimal_search_type(query)

        response_sections = []

        if search_type in ["local", "both"]:
            # Get expert guidelines from local FAISS
            docs = medical_retriever.invoke(query)
            if docs:
                local_knowledge = "📚 **Expert-Vetted Medical Guidelines:**\n\n"
                for i, doc in enumerate(docs[:k], 1):
                    doc_content = doc.page_content.strip()
                    # Truncate content for chat response to keep it concise
                    if len(doc_content) > 300:
                        doc_content = doc_content[:300] + "..."
                    local_knowledge += f"{i}. {doc_content}\n\n"
                local_knowledge += "*Source: Curated expert medical guidelines*\n\n"
                response_sections.append(local_knowledge)

        if search_type in ["web", "both"]:
            # Get current information from web
            try:
                tavily_search = TavilySearch(max_results=2)

                medical_query = (
                    f"medical information {query} recent guidelines treatment"
                )
                web_results = tavily_search.invoke({"query": medical_query})

                if web_results and "results" in web_results:
                    web_knowledge = (
                        "🌐 **Current Medical Information (Web Sources):**\n\n"
                    )
                    # Extract the actual results list from the response
                    search_results = web_results["results"][:2]
                    for i, result in enumerate(search_results, 1):
                        if isinstance(result, dict):
                            title = result.get("title", "Medical Source")
                            content = result.get("content", "")[
                                :150
                            ]  # Shorter content for chat
                            url = result.get("url", "")
                            web_knowledge += f"{i}. **{title}**\n{content}...\n*Web Source: {url}*\n\n"
                        else:
                            web_knowledge += f"{i}. {str(result)[:150]}...\n\n"
                    web_knowledge += "*Note: Web sources should be verified with healthcare professionals*\n\n"
                    response_sections.append(web_knowledge)

            except Exception as web_error:
                response_sections.append(f"⚠️ Web search unavailable: {str(web_error)}")

        # Format response
        source_type = (
            "📊 **Smart Retrieval**"
            if search_type == "smart"
            else f"🔍 **{search_type.title()} Search**"
        )

        if response_sections:
            response_text = f"{source_type} - Medical information for '{query}':\n\n"
            response_text += "\n".join(response_sections)
            response_text += "\n**Important:** Consult healthcare professionals for medical decisions."

            # Add detailed raw results for tool activity display
            response_text += (
                "\n\n"
                + "=" * 50
                + "\n🔍 **RAW SEARCH RESULTS FOR TOOL ACTIVITY:**\n"
                + "=" * 50
                + "\n"
            )

            # Add local search details if used
            if search_type in ["local", "both"]:
                try:
                    local_docs = medical_retriever.invoke(query)
                    if local_docs:
                        response_text += (
                            f"\n📚 **LOCAL SEARCH** (FAISS Vector Store):\n"
                        )
                        response_text += f"Query: '{query}'\n"
                        response_text += f"Retrieved {len(local_docs)} documents:\n\n"
                        for idx, doc in enumerate(local_docs[:k], 1):
                            response_text += f"Document {idx}:\n"
                            response_text += f"Content: {doc.page_content}\n"
                            if hasattr(doc, "metadata") and doc.metadata:
                                response_text += f"Metadata: {doc.metadata}\n"
                            response_text += "-" * 30 + "\n"
                except Exception as local_error:
                    response_text += f"\nLocal search error: {local_error}\n"

            # Add web search details if used
            if search_type in ["web", "both"]:
                try:
                    tavily_search = TavilySearch(max_results=2)
                    medical_query = (
                        f"medical information {query} recent guidelines treatment"
                    )
                    web_results = tavily_search.invoke({"query": medical_query})

                    if web_results and "results" in web_results:
                        response_text += f"\n🌐 **WEB SEARCH** (Tavily):\n"
                        response_text += f"Query: '{medical_query}'\n"
                        response_text += (
                            f"Found {len(web_results['results'])} results:\n\n"
                        )

                        for idx, result in enumerate(web_results["results"][:2], 1):
                            if isinstance(result, dict):
                                response_text += f"Result {idx}:\n"
                                response_text += (
                                    f"Title: {result.get('title', 'N/A')}\n"
                                )
                                response_text += f"URL: {result.get('url', 'N/A')}\n"
                                response_text += (
                                    f"Score: {result.get('score', 'N/A')}\n"
                                )
                                response_text += (
                                    f"Content: {result.get('content', 'N/A')}\n"
                                )
                                response_text += (
                                    f"Raw Content: {result.get('raw_content', 'N/A')}\n"
                                )
                                response_text += "-" * 30 + "\n"

                        # Add full API response details
                        response_text += (
                            f"\nFull API Response Keys: {list(web_results.keys())}\n"
                        )
                        response_text += f"Query: {web_results.get('query', 'N/A')}\n"
                        response_text += f"Follow-up Questions: {web_results.get('follow_up_questions', 'N/A')}\n"
                        response_text += f"Response Time: {web_results.get('response_time', 'N/A')}\n"
                        response_text += (
                            f"Request ID: {web_results.get('request_id', 'N/A')}\n"
                        )
                except Exception as web_error:
                    response_text += f"\nWeb search error: {web_error}\n"

        else:
            response_text = f"No medical information found for: {query}. Please consult your healthcare provider."

    except Exception as e:
        response_text = f"Error retrieving medical knowledge: {str(e)}. Please consult your healthcare provider."

    return Command(
        update={
            "messages": [
                ToolMessage(
                    response_text,
                    tool_call_id=tool_call_id,
                )
            ]
        }
    )


def _determine_optimal_search_type(query: str) -> str:
    """Determine the best search type based on query characteristics"""
    query_lower = query.lower()

    # Queries that benefit from expert guidelines (local search)
    local_indicators = [
        "diagnosis",
        "guideline",
        "recommendation",
        "treatment protocol",
        "management",
        "workup",
        "follow-up",
        "urgent",
        "emergency",
    ]

    # Queries that benefit from current information (web search)
    web_indicators = [
        "latest",
        "recent",
        "new",
        "update",
        "current",
        "2024",
        "emerging",
        "breakthrough",
        "novel",
        "research",
        "study",
        "trial",
    ]

    # Check for specific diagnosis names (prefer local guidelines)
    diagnosis_names = [
        "cardiomegaly",
        "pneumonia",
        "pleural effusion",
        "pneumothorax",
        "emphysema",
        "atelectasis",
        "mass",
        "nodule",
    ]

    if any(diag in query_lower for diag in diagnosis_names):
        return "local"
    elif any(indicator in query_lower for indicator in local_indicators):
        return "local"
    elif any(indicator in query_lower for indicator in web_indicators):
        return "web"
    else:
        return "both"  # Use both sources for comprehensive information


@tool
def generate_recommendations(
    state: Annotated[MedicalAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Generate evidence-based medical recommendations using RAG from expert guidelines. Routes patients to appropriate care levels based on diagnosis-specific protocols."""

    global medical_retriever

    xray_results = state.get("xray_results")
    patient_info = state.get("patient_info", {})

    if not xray_results:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        "No X-ray results available for generating recommendations. Please extract findings first.",
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )

    # Get top findings sorted by confidence
    sorted_findings = sorted(xray_results.items(), key=lambda x: x[1], reverse=True)

    all_recommendations = []
    highest_urgency = "routine"
    patient_age = patient_info.get("age", 0)
    patient_gender = patient_info.get("gender", "").lower()
    patient_hypertension = patient_info.get("hypertension", False)
    patient_name = patient_info.get("name", "Patient")

    for diagnosis, confidence in sorted_findings:
        # Skip very low confidence findings
        if confidence < 0.1:
            continue

        # Use RAG to get expert guidelines for this specific diagnosis
        guideline_query = f"DIAGNOSIS: {diagnosis} HIGH CONFIDENCE MODERATE CONFIDENCE guidelines recommendations"
        guidelines_docs = medical_retriever.invoke(guideline_query)

        if guidelines_docs:
            # Parse guidelines from retrieved documents
            recommendations, urgency = _parse_guidelines_from_rag(
                guidelines_docs,
                diagnosis,
                confidence,
                patient_age,
                patient_gender,
                patient_hypertension,
            )

            all_recommendations.extend(recommendations)

            # Update highest urgency level
            urgency_order = {"routine": 1, "urgent": 2, "emergency": 3}
            if urgency_order.get(urgency, 1) > urgency_order.get(highest_urgency, 1):
                highest_urgency = urgency

    # Format final recommendations with urgency context
    urgency_messages = {
        "emergency": "🚨 EMERGENCY: Seek immediate medical attention.",
        "urgent": "⚠️ URGENT: Schedule specialist consultation within 1-2 weeks.",
        "routine": "📅 ROUTINE: Follow up with primary care physician as appropriate.",
    }

    urgency_msg = urgency_messages.get(highest_urgency, urgency_messages["routine"])

    if all_recommendations:
        # Add personalized header
        personal_header = f"**Personalized Recommendations for {patient_name}:**\n"
        if patient_age > 0:
            personal_header += f"Age: {patient_age} | "
        if patient_gender:
            personal_header += f"Gender: {patient_gender.title()} | "
        if patient_hypertension:
            personal_header += f"Known Hypertension | "
        personal_header = personal_header.rstrip(" | ") + "\n\n"

        formatted_recommendations = (
            personal_header
            + f"{urgency_msg}\n\n**Evidence-Based Recommendations:**\n\n"
            + "\n".join(f"• {rec}" for rec in all_recommendations)
            + f"\n\n*Based on expert medical guidelines for confidence levels and patient factors*"
        )
    else:
        formatted_recommendations = "No specific recommendations found. Please consult with your healthcare provider for personalized medical advice."

    return Command(
        update={
            "messages": [
                ToolMessage(
                    formatted_recommendations,
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


def _parse_guidelines_from_rag(
    guidelines_docs,
    diagnosis: str,
    confidence: float,
    patient_age: int,
    patient_gender: str = "",
    patient_hypertension: bool = False,
) -> tuple[list, str]:
    """Parse expert guidelines from RAG documents to extract specific recommendations"""

    recommendations = []
    urgency = "routine"

    for doc in guidelines_docs:
        content = doc.page_content.upper()

        # Skip if this document doesn't match our diagnosis
        if f"DIAGNOSIS: {diagnosis.upper()}" not in content:
            continue

        # Determine confidence category
        if confidence >= 0.8:
            confidence_category = "HIGH CONFIDENCE"
        elif confidence >= 0.5:
            confidence_category = "MODERATE CONFIDENCE"
        else:
            confidence_category = "LOW CONFIDENCE"

        # Extract recommendations for the appropriate confidence level
        lines = content.split("\n")
        in_target_section = False

        for line in lines:
            line = line.strip()

            # Check if we're entering the right confidence section
            if confidence_category in line:
                in_target_section = True

                # Extract urgency level from this line
                if "EMERGENCY" in line:
                    urgency = "emergency"
                elif "URGENT" in line:
                    urgency = "urgent"
                else:
                    urgency = "routine"

                # Extract recommendations from this line
                if ":" in line:
                    rec_part = line.split(":", 2)[-1].strip()
                    if rec_part and rec_part not in [
                        "",
                        "URGENT",
                        "ROUTINE",
                        "EMERGENCY",
                    ]:
                        # Split on common delimiters
                        rec_items = rec_part.replace(" - ", ", ").split(", ")
                        for item in rec_items:
                            item = item.strip()
                            if (
                                item and len(item) > 10
                            ):  # Filter out short non-recommendations
                                recommendations.append(item)

            elif in_target_section and line.startswith("PATIENT FACTORS:"):
                # Add age-specific considerations
                if patient_age > 65 and "elderly" in line.lower():
                    if "urgent" in line.lower() and urgency == "routine":
                        urgency = "urgent"
                # Add hypertension-specific considerations
                if patient_hypertension and "hypertensive" in line.lower():
                    if "immediate" in line.lower() and urgency != "emergency":
                        urgency = "urgent"
                # Add gender-specific considerations
                if patient_gender == "male" and diagnosis.lower() == "pneumothorax":
                    # Young tall males are at higher risk for pneumothorax
                    if patient_age < 40 and urgency == "routine":
                        urgency = "urgent"

            elif in_target_section and line.startswith(
                ("MODERATE CONFIDENCE", "HIGH CONFIDENCE", "LOW CONFIDENCE")
            ):
                # Entering a different confidence section
                if confidence_category not in line:
                    in_target_section = False

    # If no specific recommendations found, add a general one
    if not recommendations:
        recommendations.append(
            f"Clinical correlation recommended for {diagnosis} finding (confidence: {confidence:.2f})"
        )

    return recommendations, urgency


@tool
def get_patient_information(
    state: Annotated[MedicalAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Retrieve current patient information and demographics for personalized care."""

    patient_info = state.get("patient_info", {})

    if not patient_info:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        "No patient information available in the current session.",
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )

    # Format patient information
    info_text = "**Current Patient Information:**\n\n"

    if patient_info.get("name"):
        info_text += f"👤 **Name:** {patient_info['name']}\n"

    if patient_info.get("age"):
        info_text += f"🎂 **Age:** {patient_info['age']} years\n"

    if patient_info.get("gender"):
        info_text += f"⚧ **Gender:** {patient_info['gender']}\n"

    if patient_info.get("height") and patient_info.get("weight"):
        height = patient_info["height"]
        weight = patient_info["weight"]
        # Calculate BMI
        if height > 3:  # Height in cm, convert to meters
            height_m = height / 100
        else:
            height_m = height
        bmi = weight / (height_m**2)

        info_text += f"📏 **Height:** {patient_info['height']} {'cm' if patient_info['height'] > 3 else 'm'}\n"
        info_text += f"⚖️ **Weight:** {patient_info['weight']} kg\n"
        info_text += f"📊 **BMI:** {bmi:.1f} "

        # BMI category
        if bmi < 18.5:
            info_text += "(Underweight)\n"
        elif bmi < 25:
            info_text += "(Normal)\n"
        elif bmi < 30:
            info_text += "(Overweight)\n"
        else:
            info_text += "(Obese)\n"

    if patient_info.get("blood_type"):
        info_text += f"🩸 **Blood Type:** {patient_info['blood_type']}\n"

    if patient_info.get("hypertension"):
        info_text += f"🫀 **Known Conditions:** Hypertension\n"

    info_text += (
        "\n*This information is used to provide personalized medical recommendations.*"
    )

    return Command(
        update={
            "messages": [
                ToolMessage(
                    info_text,
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


# Hardcoded guidelines function removed - now using RAG-based approach with vector store


# Initialize components
def get_llm():
    """Get LLM instance with proper error handling for environment variables"""
    api_key = os.getenv("API_KEY_MISTRAL")
    if not api_key:
        raise ValueError("API_KEY_MISTRAL environment variable is required")
    return ChatMistralAI(api_key=api_key, model="mistral-large-latest", temperature=0.3)


# Initialize LLM lazily to avoid import-time errors
llm = None

# Initialize the medical knowledge base globally
medical_vector_store, medical_retriever = setup_medical_knowledge()

# Define tools
tools = [
    extract_top_findings,
    retrieve_medical_knowledge,
    generate_recommendations,
    get_patient_information,
]


# Bind tools to LLM (initialized lazily)
def get_llm_with_tools():
    """Get LLM with tools bound, initializing if needed"""
    global llm
    if llm is None:
        llm = get_llm()
    return llm.bind_tools(tools)


llm_with_tools = None

# Create tool node
tool_node = ToolNode(tools)


# Define the main chatbot node
def medical_chatbot(state: MedicalAgentState) -> MedicalAgentState:
    """Main chatbot that can call tools"""
    # Get LLM with tools, initializing if needed
    llm_with_tools = get_llm_with_tools()
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# Build the graph
workflow = StateGraph(MedicalAgentState)

# Add nodes
workflow.add_node("medical_chatbot", medical_chatbot)
workflow.add_node("tools", tool_node)

# Add edges - ToolNode handles routing automatically
workflow.set_entry_point("medical_chatbot")
workflow.add_conditional_edges(
    "medical_chatbot",
    lambda state: "tools" if state["messages"][-1].tool_calls else END,
    ["tools", END],
)
workflow.add_edge("tools", "medical_chatbot")

# Compile the graph
graph = workflow.compile()


# Helper function to invoke (for testing/usage)
def invoke_medical_agent(inputs, **kwargs):
    """Helper function to invoke the medical agent."""
    return graph.invoke(inputs, **kwargs)


# Export for LangGraph Studio
__all__ = ["graph", "invoke_medical_agent"]
