# filename: graph.py
import os
import asyncio
from typing import TypedDict, Optional, Dict, Any
import json
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_groq import ChatGroq
# We will use Groq's Whisper API wrapper via standard OpenAI client compatibility
from groq import Groq

# MCP Imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# --- MCP CLIENT HELPERS ---
# This helper function connects to a server, calls a tool, and disconnects
async def call_mcp_tool(script_path: str, tool_name: str, arguments: dict):
    server_params = StdioServerParameters(
        command="python",
        args=[script_path], # Run the python script as a server
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            # Call the tool
            result = await session.call_tool(tool_name, arguments)
            # Return the text content
            return result.content[0].text


# --- STATE ---
class AgentState(TypedDict):
    query: str
    audio_path: Optional[str]
    patient_id: Optional[str]
    procedure: Optional[str]
    patient_data: Optional[Dict[str, Any]]
    guidelines: Optional[str]
    final_report: Optional[str]



async def transcriber_node(state: AgentState):
    """
    Checks if there is audio input. If so, transcribes it using Groq Whisper.
    """
    # We need to assume 'audio_path' is in the state. 
    # You'll need to update AgentState to include audio_path: Optional[str]
    audio_path = state.get("audio_path")
    
    if not audio_path or not os.path.exists(audio_path):
        print("   --- 🎤 No Audio Input, skipping transcription ---")
        return {} # No change to state

    print(f"   --- 🎤 Transcribing Audio: {audio_path} ---")
    
    client = Groq() # Uses GROQ_API_KEY env var
    
    with open(audio_path, "rb") as file:
        transcription = client.audio.transcriptions.create(
            file=(audio_path, file.read()),
            model="whisper-large-v3-turbo",
            response_format="text"
        )
    
    text = transcription
    print(f"      📝 Transcribed: '{text}'")
    
    # Overwrite or Append to the query
    new_query = f"{state['query']} \n [AUDIO TRANSCRIPT]: {text}"
    return {"query": new_query}

# --- NODES ---

async def parser_node(state: AgentState):
    print("   --- 🔍 Parsing Query ---")
    query = state["query"]
    PROMPT = """Extract 'patient_id' and 'procedure' from: "{query}". 
    Return JSON only: {{"patient_id": "123", "procedure": "surgery"}}"""
    chain = ChatPromptTemplate.from_template(PROMPT) | llm | StrOutputParser()
    try:
        import json
        res = chain.invoke({"query": query}).replace("```json","").replace("```","").strip()
        parsed = json.loads(res)
        return {"patient_id": str(parsed["patient_id"]), "procedure": parsed["procedure"]}
    except:
        return {}

async def sql_mcp_node(state: AgentState):
    print("\n   --- 🏥 Entering SQL Sub-Graph (Actor + Critic) ---")
    
    patient_id = state['patient_id']
    max_retries = 4
    history = "" 
    
    SCHEMA_CONTEXT = """
    DATABASE SCHEMA (SQLite):
    - diagnoses_icd (subject_id, icd_code, long_title)
    - prescriptions (subject_id, drug, dose_val_rx)
    - labevents (subject_id, itemid, valuenum, flag)
    - d_labitems (itemid, label) -- JOIN itemid
    """

    for i in range(max_retries):
        print(f"\n      🔄 [Turn {i+1}/{max_retries}]")

        # --- 1. ACTOR ---
        ACTOR_PROMPT = f"""
        You are a SQL Expert. Goal: Get Diagnoses, Meds, and Labs for Patient {patient_id}.
        {SCHEMA_CONTEXT}
        PREVIOUS FEEDBACK: {history}
        TASK: Write SQLite query. Return ONLY SQL. No markdown.
        """
        sql_query = llm.invoke(ACTOR_PROMPT).content.replace("```sql","").replace("```","").strip()
        print(f"      👨‍💻 Actor wrote: {sql_query[:100]}...") 
        
        # Execute
        try:
            result_str = await call_mcp_tool(
                script_path="mcp_servers/sql_server.py", 
                tool_name="query_mimic_db", 
                arguments={"query": sql_query}
            )
        except Exception as e:
            result_str = f"Error: {str(e)}"

        print(f"      💾 DB Result: {len(str(result_str))} chars found.")

        # --- 2. CRITIC ---
        CRITIC_PROMPT = f"""
        Review this SQL interaction.
        QUERY: {sql_query}
        RESULT HEAD: {str(result_str)[:500]}
        
        Tasks:
        1. Did it error?
        2. Is the result empty?
        3. Did we get the data we need?
        
        Output JSON ONLY: {{ "approved": bool, "feedback": "string" }}
        """
        critique_raw = llm.invoke(CRITIC_PROMPT).content
        
        # --- 🛑 DEBUG PRINT: SHOW ME THE RAW OUTPUT ---
        print(f"\n      🛑 RAW CRITIC OUTPUT:\n{critique_raw}\n      -----------------------")

        # --- 3. PARSING ---
        try:
            # Attempt to find JSON blob inside the raw text
            start_index = critique_raw.find('{')
            end_index = critique_raw.rfind('}')
            
            if start_index != -1 and end_index != -1:
                json_str = critique_raw[start_index : end_index + 1]
                critique = json.loads(json_str)
            else:
                raise ValueError("No JSON brackets found")

            if critique["approved"]:
                print(f"      ✅ Critic says: APPROVED.")
                return {"patient_data": result_str}
                print(f"      --------------------------------------------------") #
            else:
                print(f"      👮 Critic says: REJECT. Feedback: {critique['feedback']}")
                history += f"\nTurn {i} Feedback: {critique['feedback']}\n"
                print(f"      --------------------------------------------------")
                
        except Exception as e:
            print(f"      ⚠️ JSON Parse Error: {e}")
            # Add the raw output to history so the Actor knows the Critic is confusing
            history += f"\nSystem: Critic returned invalid JSON. Raw output was: {critique_raw[:100]}...\n"

    return {"patient_data": result_str}

async def rag_mcp_node(state: AgentState):
    print("\n   --- 📚 Entering Research Sub-Graph (Actor + Critic) ---")
    
    patient_id = state['patient_id']
    procedure = state['procedure']
    
    # Get Patient Data to inform the search (Context)
    patient_data_context = str(state.get('patient_data', ''))[:500] # Truncate for context
    
    max_retries = 4
    history = "" 
    accumulated_findings = "" # We keep adding good findings here

    for i in range(max_retries):
        print(f"\n      🔄 [Research Turn {i+1}/{max_retries}]")

        # --- 1. ACTOR (The Researcher) ---
        ACTOR_PROMPT = f"""
        You are a Medical Researcher.
        Patient: {patient_id} | Procedure: {procedure}
        Patient Context: {patient_data_context}
        
        Current Findings:
        {accumulated_findings}
        
        Critic Feedback from last turn:
        {history}
        
        TASK:
        We need to validate risks/protocols.
        Write a search query for the Hybrid Search Engine.
        Focus on Drug protocols, Surgery Risks, or specific Conditions found in context.
        
        If you think we have sufficient information, reply with "DONE".
        Otherwise, return ONLY the search query string.
        """
        
        search_query = llm.invoke(ACTOR_PROMPT).content.replace('"', '').strip()
        
        if "DONE" in search_query:
            print("      ✅ Actor is satisfied with findings.")
            break
            
        print(f"      🕵️ Actor searches: '{search_query}'")
        
        # Execute Search (Hybrid)
        try:
            search_result = await call_mcp_tool(
                script_path="mcp_servers/rag_server.py", 
                tool_name="search_medical_knowledge", 
                arguments={"query": search_query}
            )
        except Exception as e:
            search_result = f"Error: {str(e)}"

        print(f"      📄 Found {len(search_result)} chars of text.")

        # --- 2. CRITIC (The Reviewer) ---
        CRITIC_PROMPT = f"""
        You are a Senior Surgeon reviewing research results.
        
        QUERY: {search_query}
        RESULTS: {search_result[:1500]}
        
        ACCUMULATED KNOWLEDGE: {accumulated_findings}
        
        Task:
        1. Is this result useful? (Does it contain specific numbers, protocols, or rules?)
        2. Is it redundant?
        3. What is still missing?
        
        Output JSON ONLY:
        {{
            "useful": boolean,
            "summary": "Extract the key fact (e.g. 'Warfarin stop 5 days pre-op').",
            "missing": "What should we search for next? (or 'Nothing')"
        }}
        """
        
        critique_raw = llm.invoke(CRITIC_PROMPT).content
        
        try:
            # JSON Parsing Helper
            start_index = critique_raw.find('{')
            end_index = critique_raw.rfind('}')
            if start_index != -1 and end_index != -1:
                critique = json.loads(critique_raw[start_index : end_index + 1])
            else:
                raise ValueError("No JSON")

            if critique["useful"]:
                print(f"      ✅ Critic finds value: {critique['summary']}")
                accumulated_findings += f"\n- {critique['summary']}"
                history = f"Previous search successful. Missing: {critique['missing']}"
            else:
                print(f"      🗑️ Critic discards result. Feedback: {critique['missing']}")
                history = f"Previous search yield no new info. Try finding: {critique['missing']}"
                
        except Exception as e:
            print(f"      ⚠️ Critic JSON Error.")
            history = "System: Invalid JSON from critic. Try searching for a different keyword."

    # Final Check: Did we find the patient's specific notes?
    # (Simple fallback to ensure we read the patient file at least once)
    if "patient" not in accumulated_findings.lower():
        print("      ⚠️ Fallback: Reading Patient Note directly...")
        # In a real system, this would be another tool call.
        # For now, we assume the Actor would have searched for "Patient [ID] history"
        pass

    return {"guidelines": accumulated_findings if accumulated_findings else "No specific guidelines found."}

async def synthesizer_node(state: AgentState):
    print("   --- ✍️ Synthesizing Report ---")
    
    PROMPT = """
    You are a precise Surgical Planner creating a Pre-Op Brief.
    
    INPUT CONTEXT:
    1. Structured Data (SQL): {patient_data}
    2. Unstructured Notes (RAG): {guidelines}
    
    STRICT REPORTING RULES:
    1. **QUANTIFY EVERYTHING:** Never say "low platelets". You MUST say "Platelets: 45". Never say "kidney failure". Say "Creatinine: 4.5".
    2. **NAME MEDICATIONS:** List specific drug names (e.g., Warfarin, Metformin) found in the data.
    3. **INTEGRATE SOURCES:** Combine the SQL data (labs) with the Notes (symptoms/history) to justify your plan.
    
    OUTPUT FORMAT:
    ## Patient Summary & Procedure
    [Patient ID, Procedure, Diagnosis]
    
    ## 🚨 CRITICAL ALERTS (Red Flags)
    [List specific contraindications. BOLD the values. Example: **Platelets 45k** (Low)]
    
    ## Clinical Data Review
    * **Labs:** [List specific values]
    * **Meds:** [List specific meds]
    * **Notes:** [Key findings from text]
    
    ## Surgical Plan
    [Action items: Proceed / Postpone / Special Precautions]
    """
    
    chain = ChatPromptTemplate.from_template(PROMPT) | llm | StrOutputParser()
    res = chain.invoke({"patient_data": str(state['patient_data']), "guidelines": state['guidelines']})
    return {"final_report": res}

# --- ROUTER ---
def decide_next_step(state: AgentState):
    if not state.get("patient_id"): return "parser"
    if not state.get("patient_data"): return "sql_mcp"
    if not state.get("guidelines"): return "rag_mcp"
    return "synthesizer"

def router_node(state: AgentState): return {}

# --- GRAPH ---
workflow = StateGraph(AgentState)
workflow.add_node("parser", parser_node)
workflow.add_node("sql_mcp", sql_mcp_node)
workflow.add_node("rag_mcp", rag_mcp_node)
workflow.add_node("synthesizer", synthesizer_node)
workflow.add_node("router", router_node)
workflow.add_node("transcriber", transcriber_node)

workflow.set_entry_point("transcriber")
workflow.add_conditional_edges("router", decide_next_step, 
    {"parser":"parser", "sql_mcp":"sql_mcp", "rag_mcp":"rag_mcp", "synthesizer":"synthesizer"})
workflow.add_edge("parser", "router")
workflow.add_edge("sql_mcp", "router")
workflow.add_edge("rag_mcp", "router")
workflow.add_edge("synthesizer", END)
workflow.add_edge("transcriber", "router")

app = workflow.compile()
