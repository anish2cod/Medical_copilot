import asyncio
import json
from langchain_groq import ChatGroq
from graph import app  # Import your compiled LangGraph app

# Initialize the Judge
judge_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

test_cases = [
    {
        "patient_id": "10001",
        "query": "Pre-op brief for patient 10001",
        "golden_facts": ["Platelets 45", "Warfarin", "Bruising", "Postpone surgery"]
    },
    {
        "patient_id": "10002",
        "query": "Pre-op brief for patient 10002",
        "golden_facts": ["Creatinine 4.5", "Missed Dialysis", "Shortness of Breath"]
    }
]

async def run_evaluation():
    print("--- 📉 STARTING EVALUATION ---")

    for test in test_cases:
        print(f"\nTesting Case: Patient {test['patient_id']}")
        
        # 1. Run Your Agent (USING ASYNC AINVOKE)
        # We pass a dummy audio_path or None if testing text-only
        inputs = {"query": test['query'], "audio_path": None} 
        
        try:
            result = await app.ainvoke(inputs)
            generated_report = result['final_report']
            
            print(f"   ✅ Agent Generated Report ({len(generated_report)} chars)")
            print(f"   report: { generated_report } ")

            # 2. Run the Judge (LLM-as-a-Judge)
            EVAL_PROMPT = f"""
            You are a Senior Surgeon evaluating a Junior Doctor's report.
            
            Required Facts that MUST be in the report: {test['golden_facts']}
            
            The Junior Doctor's Report:
            "{generated_report}"
            
            Task:
            Check if ALL required facts are present.
            Score: 0 to 10. (10 = Perfect, 0 = Missing critical info)
            Critique: Explain specifically what was missed.
            
            Output JSON ONLY: {{ "score": X, "critique": "..." }}
            """
            
            # The Judge can remain synchronous (invoke) or be async (ainvoke)
            eval_result = judge_llm.invoke(EVAL_PROMPT).content
            
            # Clean up JSON format if needed
            clean_json = eval_result.replace("```json", "").replace("```", "").strip()
            print(f"   👮 Judge's Verdict: {clean_json}")
            
        except Exception as e:
            print(f"   ❌ Error running test case: {e}")

if __name__ == "__main__":
    asyncio.run(run_evaluation())
