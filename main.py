import asyncio
import os
from graph import app

async def main():
    # 1. Define Inputs
    audio_file = "input_10001.mp3"
    
    # Check if file exists (Debug Step)
    if os.path.exists(audio_file):
        print(f"✅ Audio file found: {audio_file}")
        inputs = {
            "query": "Pre-op brief for patient 10001.", 
            "audio_path": audio_file 
        }
    else:
        print(f"❌ Audio file NOT found at: {os.getcwd()}/{audio_file}")
        print("   (Running text-only mode)")
        inputs = {
            "query": "Pre-op brief for patient 10001.", 
            "audio_path": None
        }

    print(f"🤖 USER QUERY: {inputs['query']}\n")

    # 2. Run the Graph ONCE
    # We use ainvoke, which runs the whole flow and returns the final state
    final_state = await app.ainvoke(inputs)

    # 3. Print Result
    print("\n" + "="*50)
    print("📋 FINAL REPORT (Generated via MCP Architecture)")
    print("="*50)
    print(final_state["final_report"])

if __name__ == "__main__":
    asyncio.run(main())
