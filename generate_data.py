import sqlite3
import os
import json
from gtts import gTTS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Initialize Llama 3.3 for generating synthetic notes
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

DB_NAME = "mimic_demo.db"

def setup_database():
    """Creates the database with a REALISTIC MIMIC-IV Schema."""
    if os.path.exists(DB_NAME):
        os.remove(DB_NAME)
    
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # 1. DIAGNOSES_ICD
    c.execute('''CREATE TABLE diagnoses_icd (
        subject_id INT,
        icd_code TEXT,
        long_title TEXT
    )''')

    # 2. LABEVENTS (The critical table)
    c.execute('''CREATE TABLE labevents (
        subject_id INT,
        itemid INT,
        charttime TEXT,
        valuenum REAL,
        valueuom TEXT,
        flag TEXT -- 'abnormal' or null
    )''')

    # 3. PRESCRIPTIONS
    c.execute('''CREATE TABLE prescriptions (
        subject_id INT,
        drug TEXT,
        dose_val_rx TEXT,
        dose_unit_rx TEXT,
        route TEXT
    )''')

    # 4. D_LABITEMS (Dictionary)
    c.execute('''CREATE TABLE d_labitems (
        itemid INT,
        label TEXT,
        fluid TEXT,
        category TEXT
    )''')
    
    # Insert Dictionary Data (Standard MIMIC Codes)
    c.execute("INSERT INTO d_labitems VALUES (51265, 'Platelet Count', 'Blood', 'Hematology')")
    c.execute("INSERT INTO d_labitems VALUES (50912, 'Creatinine', 'Blood', 'Chemistry')")
    c.execute("INSERT INTO d_labitems VALUES (51214, 'Fibrinogen', 'Blood', 'Hematology')")
    c.execute("INSERT INTO d_labitems VALUES (50811, 'Hemoglobin', 'Blood', 'Hematology')")
    
    conn.commit()
    conn.close()
    print("✅ Database Schema Created.")

def generate_patient_data():
    """Injects 3 High-Risk Personas into the DB."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    patients = [
        # PERSONA 1: The "Bleeder" (Cholecystectomy candidate)
        {
            "id": 10001,
            "condition": "Acute Cholecystitis",
            "risks": "History of DVT, on Warfarin, Thrombocytopenia (Low Platelets)",
            "labs": [
                (51265, 45, 'K/uL', 'abnormal'), # Platelets VERY LOW (Normal 150)
                (50912, 0.9, 'mg/dL', None),     # Creatinine Normal
                (51214, 120, 'mg/dL', 'abnormal') # Fibrinogen Low
            ],
            "meds": [("Warfarin", "5mg", "PO"), ("Oxycodone", "5mg", "PO")],
            "note_prompt": "Write a frantic nursing note for Pt 10001. Mention he has large bruising on his arms and mentioned he skipped his last clotting test. He is scheduled for gallbladder surgery."
        },
        # PERSONA 2: The "Renal Risk" (Appendectomy candidate)
        {
            "id": 10002,
            "condition": "Acute Appendicitis",
            "risks": "End Stage Renal Disease, Missed Dialysis",
            "labs": [
                (51265, 200, 'K/uL', None),      # Platelets Normal
                (50912, 4.5, 'mg/dL', 'abnormal'), # Creatinine CRITICAL (Kidney Fail)
                (50811, 8.2, 'g/dL', 'abnormal')   # Hemoglobin Low (Anemia)
            ],
            "meds": [("Metoprolol", "25mg", "PO"), ("PhosLo", "667mg", "PO")],
            "note_prompt": "Write a triage note for Pt 10002. He missed dialysis yesterday. He feels short of breath and 'puffy'. He needs surgery for appendicitis."
        }
    ]

    for p in patients:
        print(f"   --- Generating Synthetic Data for Patient {p['id']} ---")
        
        # 1. SQL Data
        c.execute("INSERT INTO diagnoses_icd VALUES (?, ?, ?)", (p['id'], 'K81.0', p['condition']))
        for itemid, val, uom, flag in p['labs']:
            c.execute("INSERT INTO labevents VALUES (?, ?, ?, ?, ?, ?)", 
                      (p['id'], itemid, '2025-11-23 08:00', val, uom, flag))
        for drug, dose, route in p['meds']:
            c.execute("INSERT INTO prescriptions VALUES (?, ?, ?, ?, ?)", 
                      (p['id'], drug, dose, 'mg', route))
        
        # 2. Unstructured Note (RAG Data)
        print(f"      Writing Note with Llama 3...")
        note_content = llm.invoke(p['note_prompt']).content
        with open(f"patient_{p['id']}_notes.txt", "w") as f:
            f.write(note_content)
            
        # 3. Audio Input (The Surgeon's Voice)
        # Using gTTS (Google Text-to-Speech)
        print(f"      Recording Audio...")
        text = f"I need a pre-op brief for Patient {p['id']}. Pay close attention to his risks."
        tts = gTTS(text=text, lang='en')
        tts.save(f"input_{p['id']}.mp3")

    conn.commit()
    conn.close()
    print("✅ Synthetic Patient Factory Finished!")

if __name__ == "__main__":
    setup_database()
    generate_patient_data()
