# mcp_servers/rag_server.py
from mcp.server.fastmcp import FastMCP
import os
from rank_bm25 import BM25Okapi
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

mcp = FastMCP("ClinicalKnowledgeRAG")

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "../knowledge_docs") # Local Cheat Sheets
PERSIST_DIR = os.path.join(BASE_DIR, "../chroma_db")       # Colab PubMed DB

# Must match the model used in Colab!
embeddings = HuggingFaceEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")

# --- GLOBAL SEARCH INDICES ---
vector_db = None
bm25_index = None
all_documents = [] # Holds text for BM25

def initialize_system():
    """
    1. Loads the Persistent Vector DB (PubMed).
    2. Loads Local Text Files (Cheat Sheets).
    3. Builds an in-memory BM25 Keyword Index for EVERYTHING.
    """
    global vector_db, bm25_index, all_documents
    
    print("   📚 Initializing Hybrid Knowledge Base...")
    
    # 1. Load Vector DB (PubMed Data)
    if os.path.exists(PERSIST_DIR):
        vector_db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
        print("      ✅ Loaded Persistent PubMed DB.")
        
        # Fetch all docs from Vector DB to build BM25 index (Slow but necessary for true Hybrid)
        # Note: For huge DBs, we would optimize this. For <1000 docs, it's fine.
        existing_docs = vector_db.get() # Gets all IDs and Documents
        for txt, meta in zip(existing_docs['documents'], existing_docs['metadatas']):
            all_documents.append(Document(page_content=txt, metadata=meta))
    else:
        print("      ⚠️ No Persistent DB found. Vector search will be empty.")
        # Create ephemeral DB if missing
        vector_db = Chroma(embedding_function=embeddings)

    # 2. Load & Index Local "Cheat Sheets" (Priority Data)
    if os.path.exists(KNOWLEDGE_DIR):
        loader = DirectoryLoader(KNOWLEDGE_DIR, glob="**/*.txt", loader_cls=TextLoader)
        local_docs = loader.load()
        if local_docs:
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            split_local = splitter.split_documents(local_docs)
            
            # Add to Vector DB (Ephemeral addition for this session)
            vector_db.add_documents(split_local)
            
            # Add to Document List for BM25
            all_documents.extend(split_local)
            print(f"      ✅ Loaded {len(split_local)} chunks from Local Knowledge Docs.")

    # 3. Build BM25 Index
    print(f"      🧠 Building BM25 Keyword Index for {len(all_documents)} documents...")
    tokenized_corpus = [doc.page_content.lower().split(" ") for doc in all_documents]
    bm25_index = BM25Okapi(tokenized_corpus)
    print("      ✅ Hybrid System Ready.")

# Run initialization
initialize_system()

@mcp.tool()
def search_medical_knowledge(query: str, patient_context: str = "") -> str:
    """
    Performs HYBRID SEARCH (Vector + Keyword).
    
    Args:
        query: The medical question (e.g. "Warfarin protocol")
        patient_context: Optional text from patient notes to re-rank relevance.
    """
    results = []
    seen_content = set()
    
    print(f"   🔎 Searching for: '{query}'")

    # --- A. KEYWORD SEARCH (BM25) - Precision ---
    # Good for exact drug names like "Warfarin"
    tokenized_query = query.lower().split(" ")
    # Get top 5 keyword matches
    keyword_docs = bm25_index.get_top_n(tokenized_query, all_documents, n=5)
    
    for doc in keyword_docs:
        if doc.page_content not in seen_content:
            results.append(f"[MATCH: KEYWORD] (Source: {doc.metadata.get('source', 'Unknown')})\n{doc.page_content}")
            seen_content.add(doc.page_content)

    # --- B. VECTOR SEARCH (Semantic) - Concepts ---
    # Good for "Renal failure" matching "Kidney disease"
    vector_docs = vector_db.similarity_search(query, k=5)
    
    for doc in vector_docs:
        if doc.page_content not in seen_content:
            results.append(f"[MATCH: VECTOR] (Source: {doc.metadata.get('source', 'Unknown')})\n{doc.page_content}")
            seen_content.add(doc.page_content)

    # --- C. PATIENT NOTE SEARCH (If Context Provided) ---
    # Quick grep on patient note files if they exist
    # (We assume the Client passes the patient_id in the query or context if needed)
    
    return "\n\n".join(results)

if __name__ == "__main__":
    mcp.run()
