import os
import sys

# Force standard I/O to use UTF-8 on Windows to prevent UnicodeEncodeError when printing emojis
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

from groq import Groq
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Load .env file (reads GROQ_API_KEY automatically)
load_dotenv()

PERSIST_DIR     = "db/chroma_db_free"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── Groq model (free tier: 14,400 req/day — blazing fast inference) ──────────
# Available free models:
#   "llama3-8b-8192"           → fast, great quality
#   "llama3-70b-8192"          → best quality (1,000 req/day free)
#   "mixtral-8x7b-32768"       → long context, very good
#   "gemma-7b-it"              → lightweight
GROQ_MODEL = "openai/gpt-oss-120b"

TOP_K = 5   # How many lecture records to retrieve per query


def get_groq_key():
    """Get the Groq API key from environment or prompt the user."""
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if key:
        return key

    print("\n⚠️  GROQ_API_KEY not found in your .env file.")
    print("   Get a FREE key at: https://console.groq.com/keys")
    print("   Then add this line to your .env file:")
    print("       GROQ_API_KEY=gsk_your_key_here\n")
    key = input("   Or paste your key now (for this session only): ").strip()

    if not key:
        raise ValueError("❌ No Groq API key provided. Cannot continue.")

    os.environ["GROQ_API_KEY"] = key
    return key


def load_vectorstore(embeddings):
    """Load the existing ChromaDB vector store."""
    if not os.path.exists(PERSIST_DIR):
        raise FileNotFoundError(
            f"\n❌ Database folder '{PERSIST_DIR}' does not exist.\n"
            "   Please run this command first:\n"
            "       python free_rag_ingestion.py"
        )
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )


def build_prompt(query: str, context: str) -> str:
    """Build the RAG prompt sent to the Groq LLM."""
    return f"""You are an expert educational assistant. Answer the user's question based on the lecture materials provided.

Instructions:
- Synthesize a comprehensive answer from the retrieved lecture content
- Be specific and cite which lecture(s) the information comes from
- If the exact answer isn't in the materials, say what's available and suggest related concepts
- Provide practical examples or explanations when relevant

Retrieved Lecture Content:
{context}

User Question: {query}

Answer:"""


def main():
    print("=" * 55)
    print("  Free RAG Q&A System — Groq (Fast & No Quota Issues)")
    print("=" * 55)

    # ── Step 1: Groq API key ──────────────────────────────────────────────────
    try:
        get_groq_key()
    except ValueError as e:
        print(e)
        return

    # ── Step 2: Load embedding model ─────────────────────────────────────────
    print(f"\n[1/3] Loading local HuggingFace embeddings ({EMBEDDING_MODEL})...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # ── Step 3: Connect to the vector database ────────────────────────────────
    print(f"[2/3] Connecting to ChromaDB at '{PERSIST_DIR}'...")
    try:
        db = load_vectorstore(embeddings)
    except FileNotFoundError as e:
        print(e)
        return

    count = db._collection.count()
    print(f"       Database loaded — {count} indexed chunks ready.")

    # ── Step 4: Initialise Groq SDK client ───────────────────────────────────
    print(f"\n[3/3] Connecting to Groq ({GROQ_MODEL})...")
    print("       (Free: 14,400 req/day — keys don't expire!)")
    groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
    retriever = db.as_retriever(search_kwargs={"k": TOP_K})

    print("\n✅ System ready!  Type 'quit' or 'exit' to stop.\n")
    print("-" * 55)

    while True:
        query = input("\nYour question: ").strip()
        if not query:
            continue
        if query.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        # Retrieve relevant lectures
        docs = retriever.invoke(query)
        if not docs:
            print("⚠️  No matching lectures found in the database.")
            continue

        # Show retrieved sources
        print(f"\n📚 Top {len(docs)} retrieved lectures:")
        for i, doc in enumerate(docs, 1):
            title      = doc.metadata.get("title", "Unknown")
            instructor = doc.metadata.get("instructor", "Unknown")
            venue      = doc.metadata.get("venue", "Unknown")
            year       = doc.metadata.get("year", "Unknown")
            print(f"  [{i}] \"{title}\" — {instructor} ({venue}, {year})")

        # Build context string from retrieved documents
        context = "\n".join(
            f"[{i+1}] {doc.page_content}" for i, doc in enumerate(docs)
        )

        # Ask Groq
        prompt = build_prompt(query, context)
        print(f"\n🤖 Generating answer with Groq ({GROQ_MODEL})...")
        try:
            chat_response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=512,
            )
            print("\n" + "=" * 55)
            print(chat_response.choices[0].message.content)
            print("=" * 55)
        except Exception as e:
            print(f"❌ Groq error: {repr(e)}")
            print("   Check your GROQ_API_KEY at: https://console.groq.com/keys")


if __name__ == "__main__":
    main()
