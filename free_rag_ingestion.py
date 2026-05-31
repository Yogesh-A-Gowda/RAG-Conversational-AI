import os
import sys

# Force standard I/O to use UTF-8 on Windows to prevent UnicodeEncodeError when printing emojis
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

import csv
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader

TSV_PATH = "LectureBank/alldata.tsv"
PDF_FOLDER = "LectureBank"
PERSIST_DIR = "db/chroma_db_free"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_ROWS = 500


def load_pdf_documents(pdf_folder=PDF_FOLDER):
    """Load PDF files from the LectureBank folder and extract text."""
    documents = []
    pdf_files = list(Path(pdf_folder).glob("**/*.pdf"))

    if not pdf_files:
        print(f"⚠️  No PDF files found in '{pdf_folder}'")
        return documents

    print(f"📄 Found {len(pdf_files)} PDF files. Extracting text...")

    for pdf_path in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_path))
            docs = loader.load()

            for doc in docs:
                doc.metadata.update({
                    "source_file": pdf_path.name,
                    "source_type": "pdf",
                })
            documents.extend(docs)
            print(f"   ✓ Loaded: {pdf_path.name}")
        except Exception as e:
            print(f"   ✗ Error loading {pdf_path.name}: {e}")

    print(f"✅ Extracted {len(documents)} pages from PDFs")
    return documents


def load_lecturebank_data(tsv_path=TSV_PATH):
    """Parse the LectureBank TSV and combine with PDF content if available."""
    print(f"Loading LectureBank dataset from: {tsv_path}")

    if not os.path.exists(tsv_path):
        print(f"⚠️  TSV file not found: {tsv_path}")
        print("   Attempting to load PDFs from folder instead...")
        return load_pdf_documents()

    documents = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for i, row in enumerate(reader):
            if i >= MAX_ROWS:
                break

            content = (
                f"Lecture: {row['Title']}\n"
                f"Instructor: {row['Instructor']}\n"
                f"University: {row['Venue']}\n"
                f"Year: {row['Year']}\n"
                f"Topic: {row['Topic']}\n"
                f"Slides: {row['URL']}"
            )

            metadata = {
                "id": row["ID"],
                "instructor": row["Instructor"],
                "title": row["Title"],
                "venue": row["Venue"],
                "url": row["URL"],
                "topic": row["Topic"],
                "year": row["Year"],
                "source": tsv_path,
                "source_type": "metadata",
            }

            documents.append(Document(page_content=content, metadata=metadata))

    print(f"✅ Loaded {len(documents)} lecture records from TSV")

    pdf_docs = load_pdf_documents()
    documents.extend(pdf_docs)

    return documents


def main():
    print("=" * 55)
    print("  Free RAG Ingestion Pipeline — LectureBank Dataset")
    print("=" * 55)

    # ── Guard: skip if DB already exists ─────────────────────────────────────
    if os.path.exists(PERSIST_DIR):
        print(f"\n✅ Vector store already exists at '{PERSIST_DIR}'.")
        print("   Delete that folder if you want to re-ingest the data.")
        return

    # ── Step 1: Load raw data ─────────────────────────────────────────────────
    print(f"\n[1/4] Loading data (up to {MAX_ROWS} rows)...")
    try:
        documents = load_lecturebank_data()
    except FileNotFoundError as e:
        print(e)
        return

    # ── Step 2: Split into chunks ─────────────────────────────────────────────
    print("\n[2/4] Splitting documents into smaller chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    print(f"       Created {len(chunks)} chunks from {len(documents)} documents.")

    # ── Step 3: Load the local embedding model ────────────────────────────────
    print("\n[3/4] Loading HuggingFace embedding model...")
    print(f"       Model: {EMBEDDING_MODEL}")
    print("       (Downloads ~90 MB on first run, then cached locally — no API key needed)")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # ── Step 4: Build and persist the ChromaDB vector store ──────────────────
    print(f"\n[4/4] Building ChromaDB vector store at '{PERSIST_DIR}'...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        collection_metadata={"hnsw:space": "cosine"},
    )

    print("\n✅ Ingestion complete! The database is ready at:", PERSIST_DIR)
    print("   You can now run:  python free_rag_qa.py")


if __name__ == "__main__":
    main()
