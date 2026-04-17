import argparse
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "db")
DEFAULT_DATA_ROOT = os.getenv("KB_DATA_ROOT", "knowledge_base")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)
CHUNK_SIZE = int(os.getenv("KB_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("KB_CHUNK_OVERLAP", "200"))

SOURCE_CONFIG = {
    "docs": [
        ("**/*.md", TextLoader),
        ("**/*.txt", TextLoader),
        ("**/*.rst", TextLoader),
        ("**/*.pdf", PyPDFLoader),
    ],
    "code": [
        ("**/*.py", TextLoader),
        ("**/*.js", TextLoader),
        ("**/*.ts", TextLoader),
        ("**/*.tsx", TextLoader),
        ("**/*.java", TextLoader),
        ("**/*.go", TextLoader),
        ("**/*.rb", TextLoader),
        ("**/*.rs", TextLoader),
        ("**/*.sql", TextLoader),
        ("**/*.yaml", TextLoader),
        ("**/*.yml", TextLoader),
        ("**/*.json", TextLoader),
        ("**/*.sh", TextLoader),
    ],
    "logs": [
        ("**/*.log", TextLoader),
        ("**/*.txt", TextLoader),
        ("**/*.jsonl", TextLoader),
    ],
}


def load_source_documents(data_root: Path, source_name: str):
    source_path = data_root / source_name
    if not source_path.exists():
        return []

    docs = []
    for file_glob, loader_cls in SOURCE_CONFIG[source_name]:
        loader_kwargs = {}
        if loader_cls is TextLoader:
            loader_kwargs["encoding"] = "utf-8"
        loader = DirectoryLoader(
            str(source_path),
            glob=file_glob,
            loader_cls=loader_cls,
            loader_kwargs=loader_kwargs,
            recursive=True,
            show_progress=False,
            silent_errors=True,
        )
        docs.extend(loader.load())

    for doc in docs:
        doc.metadata["source_type"] = source_name

    return docs


def build_vector_db(data_root: str, persist_directory: str, reset: bool):
    data_root_path = Path(data_root)

    if reset and Path(persist_directory).exists():
        shutil.rmtree(persist_directory)

    documents = []
    for source_name in SOURCE_CONFIG:
        source_docs = load_source_documents(data_root_path, source_name)
        documents.extend(source_docs)
        print(f"Loaded {len(source_docs)} documents from '{source_name}'.")

    if not documents:
        raise ValueError(
            "No documents were loaded. Add files under knowledge_base/docs, "
            "knowledge_base/code, or knowledge_base/logs."
        )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(documents)
    embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=persist_directory,
    )
    print(f"Stored {len(chunks)} chunks in '{persist_directory}'.")
    vectordb = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a local-first Chroma KB for docs, code, and logs."
    )
    parser.add_argument(
        "--data-root",
        default=DEFAULT_DATA_ROOT,
        help="Root folder containing docs/, code/, and logs/ folders.",
    )
    parser.add_argument(
        "--persist-directory",
        default=PERSIST_DIRECTORY,
        help="Directory used by Chroma to persist vectors.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing vector DB before rebuilding.",
    )
    args = parser.parse_args()
    build_vector_db(args.data_root, args.persist_directory, args.reset)
