import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_classic.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_openai import OpenAI

load_dotenv()

PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "db")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "llama3.1")
RETRIEVAL_K = int(os.getenv("KB_RETRIEVAL_K", "4"))


@lru_cache(maxsize=1)
def get_embedding():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_vectordb():
    return Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=get_embedding(),
    )


def get_llm():
    if os.getenv("OPENAI_API_KEY"):
        return OpenAI()
    return Ollama(model=LOCAL_LLM_MODEL)


def build_retriever(source_type=None):
    search_kwargs = {"k": RETRIEVAL_K}
    if source_type and source_type != "all":
        search_kwargs["filter"] = {"source_type": source_type}
    return get_vectordb().as_retriever(search_kwargs=search_kwargs)


def create_qa_chain(source_type=None):
    return RetrievalQA.from_chain_type(
        llm=get_llm(),
        chain_type="stuff",
        retriever=build_retriever(source_type=source_type),
        return_source_documents=True,
    )



# Define the function to process and display results
def process_llm_response(llm_response):
    print(llm_response["result"])
    print("\n\nSources:")
    for source in llm_response["source_documents"]:
        print(source.metadata["source"])


def display_llm_response(llm_response):
    print("Custom Result Processing:")
    print("Answer:", llm_response["result"])
    print("Sources:")
    for source in llm_response["source_documents"]:
        print("- Source:", source.metadata["source"])
        print("- Score:", source.score)


def format_llm_response(llm_response):
    result = llm_response["result"]
    sources = [source.metadata["source"] for source in llm_response["source_documents"]]

    # Format the response string
    response = f"Result: \n{result}\n\nSources:\n"
    for source in sources:
        response += f"- {source}\n"

    return response


def html_llm_response(llm_response):
    result = llm_response["result"]
    sources = [source.metadata["source"] for source in llm_response["source_documents"]]

    # Format the response string with HTML tags
    response = f"<h3>Result:</h3><p>{result}</p><h3>Sources:</h3>"
    for source in sources:
        response += f"<li>{source}</li>"

    return response
