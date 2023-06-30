import os
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader

load_dotenv()
# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Load and process the text files
# loader = DocumentLoader('path/to/your/pdf/file.pdf')
loader = DirectoryLoader("./new_articles/", glob="./*.pdf")
# loader = DirectoryLoader("./new_articles/", glob="./*.txt", loader_cls=TextLoader)
documents = loader.load()

# Splitting the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Embed and store the texts
persist_directory = "db"
embedding = OpenAIEmbeddings()

vectordb = Chroma.from_documents(
    documents=texts, embedding=embedding, persist_directory=persist_directory
)

# Persist the database to disk
vectordb.persist()
vectordb = None
