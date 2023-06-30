```

.
├── LICENSE
├── README.md
├── __pycache__
├── chatbot_documents.py
├── chatbot_ui.py
├── initialize_langchain.py
├── requirements.txt
├── run_langchain.py
└── setup_langchain.py

```


### Certainly! Here's a `requirements.txt` file that includes the necessary dependencies for the code:

```
langchain
openai
tiktoken
chromadb
psycopg2
pgvector
```

You can save this content to a file named `requirements.txt`. To install the dependencies, you can use the following command:

```
pip install -r requirements.txt
```

Make sure to run this command in your Python environment to install the required packages before running the code.


Certainly! Here's the modified code split into separate scripts:

**Python Script 1 (`setup_langchain.py`):**
```python
import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"

# Load and process the text files
loader = DirectoryLoader('./new_articles/', glob="./*.txt", loader_cls=TextLoader)
documents = loader.load()

# Splitting the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Embed and store the texts
persist_directory = 'db'
embedding = OpenAIEmbeddings()

vectordb = Chroma.from_documents(
    documents=texts,
    embedding=embedding,
    persist_directory=persist_directory
)

# Persist the database to disk
vectordb.persist()
vectordb = None
```

**Python Script 2 (`initialize_langchain.py`):**
```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# Load the persisted database from disk
persist_directory = 'db'
embedding = OpenAIEmbeddings()

vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)

# Make a retriever
retriever = vectordb.as_retriever()

# Create the chain to answer questions
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Define the function to process and display results
def process_llm_response(llm_response):
    print(llm_response['result'])
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])
```

**Python Script 3 (`run_langchain.py`):**
```python
from initialize_langchain import qa_chain, process_llm_response

# Prompt for query and display response
while True:
    query = input("Enter your question (or '/stop' to quit): ")
    if query.lower() == '/stop':
        break
    
    llm_response = qa_chain(query)
    process_llm_response(llm_response)
```

To run the code, follow these steps:

1. Execute `setup_langchain.py`:
   ```
   python setup_langchain.py
   ```

2. Execute `initialize_langchain.py`:
   ```
   python initialize_langchain.py
   ```

3. Execute `run_langchain.py`:
   ```
   python run_langchain.py
   ```

The `setup_langchain.py` script sets up the database by processing the text files, embedding them, and persisting the database to disk.

The `initialize_langchain.py` script initializes the language model chain and retriever using the persisted database.

The `run_langchain.py` script prompts the user to enter a question. It will continue prompting until the user enters '/stop'. For each question, it retrieves the answer using the language model chain and displays the result and source documents.

Note: Make sure all three scripts are in the same directory, and the necessary dependencies are installed (`langchain`, `openai`, `tiktoken`, and `chromadb`).


## License

This project is licensed under the [MIT License](LICENSE).