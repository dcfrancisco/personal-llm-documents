## Overview
This project demonstrates how to create an intelligent Q&A Chatbot by combining advanced AI technologies. Our Chatbot is designed to understand, learn from, and generate language-based responses. It leverages LangChain for processing and understanding text, OpenAI Embeddings for language understanding and generation, Chroma for effective data storage and retrieval, and Gradio for an interactive user interface.

```

.
├── LICENSE
├── README.md
├── chatbot_documents.py        # Script for processing and embedding text documents
├── initialize_langchain.py     # Script for initializing the language model chain and retriever
├── requirements.txt            # File containing the necessary dependencies for the code
├── run_langchain.py            # Script for running the Q&A Chatbot
└── setup_langchain.py          # Script for setting up the database
```

```
## Features
- **Natural Language Processing (NLP)**: The application uses NLP techniques for understanding and generating human-like text responses.
- **Large Language Models (LLM)**: Harnessing the power of LLMs to improve response generation.
- **OpenAI Embeddings**: The application uses OpenAI Embeddings for efficient language representation and understanding.
- **Vector Databases**: It leverages Chroma for efficient storage and retrieval of vector data.
- **Gradio Interface**: The application provides a user-friendly interface using Gradio, allowing users to easily interact with the chatbot.

## Usage
To experience our Q&A Chatbot Demo, simply type a question into the prompt, and our Chatbot will generate a detailed, contextually accurate response.


## Installation
To run the code, you'll need to install the necessary dependencies. You can install them using the following command:


### Here's a `requirements.txt` file that includes the necessary dependencies for the code:

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

### Create a Python Environment

```
python3 -m venv venv
```

### Activate the Python Environment

```
source venv/bin/activate
```

### Install the Dependencies

```
pip install -r requirements.txt
```

### Create a `.env` File

``` 
OPENAI_API_KEY=<your-openai-api-key>
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
openAiApiKey = os.getenv("OPENAI_API_KEY")

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

## Chatbot

`![Chatbot](./assets/gradio-chatbot.png | width=250)`

  - ![](https://gyazo.com/eb5c5741b6a9a16c692170a41a49c858.png | width=100)
- `![](https://gyazo.com/eb5c5741b6a9a16c692170a41a49c858.png | width=100)`


## Future Work
We continue to explore ways to improve the performance and user experience of our chatbot. Contributions and suggestions are welcomed!



## License

This project is licensed under the [MIT License](LICENSE).




