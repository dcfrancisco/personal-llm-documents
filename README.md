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


### Here's the `requirements.txt` file that includes the necessary dependencies for the code:

```
langchain
openai
tiktoken
chromadb
gradio
```

You can save this content to a file named `requirements.txt`. To install the dependencies, you can use the following command:

```
pip install -r requirements.txt
```

### Create a Python Environment

```
python3 -m venv .venv
```

### Activate the Python Environment

```
source ./.venv/bin/activate
```

### Install the Dependencies

```
pip install -r requirements.txt
```

### Create a `.env` File

``` 
OPENAI_API_KEY=<your-openai-api-key>
```

### Initialize database 

```

```

Make sure to run this command in your Python environment to install the required packages before running the code.


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




