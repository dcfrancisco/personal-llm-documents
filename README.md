# personal-llm-documents

Local-first LLM knowledge base using **ChromaDB** for three data sources:
- `docs` (project/product documentation)
- `code` (source code)
- `logs` (runtime/support logs)

Embeddings are generated locally with `sentence-transformers` and stored in a local Chroma persistence directory.

## Project structure

```text
.
├── setup_vectordb.py         # Build/rebuild local Chroma KB from docs/code/logs
├── initialize_langchain.py   # Load vector DB + create QA chains
├── run_question_db.py        # CLI Q&A over local KB
├── chatbot_documents.py      # Gradio Q&A UI
└── chatbot_ui.py             # Alternate Gradio UI
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data layout

Create your local knowledge base folders:

```text
knowledge_base/
  docs/
  code/
  logs/
```

Supported defaults:
- docs: `.md`, `.txt`, `.rst`, `.pdf`
- code: `.py`, `.js`, `.ts`, `.tsx`, `.java`, `.go`, `.rb`, `.rs`, `.sql`, `.yaml`, `.yml`, `.json`, `.sh`
- logs: `.log`, `.txt`, `.jsonl`

## Build local Chroma KB

```bash
python setup_vectordb.py --reset
```

Optional args:
- `--data-root` (default: `knowledge_base`)
- `--persist-directory` (default: `db` or `CHROMA_PERSIST_DIRECTORY`)

## Query KB from CLI

```bash
python run_question_db.py --source all
```

`--source` can be: `all`, `docs`, `code`, `logs`.

## LLM behavior

- If `OPENAI_API_KEY` is set, OpenAI is used for answer generation.
- Otherwise, local Ollama is used (`LOCAL_LLM_MODEL`, default: `llama3.1`).
- Retrieval remains local-first in all cases via ChromaDB.

## Environment variables

- `CHROMA_PERSIST_DIRECTORY` (default: `db`)
- `KB_DATA_ROOT` (default: `knowledge_base`)
- `EMBEDDING_MODEL_NAME` (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `KB_CHUNK_SIZE` (default: `1000`)
- `KB_CHUNK_OVERLAP` (default: `200`)
- `KB_RETRIEVAL_K` (default: `4`)
- `LOCAL_LLM_MODEL` (default: `llama3.1`)

## License

MIT, see [LICENSE](LICENSE).
