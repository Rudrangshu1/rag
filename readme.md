

# RAG Demo: Document Q\&A with Milvus + OpenAI

This project is a **Retrieval-Augmented Generation (RAG)** demo built with:

* **[Streamlit](https://streamlit.io/)** for the interactive web UI
* **[LangChain](https://www.langchain.com/)** for document processing and conversational AI
* **[Milvus Lite](https://milvus.io/)** as the vector database (stored locally in a `.db` file)
* **[OpenAI](https://platform.openai.com/)** for embeddings and LLM responses

It lets you:

1. Upload documents (`.pdf`, `.txt`, `.docx`)
2. Store embeddings in **Milvus**
3. Chat with your documents using an **OpenAI GPT model**

---

##  Features

* Upload multiple documents at once
* Automatic chunking of documents for efficient retrieval
* Local storage using **Milvus Lite** (no external DB needed)
* Conversational memory for contextual Q\&A
* Chat interface built with Streamlit

---

## Installation

Clone the repo and install dependencies:



Install required Python packages:

```bash
pip install -r requirements.txt
```

---

##  Setup OpenAI API Key

Need to Set the OpenAI API key in the code or as an environment variable:

```bash
export OPENAI_API_KEY="your_api_key_here"   # Linux/Mac
setx OPENAI_API_KEY "your_api_key_here"     # Windows (PowerShell)
```

Alternatively, we can edit the script and paste the key here:

```python
OPENAI_API_KEY = "your_api_key_here"
```

---

##  Run the App

Start the Streamlit app:

```bash
streamlit run app.py
```



##  How It Works

1. **Upload Documents**

   * Supported: `.pdf`, `.txt`, `.docx`
   * Files are parsed into text using appropriate loaders

2. **Process Documents**

   * Documents are split into chunks (`chunk_size=1000`, `overlap=200`)
   * Embeddings are created with **OpenAIEmbeddings**
   * Data is stored in **Milvus Lite** (local file `milvus_demo.db`)

3. **Chat with Docs**

   * Type a question in the input box
   * A retrieval + generation pipeline fetches relevant chunks
   * OpenAI GPT (`gpt-4o-mini` in this demo) generates answers
   * Chat history is preserved in the session

---

##  Example Usage

* Upload a sampledoc.csv
* Ask: *"What are your working hours?"*
* Bot responds with Our working hours are 9 AM to 6 PM, Monday to Friday.

---
