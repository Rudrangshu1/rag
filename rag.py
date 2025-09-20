import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader

# from langchain_community.vectorstores import Milvus
from langchain_milvus import Milvus
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import tempfile
import os

OPENAI_API_KEY = ""


import os

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# --- Streamlit UI ---
st.set_page_config(page_title="RAG with Milvus + OpenAI", layout="wide")
st.title("📚 RAG Demo: Upload Docs + Milvus + OpenAI")

# Sidebar for document upload
uploaded_files = st.sidebar.file_uploader(
    "Upload Documents", accept_multiple_files=True, type=["pdf", "txt", "docx", "csv"]
)
process_btn = st.sidebar.button("Process Documents")

# Milvus Lite config (local storage)
MILVUS_URI = "https://in03-b078dd0a4e249e4.serverless.aws-eu-central-1.cloud.zilliz.com" 
MILVUS_TOEKN = "b30183989326bbdde6b40295afd2d905cb4b49347604cc303d2e993aea44821e5a9a7c93186d61629132785bbf3267f8793b98b8"
COLLECTION_NAME = "rag_docs"

# Initialize embeddings
embeddings = OpenAIEmbeddings()

if process_btn and uploaded_files:
    docs = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(tmp_path)
        elif uploaded_file.name.endswith(".txt"):
            loader = TextLoader(tmp_path)
        elif uploaded_file.name.endswith(".csv"):
            loader = TextLoader(tmp_path)        
        elif uploaded_file.name.endswith(".docx"):
            loader = Docx2txtLoader(tmp_path)
        else:
            st.warning(f"Unsupported file: {uploaded_file.name}")
            continue

        docs.extend(loader.load())
        os.remove(tmp_path)

    st.success(f"Loaded {len(docs)} documents")

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)

    # Store in Milvus
    vectorstore = Milvus.from_documents(
        documents=splits,
        embedding=embeddings,
        connection_args={"uri": MILVUS_URI, "token": MILVUS_TOEKN, "db_name": "milvus_demo"},
        drop_old=True,  # clear old data if exists
    )

    st.session_state.vs = vectorstore
    st.success("Documents indexed in Milvus ✅")

# --- Chat Interface ---
if "vs" in st.session_state:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=st.session_state.vs.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
    )

    # Chat UI
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Ask a question about your docs:")

    if user_input:
        response = qa_chain.run(user_input)
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", response))

    for speaker, msg in st.session_state.chat_history:
        st.markdown(f"**{speaker}:** {msg}")
else:
    st.info("👈 Upload and process documents first.")
 
