from dotenv import load_dotenv
import streamlit as st
import zipfile
from langchain_community.vectorstores import Chroma
from huggingface_hub import hf_hub_download
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import concurrent.futures
import os
import chromadb

load_dotenv()

# Function to get API token, checking in both os.getenv and st.secrets
def get_token(token_name):
    # First, check in environment variables
    token = os.getenv(token_name)
    if token:
        return token
    else:
        # If not found in environment variables, check Streamlit secrets
        if token_name in st.secrets:
            return st.secrets[token_name]
        else:
            raise ValueError(f"API token for '{token_name}' not found in environment variables or Streamlit secrets.")


@st.cache_resource
def download_and_extract_vector_db():
    # Hugging Face Space URL
    zip_file_path = hf_hub_download(repo_id="dummyUsrere/Pediatrics_CHAT", filename="bookVectorStore.zip")

    # Path where we want to extract the files
    extract_path = '/mnt/data/chroma_db'

    # Extract the ZIP file
    if not os.path.exists(extract_path):
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("Database extracted successfully!")
    
    return extract_path


# Asynchronous loading of the vector store
@st.cache_resource
def load_vector_db():
    extract_path = download_and_extract_vector_db()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(load_chroma_vector_store, extract_path)
        return future.result()

# Load vector database
@st.cache_resource
def load_CHROMA():
    # vectorstore_dir = download_and_extract_vectorstore()
    # print(f"Vector store is available at: {vectorstore_dir}")
    return Chroma(
        collection_name='bookVectors',
        persist_directory='./bookVectorStore_ole',
        embedding_function=HuggingFaceEmbeddings()
    )

def load_chroma_vector_store(extract_path):
    client = chromadb.Client()
    
    # Create or get the collection from the Chroma client
    collection = client.get_or_create_collection(name="bookVectors")
    
    # Load the vector store from the extracted directory
    vector_db = Chroma(persist_directory=extract_path, collection_name="bookVectors")
    return vector_db

# Set up chains
def setup_rag_chain(vector_db):
    bookVSS_ret = vector_db.as_retriever()
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. You are trained on major books of pediatrics; therefore, when a question is posted to you, "
        "you should search the details about the problem from the dataset and provide the information in a concise, "
        "informative format. "
        "If you don't know the answer, say that you don't know. "
        "Use points to structure the content and keep the answer concise."
        "\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ('system', system_prompt),
        ('human', "{input}")
    ])
    llm = ChatGroq(model='llama3-8b-8192')
    qa_chain = RetrievalQA.from_chain_type(
        retriever=bookVSS_ret,
        llm=llm,
        return_source_documents=True
    )
    return qa_chain

# Streamlit App
st.set_page_config(page_title="Pediatrics Q&A", layout="centered")
st.title("Pediatrics Chat Assistant")
st.write("Ask questions about pediatrics-related topics.")

# Load vector store and chains
vector_db = load_vector_db()
rag_chain = setup_rag_chain(vector_db)

# Input from user
user_question = st.chat_input("Type your question here...")
if user_question:
    with st.spinner("Fetching response..."):
        response = rag_chain({"query": user_question})
        st.markdown(f'**Question:** {user_question}')
        st.markdown(f"**Answer:** {response['result']}")

        # Display sources
        st.markdown("### Sources")
        if response["source_documents"]:
            for doc in response["source_documents"]:
                source = doc.metadata.get('source', 'Unknown Source')
                page = doc.metadata.get('page', 'N/A')
                st.markdown(f"- **Source:** {source}, **Page:** {page}")
        else:
            st.markdown("No source documents were found.")
