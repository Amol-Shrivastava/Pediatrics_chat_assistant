from dotenv import load_dotenv
import streamlit as st
import zipfile
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import concurrent.futures
import os
import requests
import shutil

load_dotenv()

VECTORSTORE_URL = os.getenv("DROP_BOX_LINK")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3-8b-8192")


@st.cache_resource
# Function to download the vector store

def download_and_extract_vectorstore():
    vectorstore_path = "./bookVectorStore"  # This is where you want the final structure
    zip_path = "bookVectorStore.zip"  # Temporary zip file

    if not os.path.exists(vectorstore_path):
        os.makedirs(vectorstore_path, exist_ok=True)

        # Dropbox direct download link (replace this with your Dropbox link)
        dropbox_url = VECTORSTORE_URL

        # Step 1: Download the ZIP file
        print("Downloading vector store ZIP from Dropbox...")
        response = requests.get(dropbox_url, stream=True)
        if response.status_code == 200:
            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            print("Download complete.")
        else:
            print(f"Error downloading file: {response.status_code}")
            raise Exception("Failed to download vector store.")

        # Step 2: Extract the ZIP file
        print("Extracting ZIP file...")
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                temp_extract_path = "./temp_bookVectorStore"  # Temporary path for extraction
                zip_ref.extractall(temp_extract_path)

                # Move the extracted folder and chroma.sqlite3 to the final folder
                for item in os.listdir(temp_extract_path):
                    source = os.path.join(temp_extract_path, item)
                    destination = os.path.join(vectorstore_path, item)

                    if os.path.isdir(source):
                        # Move the directory (e.g., e3d6838b-...)
                        shutil.move(source, destination)
                    elif item == "chroma.sqlite3":
                        # Move the chroma.sqlite3 file
                        shutil.move(source, destination)

                # Remove the temporary extraction folder
                shutil.rmtree(temp_extract_path)
            print("Extraction complete.")
        except zipfile.BadZipFile:
            print("Error: The downloaded file is not a valid ZIP file.")
            raise
        finally:
            os.remove(zip_path)  # Cleanup the ZIP file
    else:
        print("Vector store already exists locally.")

    return vectorstore_path


# Asynchronous loading of the vector store
@st.cache_resource
def load_vector_db():
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(load_CHROMA)
        return future.result()

# Load vector database
@st.cache_resource
def load_CHROMA():
    vectorstore_dir = download_and_extract_vectorstore()
    print(f"Vector store is available at: {vectorstore_dir}")
    return Chroma(
        collection_name='bookVectors',
        persist_directory="./bookVectorStore/bookVectorStore",
        embedding_function=HuggingFaceEmbeddings()
    )

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
    llm = ChatGroq(model=MODEL_NAME)
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
