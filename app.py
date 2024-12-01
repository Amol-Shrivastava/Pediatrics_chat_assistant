from dotenv import load_dotenv
from langsmith import expect
import streamlit as st
import zipfile
from langchain_community.vectorstores import Chroma
from langchain import hub
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts.chat import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain_core.messages import SystemMessage
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import concurrent.futures
import os
import chromadb
import requests

#for tools and agents
from langchain_community.tools import  WikipediaQueryRun, DuckDuckGoSearchRun, PubmedQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import AgentExecutor, create_tool_calling_agent
import certifi
import ssl


load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv('groq_api_key')

## fix for ssl issue
# Apply certifi's certificates for global SSL verification
ssl._create_default_https_context = ssl.create_default_context
ssl_context = ssl.create_default_context(cafile=certifi.where())

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
    # Correct URL for the file hosted in your Hugging Face Space
    url = "https://huggingface.co/spaces/dummyUsrere/Pediatrics_CHAT/resolve/main/bookVectorStore.zip"
    
    # Local file paths - use a directory within the current project folder
    zip_file_path = './data/bookVectorStore.zip'
    extract_path = './data/bookVectorStore'

    # Create the directory if it doesn't exist
    os.makedirs('./data', exist_ok=True)
    
    # Download the ZIP file from Hugging Face Space
    response = requests.get(url)
    with open(zip_file_path, 'wb') as f:
        f.write(response.content)
    
    # Extract the ZIP file if not already extracted
    if not os.path.exists(extract_path):
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("Database extracted successfully!")
    
    return extract_path

# Asynchronous loading of the vector store
@st.cache_resource
def load_vector_db():
    extract_path = download_and_extract_vector_db()
    extract_path += '/bookVectorStore'
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(load_chroma_vector_store, extract_path)
        return future.result()

#LOAD CHROMA VECTOR STORE
def load_chroma_vector_store(extract_path):
    client = chromadb.Client()
    
    # Create or get the collection from the Chroma client
    collection = client.get_or_create_collection(name="bookVectors")
    
    # Load the vector store from the extracted directory
    vector_db = Chroma(persist_directory=extract_path, collection_name="bookVectors",embedding_function=HuggingFaceEmbeddings())
    return vector_db

# Load all the tools and create agents from it
def load_tools():
    api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=5, doc_content_chars_max=6000)
    wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

    # retriever_tool = create_retriever_tool(retriever=retriever, name='Existing dataset search', description='Search data from the existing pdf dataset.')

    search=DuckDuckGoSearchRun()
        
    # yt_tool = YouTubeSearchTool()

    pubmed_tool = PubmedQueryRun()

    return [wiki_tool, search , pubmed_tool]

def load_agents(llm):
    toolkit = load_tools()
    
    # Define the custom prompt
    custom_prompt = PromptTemplate(
        input_variables=["input", "agent_scratchpad"],
        template=(
                       
                    "You are a highly intelligent and resourceful medical assistant, trained to assist both new and experienced doctors. "
        "You have access to reliable tools such as Wikipedia for general knowledge and PubMed for scholarly research. "
        "You can also find any youtube video for the query."
        "Your task is to provide a detailed, evidence-based, and structured response to any medical query. "
        "Your response should balance clarity for less experienced doctors and depth for specialists."

        "Use the following structure for your response:"

            "---"

            "1. **Introduction**:  "
            "- Provide a concise overview of the medical concept, condition, or topic.  "
        " - Explain its significance, prevalence, and relevance in clinical practice.  "

            "2. **Epidemiology**:  "
            "- Describe the global and regional prevalence of the condition.  "
            "- Include age, gender, or demographic groups most affected.  "

            "3. **Symptoms and Clinical Presentation**:"  
            "- Detail the primary and secondary symptoms. " 
            "- Highlight variations in presentation based on patient demographics or disease stages."  

            "4. **Etiology and Risk Factors**:"  
            "- Explain the underlying causes, including genetic, environmental, infectious, or lifestyle-related factors. " 
        " - Highlight specific risk factors, including comorbidities and predispositions.  "

            "5. **Pathophysiology**:  "
        "  - Offer an in-depth explanation of the biological mechanisms behind the condition.  "
            "- Include molecular, cellular, or systemic changes where applicable.  "

            "6. **Differential Diagnosis**:  "
        " - List conditions with similar presentations and how to distinguish them clinically.  "
        " - Mention specific tests or indicators useful in differentiation.  "

            "7. **Diagnostic Approach**:  "
            "- Provide a step-by-step guide for diagnosis, including:"
            " - Clinical examination findings.  "
                "- Laboratory tests (e.g., blood tests, biomarkers).  "
                "- Imaging techniques (e.g., X-ray, MRI, CT).  "
                "- Specialized diagnostic criteria or scoring systems.  "

            "8. **Treatment and Management**:  "
        "  - Summarize treatment protocols, including:"  
            " - First-line and second-line medications (names, dosages, and mechanisms). " 
            " - Non-pharmacological interventions (e.g., lifestyle changes, physiotherapy).  "
            " - Advanced options (e.g., surgeries, experimental therapies).  "
        " - Highlight evidence-based practices and recent clinical guidelines. " 

            "9. **Prognosis and Outcomes**:  "
        " - Describe expected outcomes with and without treatment. " 
        " - Discuss factors influencing prognosis, such as early diagnosis, treatment adherence, and comorbidities. " 

            "10. **Complications and Follow-Up**:  "
            " - Identify potential complications of the condition or treatment. " 
            " - Provide follow-up recommendations and monitoring protocols. " 

            "11. **Prevention and Public Health**:  "
            " - Outline preventive measures, including vaccinations, screenings, and public health strategies.  "
            "    - Highlight lifestyle modifications for at-risk populations.  "

            "12. **Research Insights and Advances**:  "
            " - Summarize recent studies, clinical trials, or technological advances from PubMed.  "
            " - Provide insights into emerging treatments, diagnostic tools, or evolving guidelines.  "

            "13. **Case Study Example (Optional)**:  "
            " - Include a brief, illustrative case study to demonstrate practical application for new doctors.  "

            "14. **Sources and References**:  "
            " - Cite all tools and references used (e.g., Wikipedia for general overviews, PubMed for research insights).  "

            "---"

            "**Additional Guidelines**:  "
            "- Use simple language to explain complex concepts for new doctors.  "
            "- Provide advanced insights, including the latest research, for experienced clinicians.  "
        " - Be precise, concise, and avoid unnecessary repetition.  "
        " - If information is limited, recommend further investigation or expert consultation."

            "User Query: {input}"
            "Agent scratchpad: {agent_scratchpad}"

            "Your response should balance practical clinical knowledge and scholarly depth, making it applicable in both academic and clinical settings."



        )
    )
    # prompt = hub.pull("hwchase17/openai-functions-agent")
    agent=create_tool_calling_agent(llm,tools=toolkit,prompt=custom_prompt)
    return AgentExecutor(agent=agent,tools=toolkit,verbose=True, return_intermediate_steps=True)

    


# Set up chains
def setup_rag_chain(vector_db):
    bookVSS_ret = vector_db.as_retriever()
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. You are trained on major books of pediatrics; therefore, when a question is posted to you, "
        "you should search the details about the problem from the dataset and provide the information in a concise, "
        "informative format. You have to use the tools and agents given to you in order to find out the best answer for a query."
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
    # llm = LLamaCpp(model_path="llama3-8b-8192", n_ctx=2048, verbose=True)
    qa_chain = RetrievalQA.from_chain_type(
        retriever=bookVSS_ret,
        llm=llm,
        return_source_documents=True
    )
    agent = load_agents(llm = llm)
    return qa_chain, agent

# Streamlit App
st.set_page_config(page_title="Pediatrics Q&A", layout="centered")
st.title("Pediatrics Chat Assistant")
st.write("Ask questions about pediatrics-related topics.")

# Load vector store and chains
vector_db = load_vector_db()
rag_chain, agent = setup_rag_chain(vector_db)

# Input from user
user_question = st.chat_input("Type your question here...")
if user_question:
    try:
        with st.spinner("Fetching response..."):
            st.markdown(f'**Question:** {user_question}')
           
            ## tool based response
            # agent_result = agent.run(user_question)
            # st.write(f"**Answer:**: {agent_result}")
            try:
                # result = agent.run(user_question)
                result = agent.invoke({'input': user_question, 'agent_scratchpad': ''}) 
                st.header("Tool based response")
                st.write(result.get('output'))
            except Exception as e:
                st.error(f"Parsing error: {e}")
                st.write("Raw LLM output:")
                st.write(result.get("output", "No output available"))
          
            ## vector db response
            response = rag_chain({"query": user_question})
            st.header("Vector based response")
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
    except Exception as e:
        st.error(f"An error occured: {e}")

