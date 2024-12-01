from dotenv import load_dotenv
import streamlit as st
import zipfile
from langchain_community.vectorstores import Chroma
from huggingface_hub import hf_hub_download
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_community.tools import  WikipediaQueryRun, YouTubeSearchTool, DuckDuckGoSearchRun, PubmedQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, DuckDuckGoSearchAPIWrapper
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools.retriever import create_retriever_tool
import concurrent.futures
import os
import chromadb
import requests
from typing import List
from langchain_core.pydantic_v1 import BaseModel, Field
import json
from langchain_core.agents import AgentActionMessageLog, AgentFinish
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.tools.base import BaseTool
# from wikipedia_tool import WikipediaQueryRun  # Custom import
# from pubmed_tool import PubmedQueryRun        # Custom import
# from duckduckgo_tool import DuckDuckGoSearchRun  # Custom import
# from youtube_tool import YouTubeSearchTool 
from langchain.tools import Tool   
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv('groq_api_key')


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
    print(f'Extract_path: {extract_path}')
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
    vector_db = Chroma(persist_directory=extract_path, collection_name="bookVectors",embedding_function=HuggingFaceEmbeddings())
    return vector_db

## Define Response Schema
class Response(BaseModel):
    """Final Answer corresponding to the question asked by the user """
    answer: str = Field(description="The final response for the user")
    source: List[str] = Field(description='Source of the response')

    # def serialize(self):
    #  return self.dict()

def parseOutput(output):
    if 'function_call' not in output.additional_kwargs:
        return AgentFinish(return_values={"content": output.content}, log=output.content)

    function_call = output.additional_kwargs['function_call']
    name = function_call.name
    inputs=json.loads(function_call.arguments)

    if name == 'Response':
        # Serialize the Response object
        # response = Response(**inputs).serialize()
        return AgentFinish(return_values=inputs, log=str(function_call))
    else:
        # Collect results from all tools and combine them
        tool_results = {}
        for tool in output.additional_kwargs['tool_calls']:
            tool_name = tool['name']
            tool_results[tool_name] = tool.get('result', 'No result')
        
        combined_response = {
            "aggregated_tool_results": tool_results,
            "final_response": output.content
        }
        return AgentFinish(return_values=combined_response, log=str(function_call))




# Set up chains
def setup_rag_chain(vector_db):
    bookVSS_ret = vector_db.as_retriever()
    # system_prompt = (
    #     "You are an assistant for question-answering tasks. "
    #     "Use the following pieces of retrieved context to answer "
    #     "the question. You are trained on major books of pediatrics; therefore, when a question is posted to you, "
    #     "you should search the details about the problem from the dataset and provide the information in a concise, "
    #     "informative format. "
    #     "You should also use all the tools and agents available to you in order to get the best output for the user input. If you find any video link to an answer show that video link as well. "
    #     "If you don't know the answer, say that you don't know. "
    #     "Use points to structure the content and keep the answer concise."
    #     "\n\n"
    #     "{{context}}"
    # )
    # system_prompt = (
    #     "You are an assistant for question-answering tasks. "
    #     "Use the following pieces of retrieved context to answer the question. "
    #     "Gather information from all available tools and provide a combined, concise response."
    #     "\n\n"
    #     "{{context}}"
    # )
    # prompt = ChatPromptTemplate.from_messages([
    #     ('system', system_prompt),
    #     ('human', "{input}"),
    #     MessagesPlaceholder(variable_name='agent_scratchpad')
    # ])
    prompt_new = ChatPromptTemplate.from_messages([
        ("system", "You are an intelligent assistant. Use the tools provided to answer the user's questions."),
        ("human", "{input}"),
        # MessagesPlaceholder(variable_name="intermediate_steps"),
        MessagesPlaceholder(variable_name='agent_scratchpad')
    ])
    llm = ChatGroq(model='llama3-8b-8192')
    # qa_chain = RetrievalQA.from_chain_type(
    #     retriever=bookVSS_ret,
    #     llm=llm,
    #     return_source_documents=True
    # )
    toolkit = create_tools_for_search(bookVSS_ret)
    agent_chain = call_toolkit(llm=llm, toolkit=toolkit, prompt_new=prompt_new)
    return agent_chain

def create_tools_for_search(retriever): 
    api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=5, doc_content_chars_max=800)
    wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

    retriever_tool = create_retriever_tool(retriever=retriever, name='Existing dataset search', description='Search data from the existing pdf dataset.')

    # search=DuckDuckGoSearchRun()
    # search = DuckDuckGoSearchRun()
    
    # yt_tool = YouTubeSearchTool()

    pubmed_tool = PubmedQueryRun()

    return [wiki_tool, retriever_tool,   pubmed_tool]

class WikipediaSearchArgs(BaseModel):
    query: str


# Correct the `create_tools_for_search` function
def create_tools_for_search_new(retriever):
    # Create tools with proper configurations
    api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=5, doc_content_chars_max=800)
    # wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)
    wiki_tool = Tool(
        name="Wikipedia Search",
        func=WikipediaQueryRun(api_wrapper=api_wrapper_wiki).run,
        description="Search general knowledge and definitions on Wikipedia.",
        args_schema=WikipediaSearchArgs,
    )
    
    retriever_tool = create_retriever_tool(retriever=retriever, name='Existing dataset search', description='Search data from the existing pdf dataset.')


    search_tool = Tool(
        name="Web Search",
        func=DuckDuckGoSearchRun().run,
        description="Search the web for the latest information."
    )
    yt_tool = Tool(
        name="YouTube Search",
        func=YouTubeSearchTool().run,
        description="Search videos on YouTube for relevant content."
    )
    pubmed_tool = Tool(
        name="PubMed Search",
        func=PubmedQueryRun().run,
        description="Search academic articles and research papers on PubMed."
    )

    return [wiki_tool, retriever_tool, search_tool, yt_tool, pubmed_tool]


def call_toolkit(llm, toolkit, prompt_new): 
   
    agent = create_tool_calling_agent(llm, tools=toolkit, prompt=prompt_new)
    agent = agent | StrOutputParser()
    return AgentExecutor(agent=agent, tools=toolkit, verbose=True)
    # llm_with_tools = llm.bind_tools(toolkit)
    # agent = (
    #     {
    #         "input": lambda x: x["input"],
    #         "agent_scratchpad": lambda x: format_to_openai_function_messages(x['intermediate_steps'])
    #     }
    #     | prompt
    #     | llm_with_tools
    #     | parseOutput
    # )
    # return AgentExecutor(tools=toolkit, agent=agent, verbose=True)


def sanitize_input(user_input):
    # Remove non-numeric conversions or validate as needed
    if isinstance(user_input, str) and not user_input.isdigit():
        return user_input.strip()
    return user_input

def process_tool_output(tool_name, tool_response):
    """
    Standardizes and structures the output from different tools for consistent aggregation.
    """
    # If the response is a dictionary and contains a 'result' key, return its value
    if isinstance(tool_response, dict):
        if 'result' in tool_response:
            return tool_response['result']
        elif 'content' in tool_response:  # Handle tools with a 'content' key
            return tool_response['content']
        else:
            return f"{tool_name} returned an unstructured response: {tool_response}"

    # If the response is a string, clean it and return
    elif isinstance(tool_response, str):
        return tool_response.strip()

    # If the response is a list, join elements into a readable string
    elif isinstance(tool_response, list):
        return '\n'.join([str(item) for item in tool_response])

    # Catch-all for unexpected formats
    return f"{tool_name} did not return a valid response or output is unsupported: {tool_response}"


# Streamlit App
st.set_page_config(page_title="Pediatrics Q&A", layout="centered")
st.title("Pediatrics Chat Assistant")
st.write("Ask questions about pediatrics-related topics.")

# Load vector store and chains
vector_db = load_vector_db()
agent_executor = setup_rag_chain(vector_db)

# Input from user
user_question = st.chat_input("Type your question here...")
if user_question:
    st.markdown(f'**Question:** {user_question}')
    response = agent_executor.invoke({"input": user_question}, return_only_outputs=True)
    # Handle the response if it's a complex object
    if isinstance(response, list): 
        # Extract string representation if the response is a list of tool actions or other non-string types
        parsed_response = str(response)
    else:
        parsed_response = response  # If it's already a string, use it directly
    
    # Display the response
    st.write(f"**Answer:** {parsed_response}")
    print(f" ---- response----")
    print(parsed_response)  
# if user_question:
#     with st.spinner("Fetching response..."):
#         response = rag_chain({"input": sanitize_input(user_question)})
#         print(response)
        # st.markdown(f'**Question:** {user_question}')
        # st.markdown(f"**Answer:** {response['result']}")

        # # Display sources
        # st.markdown("### Sources")
        # if response["source_documents"]:
        #     for doc in response["source_documents"]:
        #         source = doc.metadata.get('source', 'Unknown Source')
        #         page = doc.metadata.get('page', 'N/A')
        #         st.markdown(f"- **Source:** {source}, **Page:** {page}")
        # else:
        #     st.markdown("No source documents were found.")
