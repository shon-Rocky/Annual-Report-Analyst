from llm import llm  # Import the LLM module
from embeddings import embeddings  # Import the embeddings module
import streamlit as st  # Import the Streamlit library
from langchain_community.document_loaders import PyPDFLoader  # Import the PyPDFLoader module
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Import the RecursiveCharacterTextSplitter module
import os  # Import the os module
from dotenv import load_dotenv  # Import the load_dotenv function from the dotenv module
from langchain_community.vectorstores import FAISS  # Import the FAISS module
from langchain.embeddings import CacheBackedEmbeddings  # Import the CacheBackedEmbeddings module
from langchain.storage import LocalFileStore  # Import the LocalFileStore module

load_dotenv()  # Load environment variables from the .env file

os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")  # Set the Hugging Face API token

# Create instance of LLM
llm_instance = llm()
# Create instance of embeddings
embeddings_instance = embeddings()

# Import vector store stuff
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)

# Initialize LocalFileStore
store = LocalFileStore("./cache/")

# Create and load PDF Loader
loader = PyPDFLoader('annualreport.pdf')
# Split pages from pdf 
documents = loader.load_and_split()

# Create FAISS index from documents
store = FAISS.from_documents(documents, CacheBackedEmbeddings.from_bytes_store(embeddings_instance, store, namespace="all-MiniLM-l6-v2"))

# Define VectorStoreInfo
vectorstore_info = VectorStoreInfo(
   name="annual_report",
   description="Banking annual report (Faiss)",
   vectorstore=store
)

# Convert the document store into a langchain toolkit
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info, llm=llm_instance)

# Add the toolkit to an end-to-end LC
agent_executor = create_vectorstore_agent(
    llm=llm_instance,
    toolkit=toolkit,
    verbose=True
)
st.title('🦜🔗 GPT Analyst')
# Create a text input box for the user
prompt = st.text_input('Input your prompt here')

# If the user hits enter
if prompt:
    # Then pass the prompt to the LLM
    response = agent_executor.run(prompt)
    # Write the response to the screen
    st.write(response)

    # Use a Streamlit expander for document similarity search
    with st.expander('Document Similarity Search'):
        # Find the relevant pages
        search = store.similarity_search_with_score(prompt) 
        # Write out the first 
        st.write(search[0][0].page_content) 
