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
from langchain_groq import ChatGroq
import time

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"]="pt-giving-colon-23"

# # Record the start time
# start_time = time.time()

load_dotenv()  # Load environment variables from the .env file

api_key = os.environ['GROQ_API_KEY']
# Create instance of LLM
# llm_instance = llm() You can change it this hugging face llm
llm_instance = ChatGroq(temperature=0, groq_api_key=api_key, model_name="mixtral-8x7b-32768")
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

cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    embeddings_instance, store, namespace="all-MiniLM-l6-v2"
)

# Create and load PDF Loader
raw_documents = PyPDFLoader('Tata-motors-annual-report-2022-23.pdf').load()
# Split pages from pdf 
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(raw_documents)

# Create FAISS index from documents
store = FAISS.from_documents(documents, cached_embedder)

# Define VectorStoreInfo
vectorstore_info = VectorStoreInfo(
   name="annual_report",
   description="Compnay annual report (Faiss)",
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
st.title('🦜🔗 Annual Report Analyst')
# Create a text input box for the user
prompt = st.text_input('Input your prompt here')

# If the user hits enter
if prompt:
    # Then pass the prompt to the LLM
    response = agent_executor.run(prompt)
    # Write the response to the screen
    st.write(response)

#     # Use a Streamlit expander for document similarity search
#     with st.expander('Document Similarity Search'):
#         # Find the relevant pages
#         search = store.similarity_search_with_score(prompt) 
#         # Write out the first 
#         st.write(search[0][0].page_content) 




# prompt = 'Provide a summary on the annual report?'
# response = agent_executor.run(prompt)
# print(response)

# end_time = time.time()

# # Calculate the elapsed time
# elapsed_time = end_time - start_time

# # Print the elapsed time
# print(f"Program completed in {elapsed_time:.2f} seconds")

