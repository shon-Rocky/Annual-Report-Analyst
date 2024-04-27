from src.llm import llm
from src.embeddings import embeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
# from langchain_groq import ChatGroq

from joblib import dump, load
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)


load_dotenv()

# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
# os.environ["LANGCHAIN_PROJECT"]="pt-giving-colon-23"
# api_key = os.environ['GROQ_API_KEY']

# Create instance of LLM
llm_instance = llm() #You can change it this hugging face llm
# llm_instance = ChatGroq(temperature=0, groq_api_key=api_key, model_name="mixtral-8x7b-32768")
# Create instance of embeddings
embeddings_instance = embeddings()

# Import vector store stuff


# Initialize LocalFileStore
store = LocalFileStore("./cache/")

cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    embeddings_instance, store, namespace="all-MiniLM-l6-v2"
)


def rag():
    pdf_filename = 'pdf_file/Tata-motors-annual-report-2022-23.pdf'
    cache_suffix = 'cached.joblib'
    cache_filename = pdf_filename + cache_suffix

    # Check if cache exists
    if os.path.exists(cache_filename):
        # Load data from cache
        print("Loading data from cache...")
        documents = load(cache_filename)
    else:
        print("Processing PDF and creating cache...")
        raw_documents = PyPDFLoader(pdf_filename).load()
        # Split pages from pdf 
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        documents = text_splitter.split_documents(raw_documents)
        # Save data to cache
        dump(documents, cache_filename)

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
    
    return agent_executor, store