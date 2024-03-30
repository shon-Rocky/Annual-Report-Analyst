from llm import llm
from embeddings import embeddings
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

load_dotenv()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")



# Import vector store stuff
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)


# Create instance of  LLM
llm = llm()
embeddings = embeddings()

# Create and load PDF Loader
loader = PyPDFLoader('annualreport.pdf')
# Split pages from pdf 
documents = loader.load_and_split()
  
store = FAISS.from_documents(documents, embeddings)
vectorstore_info = VectorStoreInfo(
   name="annual_report",
   description="Banking annual report (Faiss)",
   vectorstore=store
    )
# Convert the document store into a langchain toolkit
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info, llm=llm)

# Add the toolkit to an end-to-end LC
agent_executor = create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)
st.title('🦜🔗 GPT Investment Banker')
# Create a text input box for the user
prompt = st.text_input('Input your prompt here')

# If the user hits enter
if prompt:
    # Then pass the prompt to the LLM
    response = agent_executor.run(prompt)
    # ...and write it out to the screen
    st.write(response)

    # With a streamlit expander  
    with st.expander('Document Similarity Search'):
        # Find the relevant pages
        search = store.similarity_search_with_score(prompt) 
        # Write out the first 
        st.write(search[0][0].page_content) 
