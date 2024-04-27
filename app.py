import streamlit as st
import time 

@st.cache_data
def load_rag():
    from src.rag_implimentation import rag
    return rag()

start_time = time.time() 
agent_executor, store = load_rag()
end_time = time.time()   
elapsed_time = end_time - start_time
print(f"Time taken to load RAG: {elapsed_time:.2f} seconds")

st.title('🦜🔗 Annual Report Analyst')

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
