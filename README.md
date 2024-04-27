**Annual Report Analyst**
==========================

This is a LangChain-based application that analyzes annual reports and provides insights based on user input. The application uses a combination of natural language processing (NLP) and information retrieval techniques to extract relevant information from the annual report and provide a response to the user's query.

**Features**
------------

* Analyze annual reports and extract relevant information based on user input
* Use LangChain's vector store and embeddings to enable semantic search and similarity search
* Provide a user-friendly interface for users to input their queries and receive responses

**Technical Details**
--------------------

* The application uses LangChain's vector store and embeddings to enable semantic search and similarity search.
* The annual report is processed using PyPDFLoader and split into pages using RecursiveCharacterTextSplitter.
* The application uses FAISS to create an index of the annual report pages and enable fast similarity search.
* The user interface is built using Streamlit.

**Getting Started**
---------------

To run the application, follow these steps:

1. Install the required dependencies.
```
   pip install -r requirements.txt
```
3. Run the application.
```
   streamlit run app.py
```
5. Open a web browser and navigate to
```
   http://localhost:8501
```
7. Input your query in the text box and press enter to receive a response.

**Configuration**
---------------

The application uses environment variables to configure the  API key's and other settings. You can set these variables using a `.env` file or by setting environment variables in your operating system.

**Example**
---------------

   ![image](https://github.com/shon-Rocky/GPT-Analyst/assets/140310009/052d8e6d-2fa4-421e-b010-50966ea5cdf7)
   

**License**
-------

This application is licensed under the MIT License. See the `LICENSE` file for details.

**Acknowledgments**
---------------

This application uses LangChain, a open-source framework for building AI applications. We acknowledge the contributions of the LangChain community to the development of this application.
