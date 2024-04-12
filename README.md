# GPT Analyst

This is a Python application that uses the power of large language models (LLMs) and vector databases to analyze and answer questions about a given document. In this case, the document is an annual report in PDF format.

## Prerequisites

- Python 3.8 or higher
- Hugging Face API token
- Streamlit

## Installation

1. Clone the repository:

```
git clone https://github.com/shon-Rocky/Annual-report-analyst-.git
```

2. Navigate to the project directory:

```
cd Annual-report-analyst-
```

3. Create a virtual environment and activate it:

```
python3 -m venv venv
source venv/bin/activate
```

4. Install the required packages:

```
pip install -r requirements.txt
```

5. Set your Hugging Face API token in dot env file:

```
 "HUGGINGFACEHUB_API_TOKEN=your_token_here" 
```

## Usage

1. Run the application:

```
streamlit run app.py
```

2. Input your prompt in the text box that appears. The application will then generate a response based on the annual report.

   ### example
   ![Uploading Screenshot (305).png…]()

3. For document similarity search, expand the 'Document Similarity Search' section. The application will display the most relevant pages from the annual report based on your prompt.

## Components

- `llm.py`: This module contains the `llm` function, which creates a HuggingFaceEndpoint instance for the  LLM.
- `embeddings.py`: This module contains the `embeddings` function, which creates an instance of HuggingFaceEmbeddings with the specified parameters.
- `app.py`: This is the main application file. It uses Streamlit to create a user interface, and it uses the `llm` and `embeddings` functions to generate responses and perform document similarity search.

## License

This project is licensed under the MIT License.
