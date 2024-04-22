# GPT Analyst

This is a Python application that uses the power of large language models (LLMs) and vector databases to analyze and answer questions about a given document. In this case, the document is an annual report in PDF format.

## Prerequisites

- Python 3.8 or higher
- Hugging Face API token and groq api token
- Streamlit

## Installation

1. Clone the repository:

```
git clone https://github.com/shon-Rocky/GPT-Analyst.git
```

2. Navigate to the project directory:

```
cd GPT-Analyst
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

5. Set your API token in dot env file:

Either you can use huggingface model or groq model
```
 "HUGGINGFACEHUB_API_TOKEN=your_token_here" 
 "GROQ_API_KEY=your_token_here"

```

## Usage

1. Run the application:

```
streamlit run app.py
```

2. Input your prompt in the text box that appears. The application will then generate a response based on the annual report.

   ### example
   ![Screenshot (305)](https://github.com/shon-Rocky/Annual-report-analyst-/assets/140310009/50b935f9-812b-4df3-87c6-ea9c9d889f31)


3. For document similarity search, expand the 'Document Similarity Search' section. The application will display the most relevant pages from the annual report based on your prompt.

## Components

- `llm.py`: This module contains the `llm` function, which creates a HuggingFaceEndpoint instance for the  LLM .
- `embeddings.py`: This module contains the `embeddings` function, which creates an instance of HuggingFaceEmbeddings with the specified parameters.
- `app.py`: This is the main application file. It uses Streamlit to create a user interface, and it uses the `llm` and `embeddings` functions to generate responses and perform document similarity search.

## License

This project is licensed under the MIT License.
