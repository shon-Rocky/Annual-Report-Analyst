# GPT Analyst

This is a Python application that uses the power of large language models (LLMs) and vector databases to analyze and answer questions about a given document. 

## Prerequisites

- Python 3.8 or higher
- Hugging Face API token and groq api token
- Streamlit

## Installation

1. Clone the repository:

```
https://github.com/shon-Rocky/GPT-Analyst.git
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


3. For document similarity search, expand the 'Document Similarity Search' section. The application will display the most relevant pages from the annual report based on your prompt.


### example
   ![image](https://github.com/shon-Rocky/GPT-Analyst/assets/140310009/052d8e6d-2fa4-421e-b010-50966ea5cdf7)
   

## Components

- `llm.py`: This module contains the `llm` function, which creates a HuggingFaceEndpoint instance for the  LLM .
- `embeddings.py`: This module contains the `embeddings` function, which creates an instance of HuggingFaceEmbeddings with the specified parameters.
- `app.py`: This is the main application file. It uses Streamlit to create a user interface, and it uses the `llm` and `embeddings` functions to generate responses and perform document similarity search.

## License

This project is licensed under the MIT License.
