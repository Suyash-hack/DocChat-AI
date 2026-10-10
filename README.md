
# DocChat AI

An AI-powered document question-answering application that allows users to upload PDF documents and ask questions in natural language.

## Features

- Multiple PDF uploads
- PDF text extraction and chunking
- Semantic search using Hugging Face embeddings
- FAISS vector database for retrieval
- Gemini-powered question answering
- Retrieved source excerpts with page references
- Downloadable answers and conversation history
- Modern Streamlit interface

## Tech Stack

- Python
- Streamlit
- LangChain
- Google Gemini API
- Hugging Face Sentence Transformers
- FAISS
- PyPDF

## Run Locally

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install dependencies:

   `pip install -r requirements.txt`

4. Create a `.env` file and add your Gemini API key:

   `GEMINI_API_KEY=your_api_key_here`

5. Start the application:

   `streamlit run ui.py`
