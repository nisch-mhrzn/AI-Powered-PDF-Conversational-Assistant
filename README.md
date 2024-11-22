# Project Title  
**AI-Powered PDF Conversational Assistant**  

---

## Project Description  

The **AI-Powered PDF Conversational Assistant** is a Streamlit-based application that allows users to upload and interact with multiple PDF documents using natural language queries. By leveraging state-of-the-art NLP models, vector-based search, and conversational AI, this application transforms static PDFs into interactive, context-aware chat interfaces. Users can ask questions related to the uploaded documents and receive insightful responses based on the content.  

This tool is particularly useful for researchers, professionals, and students looking to extract relevant information quickly from lengthy or multiple PDF documents.  

---

## Features  

- **Multiple PDF Upload**: Supports uploading multiple PDF documents simultaneously.  
- **Text Extraction**: Extracts textual content from all pages of the uploaded PDFs.  
- **Text Chunking**: Splits extracted text into manageable chunks to enable efficient embedding and querying.  
- **Semantic Search**: Utilizes SentenceTransformer embeddings for accurate semantic representation and retrieval.  
- **Conversational Interface**: Facilitates a conversational interaction with documents using a memory-enabled AI agent.  
- **Interactive UI**: A user-friendly interface powered by Streamlit for seamless user interactions.  
- **Custom LLM Integration**: Employs cutting-edge models such as Google FLAN-T5-XXL for natural language understanding and generation.  

---

## How It Works  

1. **Document Upload**: Users upload one or more PDF files through the application.  
2. **Text Extraction**: The application extracts text from the PDFs and processes it into structured chunks with some overlap for context.  
3. **Vector Store Creation**: Creates a FAISS vector store for storing the document embeddings using SentenceTransformer models.  
4. **Conversational Chain**: Initializes a conversational retrieval chain using a memory-enabled language model to maintain context across user interactions.  
5. **Query Handling**: Users input natural language questions, and the system retrieves the most relevant information from the documents and responds conversationally.  

---

## Technologies Used  

- **Python**: Core programming language for development.  
- **Streamlit**: Framework for building the user interface.  
- **PyPDF2**: PDF text extraction.  
- **LangChain**: Framework for integrating NLP models and vector-based retrieval.  
- **FAISS**: For efficient similarity search and clustering of text embeddings.  
- **HuggingFace Transformers**: Pretrained models for text embeddings and conversational AI.  
- **SentenceTransformers**: Embedding generation using the HKUNLP Instructor-XL model.  

---

## Requirements  

- Python 3.8+  
- Streamlit  
- PyPDF2  
- LangChain  
- HuggingFace Transformers  
- SentenceTransformers  
- FAISS  
- dotenv  

Install dependencies using:  
```bash
pip install -r requirements.txt
```  

---

## Usage  

1. Clone this repository:  
   ```bash
   git clone <https://github.com/nisch-mhrzn/AI-Powered-PDF-Conversational-Assistant>
   cd <AI-Powered-PDF-Conversational-Assistant>
   ```  
2. Install the required Python packages.  
3. Run the application:  
   ```bash
   streamlit run app.py
   ```  
4. Upload PDFs, ask questions, and explore your documents conversationally.  

---

## Folder Structure  

```
project/
├── app.py                 # Main application script
├── requirements.txt       # Python dependencies
├── htmlTemplates.py       # HTML templates for chatbot UI
├── .env                   # Environment variables
├── README.md              # Project documentation (this file)
└── ... (additional utility scripts and files)
```  

---

## Future Enhancements  

- **Support for Other File Types**: Extend support to Word documents, Excel sheets, and more.  
- **Advanced Summarization**: Integrate summarization capabilities for lengthy documents.  
- **Multi-Model Support**: Add support for OpenAI GPT or other advanced language models.  
- **Cloud Integration**: Enable secure cloud storage and access for documents.  

---

