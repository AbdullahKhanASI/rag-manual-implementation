# RAG and Agentic AI Course Notes

## Overview
This repository contains a comprehensive Udemy course on Retrieval-Augmented Generation (RAG) and Agentic AI. The course is structured into three main modules covering data ingestion, vector embeddings, and vector stores.

## Directory Structure

### 0-DataIngestParsing/
This folder contains notebooks focused on data ingestion and parsing various file formats for RAG systems.

#### 1-dataingestion.ipynb
**Purpose**: Introduction to document structure and basic data ingestion concepts

**Key Libraries:**
- `langchain_core.documents.Document` - Core document structure
- `langchain.text_splitter` (RecursiveCharacterTextSplitter, CharacterTextSplitter, TokenTextSplitter)
- `langchain_community.document_loaders.TextLoader`
- `langchain_community.document_loaders.DirectoryLoader`

**Main Functions/Methods:**
- `Document()` - Create document objects with page_content and metadata
- `TextLoader()` - Load single text files
- `DirectoryLoader()` - Batch load multiple files from directory
- `split_text()` - Split text into chunks
- `split_documents()` - Split document objects into chunks

**Key Concepts:**
- Document metadata for filtering and tracking
- Text splitting strategies (character vs recursive vs token-based)
- Chunk size and overlap parameters

#### 2-dataparsingpdf.ipynb
**Purpose**: PDF document processing and parsing

**Key Libraries:**
- `langchain_community.document_loaders.PyPDFLoader`
- `langchain_community.document_loaders.PyMuPDFLoader`
- `langchain_community.document_loaders.UnstructuredPDFLoader`

**Main Functions/Methods:**
- `PyPDFLoader()` - Basic PDF loading
- `PyMuPDFLoader()` - Fast PDF processing with better text extraction
- `load()` - Load PDF pages as documents
- Custom `SmartPDFProcessor` class for advanced PDF handling

**Key Concepts:**
- PDF extraction challenges (ligatures, formatting issues)
- Text cleaning and preprocessing
- Page-wise document processing
- Enhanced metadata for PDF chunks

#### 3-dataparsingdoc.ipynb
**Purpose**: Microsoft Word document processing

**Key Libraries:**
- `langchain_community.document_loaders.Docx2txtLoader`
- `langchain_community.document_loaders.UnstructuredWordDocumentLoader`

**Main Functions/Methods:**
- `Docx2txtLoader()` - Basic Word document loading
- `UnstructuredWordDocumentLoader()` - Structured element extraction
- Element-based parsing with categories (Title, NarrativeText, ListItem, Table)

**Key Concepts:**
- Document structure preservation
- Element categorization
- Metadata enrichment with element types

#### 4-dataparsingcsvexcel.ipynb
**Purpose**: Structured data processing (CSV and Excel files)

**Key Libraries:**
- `langchain_community.document_loaders.CSVLoader`
- `langchain_community.document_loaders.UnstructuredExcelLoader`
- `pandas` for data manipulation

**Main Functions/Methods:**
- `CSVLoader()` - Row-based document creation
- `process_csv_intelligently()` - Custom CSV processing
- `process_excel_with_pandas()` - Excel sheet processing
- `pd.read_csv()`, `pd.read_excel()` - Data loading

**Key Concepts:**
- Row-to-document mapping
- Multi-sheet Excel processing
- Metadata preservation for structured data
- Custom processing for better context

#### 5-dataparsingjson.ipynb
**Purpose**: JSON data processing and parsing

**Key Libraries:**
- `langchain_community.document_loaders.JSONLoader`
- `json` for data manipulation

**Main Functions/Methods:**
- `JSONLoader()` - JSON document loading with jq schemas
- `process_json_intelligently()` - Custom JSON processing
- jq queries for data extraction

**Key Concepts:**
- jq schema patterns for data extraction
- Nested JSON structure handling
- Context preservation in document creation

#### 6-dataparsingdatabase.ipynb
**Purpose**: SQL database integration and processing

**Key Libraries:**
- `langchain_community.utilities.SQLDatabase`
- `langchain_community.document_loaders.SQLDatabaseLoader`
- `sqlite3` for database operations

**Main Functions/Methods:**
- `SQLDatabase.from_uri()` - Database connection
- `sql_to_documents()` - Custom SQL to document conversion
- `get_table_info()` - Schema information
- Relationship queries with JOINs

**Key Concepts:**
- Database schema preservation
- Table-to-document mapping
- Relationship document creation
- SQL query result processing

### 1-VectorEmbeddingsAndDatabases/
This folder focuses on vector embeddings and their applications in RAG systems.

#### 1-embeddings.ipynb
**Purpose**: Introduction to embeddings and vector representations

**Key Libraries:**
- `langchain_huggingface.HuggingFaceEmbeddings`
- `numpy` for mathematical operations
- `matplotlib.pyplot` for visualization

**Main Functions/Methods:**
- `HuggingFaceEmbeddings()` - Initialize embedding models
- `embed_query()` - Create embeddings for single text
- `embed_documents()` - Batch embedding creation
- `cosine_similarity()` - Calculate similarity between vectors

**Key Concepts:**
- Vector representation of text
- Cosine similarity calculations
- Popular open-source embedding models (MiniLM, MPNet)
- Model comparison (speed vs quality)

#### 2-openaiembeddings.ipynb
**Purpose**: OpenAI embeddings implementation and semantic search

**Key Libraries:**
- `langchain_openai.OpenAIEmbeddings`
- `numpy` for vector operations

**Main Functions/Methods:**
- `OpenAIEmbeddings()` - Initialize OpenAI embedding models
- `embed_query()` - Single text embedding
- `embed_documents()` - Batch processing
- `semantic_search()` - Custom semantic search implementation

**Key Concepts:**
- OpenAI embedding models comparison (text-embedding-3-small vs large)
- Semantic similarity search
- Vector similarity rankings
- Cost considerations for different models

### 2-VectorStores/
This folder covers various vector database implementations for storing and retrieving embeddings.

#### 1-chromadb.ipynb
**Purpose**: Complete RAG system implementation using ChromaDB

**Key Libraries:**
- `langchain_community.vectorstores.Chroma`
- `langchain_openai.OpenAIEmbeddings`
- `langchain_openai.ChatOpenAI`
- `langchain.chains.create_retrieval_chain`
- `langchain.chains.combine_documents.create_stuff_documents_chain`

**Main Functions/Methods:**
- `Chroma.from_documents()` - Create vector store from documents
- `similarity_search()` - Find similar documents
- `similarity_search_with_score()` - Get similarity scores
- `as_retriever()` - Convert to retriever interface
- `create_retrieval_chain()` - Build RAG pipeline
- `create_stuff_documents_chain()` - Document processing chain

**Key Concepts:**
- End-to-end RAG implementation
- Vector store persistence
- Similarity search with metadata
- Conversational RAG with memory
- LCEL (LangChain Expression Language)
- Prompt templates and chain composition

#### 2-faiss.ipynb
**Purpose**: High-performance vector search using FAISS

**Key Libraries:**
- `langchain_community.vectorstores.FAISS`
- `langchain_openai.OpenAIEmbeddings`
- `langchain_core.runnables` (RunnablePassthrough, RunnableParallel)

**Main Functions/Methods:**
- `FAISS.from_documents()` - Create FAISS index
- `save_local()` - Persist vector store
- `load_local()` - Load saved vector store
- `similarity_search_with_score()` - Search with distance scores
- Metadata filtering capabilities

**Key Concepts:**
- High-performance vector indexing
- Local persistence and loading
- Streaming responses
- Conversational RAG implementation
- Multiple chain variants (simple, streaming, conversational)

#### 3-othervectorstores.ipynb
**Purpose**: InMemoryVectorStore implementation

**Key Libraries:**
- `langchain_core.vectorstores.InMemoryVectorStore`
- `langchain_openai.OpenAIEmbeddings`

**Main Functions/Methods:**
- `InMemoryVectorStore()` - Initialize in-memory store
- `add_documents()` - Add documents to store
- `similarity_search()` - Search functionality
- `as_retriever()` - Retriever conversion

**Key Concepts:**
- In-memory vector storage
- Simple vector store implementation
- Cosine similarity computation
- Retriever interface usage

#### 4-pineconevectordb.ipynb
**Purpose**: Cloud-based vector storage with Pinecone

**Key Libraries:**
- `pinecone` - Pinecone client
- `langchain_pinecone.PineconeVectorStore`
- `pinecone.ServerlessSpec`

**Main Functions/Methods:**
- `Pinecone()` - Initialize Pinecone client
- `create_index()` - Create vector index
- `PineconeVectorStore()` - LangChain Pinecone integration
- `similarity_search_with_score()` - Scored search
- `similarity_score_threshold` - Threshold-based retrieval

**Key Concepts:**
- Cloud vector database
- Serverless vector indexing
- Metadata filtering
- Similarity threshold retrieval
- Production-scale vector storage

## Key Technologies and Libraries Used

### Core LangChain Components
- **Document**: Base document structure with content and metadata
- **Text Splitters**: Various chunking strategies for optimal retrieval
- **Document Loaders**: File format-specific loading utilities
- **Vector Stores**: Storage and retrieval of vector embeddings
- **Retrievers**: Search interfaces for vector stores
- **Chains**: Composable processing pipelines

### Embedding Models
- **HuggingFace Models**: Open-source embedding models (MiniLM, MPNet)
- **OpenAI Embeddings**: Commercial embedding APIs (text-embedding-3-small/large)

### Vector Databases
- **ChromaDB**: Open-source vector database with persistence
- **FAISS**: Facebook's fast similarity search library
- **Pinecone**: Cloud-native vector database
- **InMemoryVectorStore**: Simple in-memory vector storage

### Language Models
- **OpenAI GPT**: GPT-3.5-turbo, GPT-4 for text generation
- **Chat Models**: Conversational AI interfaces

### Data Processing
- **Pandas**: Structured data manipulation
- **NumPy**: Numerical computations and vector operations
- **SQLite**: Database operations and SQL queries

## Common Patterns and Best Practices

### Document Processing
1. **Chunking Strategy**: Use RecursiveCharacterTextSplitter for most texts
2. **Metadata Preservation**: Always maintain source and context information
3. **Preprocessing**: Clean text (remove ligatures, fix formatting)
4. **Overlap**: Use 10-20% overlap between chunks for context preservation

### Vector Store Selection
- **ChromaDB**: Best for development and small to medium datasets
- **FAISS**: Optimal for high-performance local deployments
- **Pinecone**: Ideal for production cloud deployments
- **InMemory**: Suitable for testing and small datasets

### RAG Implementation
1. **Retrieval**: Use k=3-5 documents for context
2. **Scoring**: Consider similarity thresholds for quality filtering
3. **Prompting**: Clear instructions for context usage
4. **Memory**: Implement conversational history for follow-up questions

### Performance Optimization
- **Embedding Models**: Balance between speed (MiniLM) and quality (MPNet)
- **Chunk Size**: 200-500 tokens optimal for most use cases
- **Indexing**: Persist vector stores for faster startup
- **Caching**: Reuse embeddings when possible

This comprehensive course provides a solid foundation for building production-ready RAG systems with various data sources and vector storage solutions.