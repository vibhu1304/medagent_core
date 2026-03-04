🚀 DocuMind: Multi-Domain Document Intelligence System
DocuMind is an advanced Retrieval-Augmented Generation (RAG) platform that transforms static PDF documents into actionable intelligence. Unlike standard chatbots, DocuMind uses Intent-Based Routing to switch between strict factual extraction and expert-level analytical reasoning, making it applicable to Legal, Medical, and Professional (Resume) domains.

🧠 Key Architectural Features
Intent-Based Routing: Dynamically classifies user queries to select the optimal generation path: Strict Extraction for facts or Hybrid Analysis for evaluations.

Semantic-Keyword Hybrid Retrieval: Combines FAISS-based vector search with keyword boosting to ensure high precision in document lookups.

Conditional Query Expansion: Utilizes a FLAN-T5 rewriter to transform vague user prompts into search-optimized queries, improving retrieval recall.

Heuristic Content Filtering: Implements a "header-scrubber" logic to bypass contact information and metadata, focusing the LLM on core document body content.

Production-Ready Memory Management: Uses singleton patterns and Streamlit resource caching to manage large-scale transformer models (FLAN-T5-Large) efficiently on consumer hardware.

Technical Stack
1)Large Language Model (LLM): google/flan-t5-large is used as the primary engine for document analysis, reasoning, and final answer generation.

2)Query Transformation Model: google/flan-t5-base is utilized specifically for rewriting and expanding user queries to improve search accuracy.    

3)Vector Database: FAISS (Facebook AI Similarity Search) manages the semantic indexing and high-speed retrieval of document chunks using Inner Product (IP) similarity.

4)Embedding Model: all-MiniLM-L6-v2 from the sentence-transformers library is used to convert text chunks and user queries into dense mathematical vectors.

5)Backend Framework: Streamlit provides the reactive web interface for document uploading and real-time user interaction.

6)Deep Learning Library: PyTorch (torch) serves as the underlying framework for running the transformer models and managing hardware acceleration (CUDA/MPS).

7)NLP Processing: NLTK is used for sentence-level tokenization to ensure clean document chunking.

8)PDF Extraction: pypdf (PdfReader) is used to parse and extract raw text from uploaded PDF files.

8)Numerical Computation: NumPy is used for handling vector arrays and mathematical operations within the retrieval system.

System WorkflowIngestion & Chunking: Documents are parsed and split using a sliding-window strategy (12-sentence chunks with 3-sentence overlap) to preserve semantic context across boundaries.
Vector Indexing: Chunks are embedded and stored in an Inner Product (IP) FAISS index for efficient cosine similarity matching.
Query Transformation: Raw user queries are expanded to improve the mathematical overlap with the document's vector space.
Content Filtering: The system retrieves the top $k$ chunks and filters out non-professional metadata (headers, contact info) to ensure clean context.
Routed Generation: The FLAN-T5 model generates a response based on the detected intent—ensuring the AI "thinks" as a Subject Matter Expert.

How to Run
Clone the Repository:
Bash
git clone https://github.com/vibhu1304/DocuMind-AI.git
cd DocuMind-AI


Install Dependencies:

Bash
pip install -r requirements.txt


Launch the Application:

Bash
streamlit run app.py
