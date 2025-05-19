# RAG (Retrieval Augmented Generation)

## Overview
Retrieval Augmented Generation (RAG) is a technique that enhances the capabilities of language models by combining them with a retrieval system. This project demonstrates how to build RAG-based chatbots and information retrieval systems using Python, LangChain, and Ollama, with a focus on real-world policy document use cases.

## Features
- **RAG Chatbot**: Ask questions and get answers grounded in your own document collection.
- **Policy Document Scraper**: Download and process policy documents from PolicyStat.
- **Vector Search**: Efficient retrieval using vector databases (FAISS, ChromaDB).
- **Integration with Ollama**: Run LLMs locally for privacy and speed.
- **Jupyter Notebooks**: Modular, step-by-step notebooks for learning and experimentation.

## Notebooks Included
- `RAG.ipynb` / `RAG-Backup.ipynb`: Main RAG workflow, vectorization, retrieval, and generation.
- `PolicyStat_Chatbot.ipynb`: Specialized chatbot for policy documents.
- `download_policies.ipynb`: Scraper for downloading policy documents from PolicyStat.
- `roboflow.ipynb`, `movie_recommendation.ipynb`, `LLM_quickstart.ipynb`, `datascience.ipynb`: Additional experiments and demos.

## Setup Instructions
1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd RAG
   ```
2. **Install Python dependencies**
   You can use pip to install the main dependencies. Run this in your terminal:
   ```bash
   pip install langchain langchain_community chromadb faiss-cpu matplotlib pandas numpy pydantic networkx transformers sentence-transformers cmake requests
   ```
   Some notebooks may require additional packages (e.g., `ultralytics`, `roboflow`, `opencv-python`, `Pillow`).

3. **Ollama Setup (for local LLMs)**
   - Download and install [Ollama](https://ollama.com/).
   - Start the Ollama server:
     ```bash
     ollama serve
     ```
   - Download a model (e.g., Mistral):
     ```bash
     ollama pull mistral
     ```

4. **Jupyter Notebook**
   - Launch JupyterLab or Jupyter Notebook:
     ```bash
     jupyter lab
     # or
     jupyter notebook
     ```
   - Open the desired notebook and follow the instructions in each cell.

## Usage
- Run `download_policies.ipynb` to scrape and download policy documents.
- Use `PolicyStat_Chatbot.ipynb` or `RAG.ipynb` to build and interact with your RAG chatbot.
- Modify the code to use your own documents or data sources as needed.

## Dependencies
- Python 3.8+
- langchain, langchain_community
- chromadb, faiss-cpu
- pandas, numpy, matplotlib, networkx
- transformers, sentence-transformers
- cmake, requests, pydantic
- (Optional) ultralytics, roboflow, opencv-python, Pillow
- Ollama (for local LLMs)

## Credits
- Built with [LangChain](https://github.com/langchain-ai/langchain), [Ollama](https://ollama.com/), and open-source Python libraries.
- Policy document scraping inspired by PolicyStat.

## License
This project is for educational and research purposes. Please check individual notebook headers for additional licensing or data usage notes.