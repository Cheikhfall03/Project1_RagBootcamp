# ZoomYourQuery – Chat with Your PDFs

This project implements a **Retrieval-Augmented Generation (RAG)** chatbot that lets you **upload a PDF** and **interact with their content** using **Tiny LLama LLM** and an **in-memory Chroma vector store**.

---

## 🚀 Features

- 📄 Upload and process **a PDF file**
- 🧠 Store document embeddings in **Chroma** (in-memory)
- 💬 Query with **TINY Llama LLM** using **RAG**
- 🔍 Inspect vector store chunks from the sidebar
- 🛠️ Modular, well-commented code for easy customization

---

## 🛠 Setup Instructions

### 1. Create a Virtual Environment

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```



## ▶️ Run the App

After installing dependencies, start the Streamlit app:

```bash
streamlit run index.py
```

The app will open in your default web browser at:


[Project1 RAG Bootcamp Demo](https://project1ragbootcamp.streamlit.app/)


If it doesn’t open automatically, copy and paste the URL from your terminal into your browser.

> 💡 Build your own custom RAG chatbot effortlessly!
