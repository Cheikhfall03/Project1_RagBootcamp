import os
import shutil
import pickle
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS

# Try to import different embedding options
try:
    from langchain_openai import OpenAIEmbeddings
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    import torch
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

# Persistent directory for FAISS vectorstore
PERSIST_DIR = "./faiss_store"
FAISS_INDEX_FILE = os.path.join(PERSIST_DIR, "index.faiss")
FAISS_PKL_FILE = os.path.join(PERSIST_DIR, "index.pkl")

# Force CPU usage and disable CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["NO_CUDA"] = "1"

def check_vectorstore_exists() -> bool:
    """
    Checks if the FAISS vectorstore exists in the persistent directory.
    
    Returns:
        bool: True if vectorstore exists, False otherwise
    """
    return (os.path.exists(FAISS_INDEX_FILE) and 
            os.path.exists(FAISS_PKL_FILE))

def reset_vectorstore():
    """Reset the vectorstore by removing the directory"""
    if os.path.exists(PERSIST_DIR):
        try:
            shutil.rmtree(PERSIST_DIR)
            print("🗑️ Vectorstore reset successfully")
        except Exception as e:
            print(f"⚠️ Could not reset vectorstore: {e}")

def create_embeddings():
    """Create embeddings with fallback options"""
    
    # Option 1: Try OpenAI embeddings (most reliable)
    if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
        try:
            print("🔄 Using OpenAI embeddings...")
            return OpenAIEmbeddings(model="text-embedding-3-small")
        except Exception as e:
            print(f"❌ OpenAI embeddings failed: {e}")
    
    # Option 2: Try HuggingFace embeddings with fixes
    if HUGGINGFACE_AVAILABLE:
        try:
            print("🔄 Using HuggingFace embeddings...")
            # Force CPU usage
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Create embeddings with minimal configuration
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={
                    'device': 'cpu',
                    'trust_remote_code': False,
                },
                encode_kwargs={
                    'normalize_embeddings': True,
                    'batch_size': 1
                }
            )
            print("✅ HuggingFace embeddings created successfully")
            return embeddings
            
        except Exception as e:
            print(f"❌ HuggingFace embeddings failed: {e}")
            
            # Try alternative smaller model
            try:
                print("🔄 Trying alternative HuggingFace model...")
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
                    model_kwargs={'device': 'cpu'}
                )
                print("✅ Alternative HuggingFace model loaded successfully")
                return embeddings
            except Exception as e2:
                print(f"❌ Alternative model also failed: {e2}")
    
    # Option 3: Use fake embeddings as last resort
    print("⚠️ Using fake embeddings for testing - not suitable for production!")
    from langchain_community.embeddings import FakeEmbeddings
    return FakeEmbeddings(size=384)

def store_chunks(chunks: List[Document]):
    """Stores document chunks in FAISS with embeddings."""
    try:
        # Create embeddings
        embeddings = create_embeddings()
        if embeddings is None:
            raise Exception("Failed to create any embeddings model")
        
        # Create the persist directory if it doesn't exist
        os.makedirs(PERSIST_DIR, exist_ok=True)
        
        # Check if vectorstore exists and is valid
        vectorstore_exists = check_vectorstore_exists()
        
        if vectorstore_exists:
            try:
                print("📂 Loading existing FAISS vectorstore...")
                vectorstore = FAISS.load_local(
                    PERSIST_DIR, 
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                # Add new documents to existing vectorstore
                vectorstore.add_documents(chunks)
                # Save the updated vectorstore
                vectorstore.save_local(PERSIST_DIR)
                print(f"✅ Added {len(chunks)} chunks to existing vectorstore")
            except Exception as e:
                print(f"❌ Error with existing vectorstore: {e}")
                print("🔄 Resetting and creating new vectorstore...")
                reset_vectorstore()
                os.makedirs(PERSIST_DIR, exist_ok=True)
                vectorstore = FAISS.from_documents(
                    documents=chunks,
                    embedding=embeddings
                )
                vectorstore.save_local(PERSIST_DIR)
                print(f"✅ Created new vectorstore with {len(chunks)} chunks")
        else:
            # Create new vectorstore
            print("🆕 Creating new FAISS vectorstore...")
            vectorstore = FAISS.from_documents(
                documents=chunks,
                embedding=embeddings
            )
            vectorstore.save_local(PERSIST_DIR)
            print(f"✅ Created new vectorstore with {len(chunks)} chunks")
        
        return vectorstore
        
    except Exception as e:
        print(f"❌ Error in store_chunks: {e}")
        # Try to reset and recreate
        try:
            print("🔄 Attempting to reset and recreate vectorstore...")
            reset_vectorstore()
            os.makedirs(PERSIST_DIR, exist_ok=True)
            embeddings = create_embeddings()
            vectorstore = FAISS.from_documents(
                documents=chunks,
                embedding=embeddings
            )
            vectorstore.save_local(PERSIST_DIR)
            print(f"✅ Successfully recreated vectorstore with {len(chunks)} chunks")
            return vectorstore
        except Exception as e2:
            print(f"❌ Final attempt failed: {e2}")
            raise e2

def get_bm25_retriever(chunks: List[Document]):
    """Initializes BM25Retriever with proper configuration."""
    try:
        if not chunks:
            print("⚠️ No chunks provided for BM25, creating empty retriever")
            # Create a dummy document for BM25 initialization
            dummy_doc = Document(page_content="dummy", metadata={})
            return BM25Retriever.from_documents([dummy_doc], k=4)
        return BM25Retriever.from_documents(chunks, k=4)
    except Exception as e:
        print(f"❌ Error creating BM25 retriever: {e}")
        return None

def get_vectorstore():
    """Loads existing FAISS vectorstore."""
    try:
        if not check_vectorstore_exists():
            print("❌ No existing vectorstore found")
            return None
            
        embeddings = create_embeddings()
        if embeddings is None:
            raise Exception("Failed to create embeddings model")
        
        vectorstore = FAISS.load_local(
            PERSIST_DIR, 
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Test the connection by checking the number of documents
        try:
            doc_count = vectorstore.index.ntotal
            print(f"✅ Loaded existing FAISS vectorstore with {doc_count} documents")
            return vectorstore
        except Exception as e:
            print(f"❌ Vectorstore connection test failed: {e}")
            print("🔄 Resetting corrupted vectorstore...")
            reset_vectorstore()
            return None
        
    except Exception as e:
        print(f"❌ Error loading vectorstore: {e}")
        return None